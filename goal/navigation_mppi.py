import numpy as np
import os
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math
import time
import random
import cv2

"""
python# 성능 vs 품질 트레이드오프
self.N = 500      # 샘플 수 (증가시 품질↑, 속도↓)
self.H = 10       # 예측 지평선 (증가시 미래 예측↑, 계산량↑)

# 제어 민감도
alpha=2.0         # 가속 응답성
beta=2.0          # 조향 응답성
lambda_=0.5       # 탐색 vs 활용 균형 (작을수록 탐욕적)

# 안전 거리
min_safe_distance = 4.0      # 장애물 안전 거리
emergency_distance = 2.0     # 긴급 제동 거리
"""

# 환경 설정
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True,
    map='X',
    manual_control=False,
    crswalk_density=1,
    object_density=0.01,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=1000,  # 에피소드 최대 길이
    
    vehicle_config=dict(enable_reverse=True), # 후진 기능 활성화
    
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    
    # 시나리오 설정
    random_spawn_lane_index=True,
    num_scenarios=100000,
    accident_prob=0,
    max_lateral_dist=5.0,
    
    agent_type='coco', # 에이전트 타입
    
    relax_out_of_road_done=False, # 경로 이탈 시 종료 조건 강화
    
    agent_observation=ThreeSourceMixObservation,
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50, # 로그 레벨 (50은 에러만 표시)
)

# --- 유틸리티 함수 ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """
    월드 좌표계의 목표 지점을 에이전트 중심의 자기(egocentric) 좌표계로 변환합니다.
    """
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])

    ego_vector = rotation_matrix @ vec_in_world
    return ego_vector

def extract_obs(obs):
    """관찰값에서 이미지 데이터 추출"""
    o_1 = obs["depth"][..., -1] # (H,W,1)
    o_1 = np.concatenate([o_1, o_1, o_1], axis=-1)
    o_2 = obs["image"][..., -1] # (H,W,3)
    o_3 = obs["semantic"][..., -1] # (H,W,3)
    return o_1, o_2, o_3

# MPPI Controller 클래스
class MPPI:
    def __init__(self, alpha=2.0, beta=2.0, dt=0.1, lambda_=1.0):
        """
        MPPI 컨트롤러 초기화
        
        Args:
            alpha: 가속도 제어 게인
            beta: 조향 제어 게인  
            dt: 시간 스텝
            lambda_: 온도 매개변수 (작을수록 더 탐욕적)
        """
        self.alpha = alpha
        self.beta = beta 
        self.dt = dt
        self.lambda_ = lambda_
        
        # MPPI 파라미터
        self.N = 500  # 샘플링할 궤적 수 (성능을 위해 줄임)
        self.H = 10   # 예측 지평선 (시간 스텝)
        
        # 제어 입력 제한
        self.u_min = np.array([-1.0, -0.5])  # [steering, throttle] 최소값
        self.u_max = np.array([1.0, 1.0])    # [steering, throttle] 최대값
        
        # 노이즈 공분산
        self.sigma = np.array([[0.4, 0.0], [0.0, 0.3]])  # [steering_noise, throttle_noise]
        
        # 이전 제어 시퀀스 저장 (warm start)
        self.prev_u_seq = np.zeros((self.H, 2))
        # 기본 제어 입력으로 초기화
        self.prev_u_seq[:, 1] = 0.4  # 기본 속도
    
    def dynamics_model(self, state, action):
        """
        차량 동역학 모델 (bicycle model 근사)
        
        Args:
            state: [x, y, theta, velocity, angular_velocity]
            action: [steering, throttle]
            
        Returns:
            next_state: 다음 상태
        """
        x, y, theta, velocity, angular_vel = state
        steering, throttle = action
        
        # 차량 동역학 업데이트
        next_x = x + np.cos(theta) * velocity * self.dt
        next_y = y + np.sin(theta) * velocity * self.dt
        next_theta = theta + angular_vel * self.dt
        next_velocity = velocity + self.alpha * throttle * self.dt
        next_angular_vel = angular_vel * 0.8 + self.beta * steering * self.dt  # 감쇠 추가
        
        # 속도와 각속도 제한
        next_velocity = np.clip(next_velocity, -2.0, 8.0)
        next_angular_vel = np.clip(next_angular_vel, -1.5, 1.5)
        
        return np.array([next_x, next_y, next_theta, next_velocity, next_angular_vel])
    
    def rollout_dynamics(self, initial_state, u_sequences):
        """
        N개의 제어 시퀀스에 대해 H 스텝 동안 궤적을 시뮬레이션
        
        Args:
            initial_state: 초기 상태 [x, y, theta, velocity, angular_velocity]
            u_sequences: (N, H, 2) 제어 시퀀스들
            
        Returns:
            trajectories: (N, H+1, 5) 궤적들
        """
        N = u_sequences.shape[0]
        trajectories = np.zeros((N, self.H + 1, 5))
        
        # 모든 궤적의 초기 상태 설정
        trajectories[:, 0, :] = initial_state
        
        # 각 시간 스텝에 대해 동역학 시뮬레이션
        for t in range(self.H):
            for n in range(N):
                trajectories[n, t+1, :] = self.dynamics_model(
                    trajectories[n, t, :], 
                    u_sequences[n, t, :]
                )
        
        return trajectories
    
    def calculate_collision_risk(self, x, y, theta, lidar_data):
        """
        라이다 데이터를 이용한 충돌 위험도 계산
        
        Args:
            x, y, theta: 차량 위치와 방향
            lidar_data: (240,) 라이다 거리 데이터
            
        Returns:
            risk: 충돌 위험도 (0~1)
        """
        # 차량 전방과 측면의 라이다 빔들만 고려
        front_indices = list(range(110, 130))  # 전방 20도 범위
        left_indices = list(range(90, 110))    # 좌측
        right_indices = list(range(130, 150))  # 우측
        
        # 예상 차량 위치에서의 장애물까지 거리 추정
        min_safe_distance = 4.0
        emergency_distance = 2.0
        risk = 0.0
        
        # 전방 위험도 (가중치 높음)
        front_distances = lidar_data[front_indices]
        front_min_dist = np.min(front_distances)
        
        if front_min_dist < emergency_distance:
            risk += 5.0  # 긴급 상황
        elif front_min_dist < min_safe_distance:
            risk += 2.0 * (min_safe_distance - front_min_dist) / min_safe_distance
        
        # 측면 위험도 - 차량이 해당 방향으로 이동할 때만 고려
        predicted_y = y + np.sin(theta) * 2.0  # 2초 후 예상 위치
        
        if predicted_y > 0.5:  # 좌측으로 이동
            left_distances = lidar_data[left_indices]
            left_min_dist = np.min(left_distances)
            if left_min_dist < min_safe_distance/2:
                risk += 1.0 * (min_safe_distance/2 - left_min_dist) / (min_safe_distance/2)
        
        if predicted_y < -0.5:  # 우측으로 이동
            right_distances = lidar_data[right_indices]
            right_min_dist = np.min(right_distances)
            if right_min_dist < min_safe_distance/2:
                risk += 1.0 * (min_safe_distance/2 - right_min_dist) / (min_safe_distance/2)
        
        return np.clip(risk, 0.0, 5.0)
    
    def cost_function(self, trajectories, lidar_data, goal_position):
        """
        궤적들에 대한 비용 계산
        
        Args:
            trajectories: (N, H+1, 5) 궤적들
            lidar_data: (240,) 라이다 데이터
            goal_position: [x, y] 목표 위치 (ego 좌표계)
            
        Returns:
            costs: (N,) 각 궤적의 비용
        """
        N = trajectories.shape[0]
        costs = np.zeros(N)
        
        for n in range(N):
            traj = trajectories[n]
            cost = 0.0
            
            for t in range(1, self.H + 1):
                x, y, theta, vel, angular_vel = traj[t]
                
                # 1. 목표 추적 비용 (가장 중요)
                goal_distance = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)
                cost += 15.0 * goal_distance
                
                # 2. 목표 방향 비용 (목표를 향해 가도록)
                goal_angle = np.arctan2(goal_position[1] - y, goal_position[0] - x)
                angle_diff = abs(theta - goal_angle)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # 각도 정규화
                cost += 5.0 * angle_diff
                
                # 3. 전진 장려 비용
                forward_velocity = vel * np.cos(theta)
                if forward_velocity > 0:
                    cost -= 3.0 * forward_velocity
                else:
                    cost += 5.0 * abs(forward_velocity)  # 후진 페널티
                
                # 4. 장애물 회피 비용
                collision_risk = self.calculate_collision_risk(x, y, theta, lidar_data)
                cost += 100.0 * collision_risk
                
                # 5. 차선 중앙 유지 비용 (y=0 근처 유지)
                cost += 8.0 * y**2
                
                # 6. 제어 입력 부드러움 비용
                if t > 1:
                    prev_angular_vel = traj[t-1, 4]
                    angular_vel_change = abs(angular_vel - prev_angular_vel)
                    cost += 2.0 * angular_vel_change**2
                
                # 7. 속도 유지 비용 (너무 느리면 안됨)
                if vel < 1.0:
                    cost += 3.0 * (1.0 - vel)**2
            
            # 8. 최종 목표 도달 보상
            final_x, final_y = traj[-1, 0], traj[-1, 1]
            final_goal_distance = np.sqrt((final_x - goal_position[0])**2 + (final_y - goal_position[1])**2)
            cost += 20.0 * final_goal_distance
            
            costs[n] = cost
        
        return costs
    
    def sample_control_sequences(self):
        """
        제어 시퀀스들을 샘플링 (가우시안 노이즈 + warm start)
        
        Returns:
            u_sequences: (N, H, 2) 제어 시퀀스들
        """
        u_sequences = np.zeros((self.N, self.H, 2))
        
        for n in range(self.N):
            for t in range(self.H):
                # 이전 해에 노이즈를 추가 (warm start)
                if t < len(self.prev_u_seq):
                    base_u = self.prev_u_seq[t]
                else:
                    base_u = np.array([0.0, 0.4])  # 기본값: 직진, 적당한 속도
                
                # 가우시안 노이즈 추가
                noise = np.random.multivariate_normal([0, 0], self.sigma)
                u_sequences[n, t] = np.clip(base_u + noise, self.u_min, self.u_max)
        
        return u_sequences
    
    def path_integral_update(self, costs, u_sequences):
        """
        Path Integral을 이용한 제어 시퀀스 업데이트
        
        Args:
            costs: (N,) 각 궤적의 비용
            u_sequences: (N, H, 2) 제어 시퀀스들
            
        Returns:
            optimal_u_seq: (H, 2) 최적 제어 시퀀스
        """
        # 비용을 음수로 변환하고 정규화 (낮은 비용 = 높은 가중치)
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # 가중 평균으로 최적 제어 시퀀스 계산
        optimal_u_seq = np.zeros((self.H, 2))
        for t in range(self.H):
            optimal_u_seq[t] = np.sum(weights[:, np.newaxis] * u_sequences[:, t, :], axis=0)
        
        return optimal_u_seq
    
    def optimize(self, current_state, lidar_data, goal_position):
        """
        MPPI 최적화 수행
        
        Args:
            current_state: [x, y, theta, velocity, angular_velocity]
            lidar_data: (240,) 라이다 데이터
            goal_position: [x, y] 목표 위치 (ego 좌표계)
            
        Returns:
            optimal_action: [steering, throttle] 최적 제어 입력
            proximal_goal: [x, y] 단기 목표 위치
        """
        # 1. 제어 시퀀스 샘플링
        u_sequences = self.sample_control_sequences()
        
        # 2. 동역학 시뮬레이션
        trajectories = self.rollout_dynamics(current_state, u_sequences)
        
        # 3. 비용 계산
        costs = self.cost_function(trajectories, lidar_data, goal_position)
        
        # 4. Path Integral 업데이트
        optimal_u_seq = self.path_integral_update(costs, u_sequences)
        
        # 5. 다음 반복을 위해 최적 시퀀스 저장 (shift + padding)
        self.prev_u_seq[:-1] = optimal_u_seq[1:]
        self.prev_u_seq[-1] = optimal_u_seq[-1]  # 마지막 제어 입력으로 패딩
        
        # 6. 최적 궤적에서 단기 목표 위치 추출 (k번째 시점)
        best_traj_idx = np.argmin(costs)
        best_trajectory = trajectories[best_traj_idx]
        
        k = min(3, self.H)  # 3 스텝 앞 또는 지평선 끝
        proximal_goal = best_trajectory[k, :2]  # [x, y]만 추출
        
        return optimal_u_seq[0], proximal_goal

# MPPI 컨트롤러 초기화
mppi_controller = MPPI(alpha=2.0, beta=2.0, dt=0.1, lambda_=0.5)

# --- 메인 실행 로직 ---
def main():
    # 환경 초기화
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    running = True

    try:
        # 여러 에피소드 실행
        for episode in range(10):
            print(f"\n=== Episode {episode + 1} ===")
            obs, info = env.reset(seed=episode + 2)
            
            # 충분한 웨이포인트가 있는지 확인
            waypoints = env.agent.navigation.checkpoints 
            print(f'Waypoint num: {len(waypoints)}')
            
            reset_seed = episode + 2000 
            while len(waypoints) < 30:
                obs, info = env.reset(seed=reset_seed)
                reset_seed = random.randint(1, 40000)
                waypoints = env.agent.navigation.checkpoints 
                print(f'Insufficient waypoints, trying seed {reset_seed}')
                
            num_waypoints = len(waypoints)
            k = 5  # 목표 웨이포인트 인덱스
            step_count = 0
            max_steps = 500
            
            # 에피소드 루프
            while running and step_count < max_steps:
                step_count += 1
                
                if not running:
                    break

                # --- 목표 지점 계산 (Egocentric) ---
                nav = env.agent.navigation
                waypoints = nav.checkpoints
                
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                
                # k 번째 waypoint의 ego coordinate 기준 좌표 
                ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
                
                # 라이다 데이터 추출
                lidar_data = obs['state'][33:]  # shape = (240,)
                
                # 현재 상태 구성 (ego coordinate 기준)
                current_velocity = np.linalg.norm(obs['state'][1:3])  # 속도 크기
                current_angular_vel = obs['state'][6] if len(obs['state']) > 6 else 0.0  # 각속도
                
                current_state = np.array([
                    0.0,  # ego coordinate에서 자신의 x는 항상 0
                    0.0,  # ego coordinate에서 자신의 y는 항상 0
                    0.0,  # ego coordinate에서 자신의 heading은 항상 0
                    current_velocity,
                    current_angular_vel
                ])
                
                # MPPI 최적화 수행
                try:
                    optimal_action, proximal_goal = mppi_controller.optimize(
                        current_state, lidar_data, ego_goal_position)
                    
                    # 최적 액션 사용
                    action = optimal_action.tolist()
                    
                except Exception as e:
                    print(f"MPPI optimization failed: {e}")
                    # 폴백: 단순한 목표 추적
                    steering = np.clip(ego_goal_position[1] * 0.5, -1.0, 1.0)
                    throttle = 0.3
                    action = [steering, throttle]
                
                # ----------- 목표 웨이포인트 업데이트 ---------------- 
                distance_to_target = np.linalg.norm(ego_goal_position)
                
                if distance_to_target < 8.0:  # 목표 도달 임계값
                    k += 1
                    if k >= num_waypoints:
                        k = num_waypoints - 1
                        print("Final waypoint reached!")
                        
                # 환경 스텝 실행
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 관찰값 처리
                extract_obs(obs)
                
                # 디버그 정보 출력
                if step_count % 20 == 0:  # 20스텝마다 출력
                    print(f'Step {step_count}: Goal distance = {distance_to_target:.2f}, '
                          f'Action = [{action[0]:.2f}, {action[1]:.2f}], Reward = {reward:.2f}')
                
                # 충돌이나 실패 체크
                if reward < -5.0:
                    print('Collision or major penalty detected!')
                    break 
                
                # 환경 렌더링 및 정보 표시
                env.render(
                    text={
                        "Episode": f"{episode + 1}",
                        "Step": f"{step_count}",
                        "Agent Position": np.round(env.agent.position, 2),
                        "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                        "Reward": f"{reward:.2f}",
                        "Ego Goal Position": np.round(ego_goal_position, 2),
                        "Distance to Goal": f"{distance_to_target:.2f}",
                        "Current Waypoint": f"{k}/{num_waypoints}",
                        "Action": f"[{action[0]:.2f}, {action[1]:.2f}]"
                    }
                )

                # 에피소드 종료 조건 확인
                if terminated or truncated:
                    print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
                    break
                    
            print(f"Episode {episode + 1} completed after {step_count} steps")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # 종료 시 리소스 정리
        env.close()

if __name__ == "__main__":
    main()