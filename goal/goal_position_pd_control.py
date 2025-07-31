import numpy as np
import os
import pygame
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math

# --- 설정 ---

# 키보드 액션 매핑: [조향, 가속/브레이크]
ACTION_MAP = {
    pygame.K_w: [0, 1.0],   # 전진
    pygame.K_s: [0, -1.0],  # 후진/브레이크
    pygame.K_a: [0.5, 0.5], # 좌회전
    pygame.K_d: [-0.5, 0.5]  # 우회전
}

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
    random_spawn_lane_index=False,
    num_scenarios=100,
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

# --- PD Controller 클래스 ---

class PDController:
    def __init__(self, kp_steering=2.0, kd_steering=0.5, kp_speed=1.0, kd_speed=0.2, 
                 target_speed=0.8, max_steering=1.0, max_throttle=1.0):
        """
        PD Controller for vehicle navigation
        
        Args:
            kp_steering: Proportional gain for steering
            kd_steering: Derivative gain for steering  
            kp_speed: Proportional gain for speed
            kd_speed: Derivative gain for speed
            target_speed: Target driving speed
            max_steering: Maximum steering angle
            max_throttle: Maximum throttle value
        """
        self.kp_steering = kp_steering
        self.kd_steering = kd_steering
        self.kp_speed = kp_speed
        self.kd_speed = kd_speed
        self.target_speed = target_speed
        self.max_steering = max_steering
        self.max_throttle = max_throttle
        
        # Previous errors for derivative calculation
        self.prev_lateral_error = 0.0
        self.prev_speed_error = 0.0
        
        # Waypoint tracking
        self.current_waypoint_idx = 0
        self.waypoint_reached_threshold = 3.0  # Distance threshold to consider waypoint reached
        
    def get_current_target_waypoint(self, waypoints, agent_pos):
        """
        Find the current target waypoint based on agent position
        
        Args:
            waypoints: List of waypoints
            agent_pos: Current agent position
            
        Returns:
            Target waypoint position and index
        """
        if len(waypoints) == 0:
            return None, -1
            
        # Find the closest waypoint ahead of current position
        min_distance = float('inf')
        target_idx = self.current_waypoint_idx
        
        # Look ahead from current waypoint
        for i in range(self.current_waypoint_idx, min(len(waypoints), self.current_waypoint_idx + 10)):
            distance = np.linalg.norm(waypoints[i] - agent_pos)
            if distance < min_distance:
                min_distance = distance
                target_idx = i
        
        # If current waypoint is reached, move to next one
        if min_distance < self.waypoint_reached_threshold and target_idx < len(waypoints) - 1:
            self.current_waypoint_idx = min(target_idx + 5, len(waypoints) - 1)  # Look ahead 5 waypoints
            target_idx = self.current_waypoint_idx
            
        return waypoints[target_idx], target_idx
    
    def compute_action(self, waypoints, agent_pos, agent_heading, agent_velocity):
        """
        Compute steering and throttle action using PD control
        
        Args:
            waypoints: List of navigation waypoints
            agent_pos: Current agent position [x, y]
            agent_heading: Current agent heading in radians
            agent_velocity: Current agent velocity
            
        Returns:
            action: [steering, throttle] where steering is [-1, 1] and throttle is [-1, 1]
        """
        if len(waypoints) == 0:
            return [0.0, 0.0]
        
        # Get current target waypoint
        target_waypoint, waypoint_idx = self.get_current_target_waypoint(waypoints, agent_pos)
        
        if target_waypoint is None:
            return [0.0, 0.0]
        
        # Convert target to egocentric coordinates
        ego_target = convert_to_egocentric(target_waypoint, agent_pos, agent_heading)
        
        # Calculate lateral error (how far left/right from target)
        lateral_error = ego_target[0]  # x component in ego coordinates
        
        # Calculate longitudinal distance to target
        longitudinal_distance = ego_target[1]  # y component in ego coordinates
        
        # --- Steering Control (PD) ---
        # Proportional term
        steering_p = self.kp_steering * lateral_error
        
        # Derivative term
        lateral_error_derivative = lateral_error - self.prev_lateral_error
        steering_d = self.kd_steering * lateral_error_derivative
        
        # Combined steering command
        steering = -(steering_p + steering_d)  # Negative because positive ego x means steer right
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        # --- Speed Control (PD) ---
        # Current speed (magnitude of velocity)
        current_speed = np.linalg.norm(agent_velocity) if hasattr(agent_velocity, '__len__') else abs(agent_velocity)
        
        # Adjust target speed based on steering magnitude (slow down in turns)
        adjusted_target_speed = self.target_speed * (1.0 - 0.3 * abs(steering))
        
        # Speed error
        speed_error = adjusted_target_speed - current_speed
        
        # Proportional term
        throttle_p = self.kp_speed * speed_error
        
        # Derivative term  
        speed_error_derivative = speed_error - self.prev_speed_error
        throttle_d = self.kd_speed * speed_error_derivative
        
        # Combined throttle command
        throttle = throttle_p + throttle_d
        throttle = np.clip(throttle, -self.max_throttle, self.max_throttle)
        
        # Store previous errors for next iteration
        self.prev_lateral_error = lateral_error
        self.prev_speed_error = speed_error
        
        return [steering, throttle]
    
    def reset(self):
        """Reset controller state for new episode"""
        self.prev_lateral_error = 0.0
        self.prev_speed_error = 0.0
        self.current_waypoint_idx = 0

# --- 유틸리티 함수 ---

def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """
    월드 좌표계의 목표 지점을 에이전트 중심의 자기(egocentric) 좌표계로 변환합니다.

    :param global_target_pos: 월드 좌표계에서의 목표 지점 [x, y]
    :param agent_pos: 월드 좌표계에서의 에이전트 위치 [x, y]
    :param agent_heading: 에이전트의 현재 진행 방향 (라디안)
    :return: 에이전트 기준 상대 위치 [x, y]. x: 좌/우, y: 전/후
    """
    # 1. 월드 좌표계에서 에이전트로부터 목표 지점까지의 벡터 계산
    vec_in_world = global_target_pos - agent_pos

    # 2. 에이전트의 heading의 "음수" 각도를 사용하여 회전 변환
    # 월드 좌표계에서 에이전트 좌표계로 바꾸려면, 에이전트의 heading만큼 반대로 회전해야 함
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])

    # 3. 회전 행렬을 적용하여 에이전트 중심 좌표계의 벡터를 얻음
    ego_vector = rotation_matrix @ vec_in_world
    
    return ego_vector


# --- 메인 실행 로직 ---

# 환경 및 Pygame 초기화
env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
pygame.init()
screen = pygame.display.set_mode((400, 150))
pygame.display.set_caption("PD Controller - Control Agent with WASD or Auto")
clock = pygame.time.Clock()

# PD Controller 초기화
pd_controller = PDController(
    kp_steering=1.5,    # 조향 비례 게인
    kd_steering=0.3,    # 조향 미분 게인
    kp_speed=1.2,       # 속도 비례 게인
    kd_speed=0.1,       # 속도 미분 게인
    target_speed=0.6,   # 목표 속도
    max_steering=1.0,   # 최대 조향각
    max_throttle=1.0    # 최대 스로틀
)

running = True
auto_mode = True  # 자동 모드 (PD Controller 사용)

try:
    # 여러 에피소드 실행
    for episode in range(10):
        obs, info = env.reset(seed=episode + 1)
        pd_controller.reset()  # 컨트롤러 상태 리셋
        
        waypoints = env.agent.navigation.checkpoints 
        
        if len(waypoints) < 30:
            obs, info = env.reset()
            print(f'Episode {episode}: Insufficient waypoints, resetting...')
            continue
            
        print(f"Episode {episode + 1}: Starting with {len(waypoints)} waypoints")
        step_count = 0
        
        # 에피소드 루프
        while running:
            # 기본 액션 (아무 키도 누르지 않았을 때)
            action = [0, 0]
            
            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # 스페이스바로 자동/수동 모드 전환
                        auto_mode = not auto_mode
                        print(f"Mode switched to: {'Auto (PD Controller)' if auto_mode else 'Manual'}")
                    elif not auto_mode and event.key in ACTION_MAP:
                        action = ACTION_MAP[event.key]
            
            if not running:
                break

            # --- PD Controller를 사용한 액션 계산 ---
            if auto_mode:
                # 현재 에이전트 상태 가져오기
                agent_pos = np.array(env.agent.position)
                agent_heading = env.agent.heading_theta
                agent_velocity = np.array(env.agent.velocity) if hasattr(env.agent, 'velocity') else [0, 0]
                
                # PD Controller로 액션 계산
                action = pd_controller.compute_action(waypoints, agent_pos, agent_heading, agent_velocity)
            
            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # 현재 목표 waypoint 계산 (시각화용)
            agent_pos = np.array(env.agent.position)
            agent_heading = env.agent.heading_theta
            current_target, waypoint_idx = pd_controller.get_current_target_waypoint(waypoints, agent_pos)
            
            if current_target is not None:
                ego_goal_position = convert_to_egocentric(current_target, agent_pos, agent_heading)
            else:
                ego_goal_position = np.array([0.0, 0.0])
            
            # 환경 렌더링 및 정보 표시
            env.render(
                text={
                    "Mode": "Auto (PD)" if auto_mode else "Manual (WASD)",
                    "Episode": f"{episode + 1}/10",
                    "Step": step_count,
                    "Waypoint": f"{waypoint_idx + 1}/{len(waypoints)}" if waypoint_idx >= 0 else "N/A",
                    "Agent Position": np.round(env.agent.position, 2),
                    "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f}°",
                    "Action [Steer, Throttle]": np.round(action, 3),
                    "Reward": f"{reward:.2f}",
                    "Ego Goal Position": np.round(ego_goal_position, 2),
                    "Distance to Goal": f"{np.linalg.norm(ego_goal_position):.2f}m",
                    "Controls": "SPACE: Auto/Manual, WASD: Manual control"
                }
            )

            # 루프 속도 제어
            clock.tick(60)

            # 에피소드 종료 조건 확인
            if terminated or truncated:
                success = waypoint_idx >= len(waypoints) - 5  # 마지막 몇 개 waypoint 근처에 도달하면 성공
                print(f"Episode {episode + 1} finished after {step_count} steps.")
                print(f"Result: {'SUCCESS' if success else 'FAILED'} - Reached waypoint {waypoint_idx + 1}/{len(waypoints)}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print("-" * 50)
                break
                
finally:
    # 종료 시 리소스 정리
    env.close()
    pygame.quit()


"""    
PD Controller 구현 설명:

1. **PDController 클래스**:
   - 조향(steering)과 속도(speed) 제어를 위한 별도의 PD 제어기
   - kp, kd 게인을 통해 제어 성능 조정 가능
   - waypoint 추적 로직 포함

2. **주요 기능**:
   - `get_current_target_waypoint()`: 현재 목표 waypoint 선택
   - `compute_action()`: PD 제어를 통한 액션 계산
   - 조향: 좌우 위치 오차 기반 제어
   - 속도: 목표 속도와 현재 속도 차이 기반 제어

3. **제어 로직**:
   - 측면 오차(lateral error)를 통한 조향 제어
   - 목표 속도 추적을 통한 가속/브레이크 제어
   - 급회전 시 속도 자동 감속

4. **사용법**:
   - 스페이스바: 자동(PD)/수동(WASD) 모드 전환
   - 자동 모드에서 waypoint를 따라 자동 주행
   - 수동 모드에서 WASD로 직접 제어

5. **튜닝 가능한 파라미터**:
   - kp_steering, kd_steering: 조향 반응성 조정
   - kp_speed, kd_speed: 속도 제어 조정
   - target_speed: 주행 속도 설정
   - waypoint_reached_threshold: waypoint 도달 판정 거리
"""