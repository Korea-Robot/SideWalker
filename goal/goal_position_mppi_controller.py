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
    crswalk_density=1, #1,
    object_density=0.01, #0.1,
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
    # random_spawn_lane_index=False,
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


######################## PD Controller #############################
import math

import time 
import numpy as np

def wrap_angle(angle):
    """[-pi, pi]로 감싸는 함수"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class MPPIController:
    def __init__(
        self,
        horizon=15,
        num_samples=500,
        lambda_=1.0,
        steering_gain=1.0,  # heading 변화 비례 상수
        v_max=2.0,          # 최고 전진 속도
        dt=0.1,
        alpha=1.0,          # 거리 가중치
        beta=2.0,           # 각도 오차 가중치
        gamma=0.1,          # 제어 정규화 가중치
        throttle_pref=0.6,  # throttle의 기본 선호값 (전진 유도)
        noise_cov=np.diag([0.5, 0.3])  # steer, throttle 노이즈 분산
    ):
        self.H = horizon
        self.N = num_samples
        self.lambda_ = lambda_
        self.steer_gain = steering_gain
        self.v_max = v_max
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.throttle_pref = throttle_pref

        self.noise_cov = noise_cov  # 2x2 covariance for [steer, throttle]
        # precompute inverse if needed for control cost
        self.noise_cov_inv = np.linalg.inv(noise_cov)
        # warm start sequence: shape (H, 2)
        self.prev_u = np.zeros((self.H, 2))
        # initialize throttle to a positive bias to move forward
        self.prev_u[:, 1] = throttle_pref

    def rollout(self, init_pos, init_theta, goal_pos, control_sequence):
        """
        하나의 제어 시퀀스로 H-step 예측 경로와 비용 계산
        Vectorized version will be outside; this is for clarity.
        Returns total cost.
        """
        x = init_pos[0]
        y = init_pos[1]
        theta = init_theta
        total_cost = 0.0

        for t in range(self.H):
            steer, throttle = control_sequence[t]
            # heading update
            theta = theta + steer * self.steer_gain * self.dt
            theta = wrap_angle(theta)
            # velocity
            v = throttle * self.v_max
            # position update
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt

            # cost terms
            pos = np.array([x, y])
            vec_to_goal = goal_pos - pos
            dist = np.linalg.norm(vec_to_goal)
            desired_heading = math.atan2(vec_to_goal[1], vec_to_goal[0])
            heading_error = wrap_angle(desired_heading - theta)

            distance_cost = self.alpha * dist
            heading_cost = self.beta * abs(heading_error)
            control_deviation = np.array([steer, throttle - self.throttle_pref])
            control_cost = self.gamma * (control_deviation @ control_deviation)

            total_cost += distance_cost + heading_cost + control_cost

        return total_cost

    def get_action(self, current_pos, current_theta, goal_pos):
        """
        현재 상태 기준으로 MPPI를 돌려서 [steer, throttle] 반환
        """
        # 1. 샘플링: 이전 optimal sequence + noise => (N, H, 2)
        noise = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=self.noise_cov,
            size=(self.N, self.H)
        )  # shape (N, H, 2)
        candidate_sequences = self.prev_u[np.newaxis, :, :] + noise  # broadcast to (N, H, 2)

        # optional: clamp steer and throttle to valid ranges
        candidate_sequences[..., 0] = np.clip(candidate_sequences[..., 0], -1.0, 1.0)  # steer
        candidate_sequences[..., 1] = np.clip(candidate_sequences[..., 1], -1.0, 1.0)  # throttle (allow small reverse if needed)

        # 2. Rollout all sequences vectorized
        # initialize arrays
        # positions and headings per sample
        pos = np.tile(np.array(current_pos), (self.N, 1))  # (N,2)
        theta = np.full((self.N,), current_theta)          # (N,)
        total_costs = np.zeros(self.N)

        for t in range(self.H):
            steer_t = candidate_sequences[:, t, 0]
            throttle_t = candidate_sequences[:, t, 1]

            theta = theta + steer_t * self.steer_gain * self.dt
            theta = wrap_angle(theta)
            v = throttle_t * self.v_max

            pos[:, 0] = pos[:, 0] + v * np.cos(theta) * self.dt
            pos[:, 1] = pos[:, 1] + v * np.sin(theta) * self.dt

            # cost components
            vec_to_goal = goal_pos[np.newaxis, :] - pos  # (N,2)
            dists = np.linalg.norm(vec_to_goal, axis=1)  # (N,)
            desired_heading = np.arctan2(vec_to_goal[:,1], vec_to_goal[:,0])  # (N,)
            heading_error = wrap_angle(desired_heading - theta)  # (N,)

            distance_cost = self.alpha * dists
            heading_cost = self.beta * np.abs(heading_error)
            control_deviation = np.stack([steer_t, throttle_t - self.throttle_pref], axis=1)  # (N,2)
            control_cost = self.gamma * np.sum(control_deviation**2, axis=1)  # (N,)

            total_costs += distance_cost + heading_cost + control_cost

        # 3. Weight 계산 (numerically stable)
        min_cost = np.min(total_costs)
        exp_term = np.exp(-(total_costs - min_cost) / self.lambda_)
        weights = exp_term / (np.sum(exp_term) + 1e-10)  # (N,)

        # 4. Optimal sequence 업데이트 (가중 평균)
        optimal_sequence = np.sum(candidate_sequences * weights[:, np.newaxis, np.newaxis], axis=0)  # (H,2)

        # 5. Shift for warm start
        next_prev = np.roll(optimal_sequence, -1, axis=0)
        next_prev[-1] = np.array([0.0, self.throttle_pref])  # 마지막은 default

        self.prev_u = next_prev  # 저장

        # 첫 스텝 제어 반환, 클램핑
        steer_cmd = float(np.clip(optimal_sequence[0, 0], -1.0, 1.0))
        throttle_cmd = float(np.clip(optimal_sequence[0, 1], -1.0, 1.0))

        return [steer_cmd, throttle_cmd]
    
import math
# MPPI controller 생성: 필요하다면 파라미터 튜닝
mppi = MPPIController(
    horizon=15,
    num_samples=500,
    lambda_=1.0,
    steering_gain=1.0,
    v_max=2.0,
    dt=0.1,
    alpha=1.0,
    beta=2.0,
    gamma=0.05,
    throttle_pref=0.5,
    noise_cov=np.diag([0.4, 0.2])
)




import cv2 

frames = [] 
def extract_obs(obs):
    global frames 
    # visualize image observation
    o_1 = obs["depth"][..., -1] # (H,W,1)
    o_1 = np.concatenate([o_1, o_1, o_1], axis=-1) # align channel # 3채널로 저장 혹은 아래 처럼  1채널 저장 
    # o_1 = 
    o_2 = obs["image"][..., -1] # (H,W,3)
    o_3 = obs["semantic"][..., -1] # (H,W,3)
    
    # breakpoint()


# --- 메인 실행 로직 ---

# 환경 및 Pygame 초기화
env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
pygame.init()
screen = pygame.display.set_mode((400, 150))
pygame.display.set_caption("Control Agent with WASD")
clock = pygame.time.Clock()

running = True

import random 

  # 5번째 웨이포인트를 목표로 설정


try:
    # 여러 에피소드 실행
    for i in range(10):
        obs,info = env.reset(seed=i + 2)
        
        
        waypoints = env.agent.navigation.checkpoints 
        print('wayppoint num: ',len(waypoints))
        
        reset = i+2000 
        while len(waypoints)<30:
            obs,info = env.reset(seed= reset)
            reset = random.randint(1,40000)
            waypoints = env.agent.navigation.checkpoints 
            print('i do not have sufficient waypoints ',i,' th')
            print(len(waypoints))
            
        num_waypoints = len(waypoints)
        k = 5
        
        # 에피소드 루프
        while running:
            # 기본 액션 (아무 키도 누르지 않았을 때)
            action = [0, 0]

            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # 키가 눌렸을 때 해당 키가 ACTION_MAP에 있는지 확인
                elif event.type == pygame.KEYDOWN and event.key in ACTION_MAP:
                    action = ACTION_MAP[event.key]
            
            if not running:
                break

            # --- 목표 지점 계산 (Egocentric) ---
            ego_goal_position = np.array([0.0, 0.0]) # 기본값 초기화
            nav = env.agent.navigation
            waypoints = nav.checkpoints
            
            # 웨이포인트가 충분히 있는지 확인

            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            
            # k 번째 waypoint의 ego coordinate 기준 좌표 
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

            # action = [1,0.3] # 왼쪽 주행. 
            # action = [-1,0.3] # 오른쪽 주행. 
            
            # action = [0,1]
            # action,k = update_action_goal(ego_goal_position,k)
            action = mppi.get_action(env.agent.position, env.agent.heading_theta, global_target)
            
            # ----------- 목표 웨이포인트 업데이트 ---------------- 
            # 목표지점까지 직선거리 계산 
            distance_to_target = np.linalg.norm(ego_goal_position)
            
            if distance_to_target< 5.0:
                k +=1
                if k>= num_waypoints:
                    k = num_waypoints-1
                    
            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            breakpoint()
            
            
            # Observation 설명 
            """
            (Pdb) obs.keys()
            dict_keys(['image', 'state', 'depth', 'semantic'])
            (Pdb) obs['state'].shape
            (273,)
            (Pdb) obs['image'].shape
            (160, 256, 3, 3)
            (Pdb) obs['depth'].shape
            (160, 256, 1, 3)
            (Pdb) obs['semantic'].shape
            (160, 256, 3, 3)
            """
            
            extract_obs(obs)
            
            print('ego goal position',ego_goal_position)
            print('action: ',action)
            
            print(distance_to_target)
            

            future_agent_pos = env.agent.position
            
            length = np.linalg.norm(future_agent_pos-agent_pos)
            
            if reward <0:
                print('crash!!')
                break 

            
            # 환경 렌더링 및 정보 표시
            env.render(
                text={
                    "Agent Position": np.round(env.agent.position, 2),
                    "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                    "Reward": f"{reward:.2f}",
                    "Ego Goal Position": np.round(ego_goal_position, 2)
                }
            )

            # 루프 속도 제어
            clock.tick(60)

            # 에피소드 종료 조건 확인
            if terminated or truncated:
                print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
                break
finally:
    # 종료 시 리소스 정리
    env.close()
    pygame.quit()



"""    
1.  **목표 지점 계산 로직 추가**: 메인 루프 안에서 `nav.checkpoints`를 가져와 마지막 웨이포인트를 목표 지점으로 설정하고, 계속 자기위치 기반으로 바로 앞에 가야할 위치를 업데이트 하면서 조종
2. PD controller를 통해  이동하도록 지시 
"""