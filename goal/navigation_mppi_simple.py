import numpy as np
import os
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math

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
class PD_Controller:
    def __init__(self,kp=0.3,kd=0.1,min_dt=0.1):
        """
        PID 제어기 초기화
            kp: 비례 상수
            ki: 적분 상수
            kd: 미분 상수
            setpoint: 목표치
            output_limit: 제어 출력의 최대/최소 한계 (anti-windup 적용)
            min_dt: 최소 시간 간격 (너무 작은 dt로 인한 미분 항 폭주 방지)
        """
        self.kp = kp 
        self.kd = kd 
        self.min_dt = min_dt
        self.last_error = 0.0 
        self.last_time = time.time()
    
    def update(self,measurement):
        """
        측정값(현재 오차)을 기반으로 제어 신호를 계산하고 상태를 업데이트합니다.
        :param measurement: 제어할 값 (에이전트 중심 좌표계에서의 목표 지점 y값, 즉 횡방향 오차)
        :return: 제어 신호 (조향값)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < self.min_dt:
            # derivative explode 방지
            dt = self.min_dt
        
        error = measurement # goal position of y (~=yaw)
        
        derivative = (error-self.last_error)/(dt+1e-9)
        
        pd_control = self.kp * error + self.kd * derivative
        
        print(' derivative',derivative)
        self.last_error = error 
        self.last_time = current_time
        
        # min max cut 
        pd_control = min(max(-1,pd_control),1)

        default_throttle =0.4
        
        return [pd_control,default_throttle]
        

pd_controller = PD_Controller(kp=0.2,kd=0.0)


import cv2 



"""
LiDAR 데이터 
obs['state'].shaep = (273,)

LidarStateObservation 클래스가 사용될 때의 일반적인 설정 기준입니다.

- 차량 자체 상태 (Ego State): 9개
- 주변 차량 정보 (Other Vehicles): 16개
- 내비게이션 정보 (Navigation): 8개
- 라이다 센서 정보 (Lidar Points): 240개

총합: 9 + 16 + 8 + 240 = 273개


## 1. 차량 자체 상태 (Ego State): 9개


        에이전트 차량 자신의 물리적 상태와 관련된 정보입니다.

        #	정보	개수	설명
        1	도로 경계 거리	2	좌측 및 우측 도로 경계선(보도블록 등)까지의 거리
        2	주행 방향 차이	1	현재 차선 방향과 차량의 진행 방향 사이의 각도 차이
        3	현재 속도	1	정규화된(0~1) 현재 차량 속도
        4	현재 조향각	1	정규화된(0~1) 현재 스티어링 휠의 각도
        5	이전 행동 (가속/조향)	2	바로 이전 스텝에서 AI가 내린 가속/브레이크 및 조향 값
        6	요 레이트 (Yaw Rate)	1	차량이 얼마나 빠르게 회전하고 있는지 나타내는 값
        7	차선 중앙 이탈 정도	1	현재 주행 중인 차선의 중앙으로부터 얼마나 벗어났는지


## 2. 주변 차량 정보 (Other Vehicles): 16개


        라이다로 감지된, 나와 가장 가까운 4대의 다른 차량에 대한 정보입니다.

        각 차량당 4개의 정보를 가집니다.

        상대적 종방향 거리: 나와의 앞뒤 거리

        상대적 횡방향 거리: 나와의 좌우 거리

        상대적 종방향 속도: 나와의 앞뒤 상대 속도

        상대적 횡방향 속도: 나와의 좌우 상대 속도

        계산: 4대 차량 * 차량당 4개 정보 = 16개



## 3. 내비게이션 정보 (Navigation): 8개


        목표 지점(체크포인트)까지의 경로 정보입니다.

        일반적으로 2개의 연속된 체크포인트에 대한 정보를 포함합니다.

        각 체크포인트당 4개의 정보로 구성될 수 있습니다.

        예: 목표까지의 전방/측면 거리, 해당 경로의 곡률(휘는 정도), 방향 등

        계산: 2개 체크포인트 * 4개 정보 = 8개


## 4. 라이다 센서 정보 (Lidar Points): 240개
        차량에 장착된 360도 라이다 센서의 원시 데이터입니다.

        240개의 레이저 빔이 각각 측정한 장애물까지의 거리를 나타냅니다.

        이 값들은 AI가 주변의 정적인 또는 동적인 장애물의 형태를 직접 파악하는 데 사용됩니다.

"""

import numpy as np
import math

# MPPI 컨트롤러 설정값
N_SAMPLES = 500      # 한 스텝마다 생성할 무작위 경로 샘플 개수
HORIZON = 20         # 예측할 미래 스텝의 길이
STEER_STD = 0.8      # 조향각 샘플링 시 추가할 노이즈의 표준편차
THROTTLE_STD = 1.0   # 가속/브레이크 샘플링 시 추가할 노이즈의 표준편차
LAMBDA = 1.0         # MPPI 가중치 계산에 사용되는 온도 파라미터
LIDAR_MAX_RANGE = 50 # 라이다 센서의 최대 측정 거리 (m), 정규화된 값을 실제 거리로 변환시 필요
COLLISION_COST_WEIGHT = 5000 # 충돌 비용의 가중치
GOAL_COST_WEIGHT = 1.0      # 목표 지향 비용의 가중치
AGENT_RADIUS = 1.5   # 에이전트의 안전 반경 (m)

class MPPI:
    def __init__(self, dt=0.1):
        self.dt = dt  # 시뮬레이션의 시간 간격 (env의 스텝 시간과 유사하게 설정)
        # 라이다 각도를 미리 계산하여 매번 다시 계산하는 것을 방지
        self.lidar_angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        # 이전 스텝의 최적 행동을 저장 (다음 스텝의 샘플링 기준으로 사용)
        self.last_optimal_action_sequence = np.zeros((HORIZON, 2))

    def _dynamics(self, state, action):
        """한 스텝 동안의 차량 움직임을 예측하는 동역학 모델"""
        x, y, theta, v = state
        steer, throttle = action

        # 간단한 자전거 모델 (Bicycle Model)
        new_theta = theta + v * np.tan(steer) / 2.5 * self.dt  # L=2.5 (차량 축간거리 가정)
        new_v = v + throttle * self.dt
        # new_v = max(0, new_v) # 속도는 0 이상
        new_v = np.maximum(0, new_v) # 속도는 0 이상
        
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        
        return np.array([new_x, new_y, new_theta, new_v])

    def _compute_costs(self, trajectories, lidar_cloud, goal):
        """
        예측된 경로들에 대한 비용을 계산합니다.
        라이다 데이터를 이용한 충돌 비용이 핵심입니다.
        """
        # 1. 목표 지향 비용 계산
        # 각 경로의 마지막 지점과 목표 지점 사이의 거리 계산
        final_points = trajectories[:, -1, :2] # (N_SAMPLES, 2)
        goal_cost = np.linalg.norm(final_points - goal, axis=1) * GOAL_COST_WEIGHT

        # 2. 충돌 비용 계산 (라이다 데이터 활용)
        collision_cost = np.zeros(N_SAMPLES)
        
        if lidar_cloud.shape[0] == 0: # 감지된 장애물이 없으면 충돌 비용은 0
            return goal_cost

        # 각 경로의 모든 지점과 모든 라이다 장애물 점 사이의 거리를 효율적으로 계산
        # trajectories: (N_SAMPLES, HORIZON, 4), lidar_cloud: (N_LIDAR_PTS, 2)
        # 결과 dists의 shape: (N_SAMPLES, HORIZON, N_LIDAR_PTS)
        dists = np.linalg.norm(trajectories[:, :, np.newaxis, :2] - lidar_cloud[np.newaxis, np.newaxis, :, :], axis=3)
        
        # 거리가 에이전트 반경보다 작은 경우 충돌로 간주
        is_collision = dists < AGENT_RADIUS
        
        # 한 경로에서 한 번이라도 충돌이 발생하면 높은 페널티 부여
        collision_per_sample = np.any(is_collision, axis=(1, 2))
        collision_cost[collision_per_sample] = COLLISION_COST_WEIGHT

        total_cost = goal_cost + collision_cost
        return total_cost

    def compute_action(self, current_v, lidar_data, goal_pos, prior_action):
        """최적의 행동을 계산하는 MPPI 메인 함수"""
        # 1. 라이다 원시 데이터를 2D 포인트 클라우드로 변환
        valid_lidar_indices = lidar_data < 1.0
        distances = lidar_data[valid_lidar_indices] * LIDAR_MAX_RANGE
        angles = self.lidar_angles[valid_lidar_indices]
        lidar_cloud = np.array([distances * np.cos(angles), distances * np.sin(angles)]).T

        # 2. 행동 샘플링
        # PD 컨트롤러가 제안한 행동 또는 이전 최적 행동을 기준으로 노이즈 추가
        base_action_sequence = np.roll(self.last_optimal_action_sequence, -1, axis=0)
        base_action_sequence[-1] = prior_action
        
        noise = np.random.normal(0, 1, (N_SAMPLES, HORIZON, 2))
        noise[:, :, 0] *= STEER_STD
        noise[:, :, 1] *= THROTTLE_STD
        sampled_actions = base_action_sequence[np.newaxis, :, :] + noise
        sampled_actions[:,:,0] = np.clip(sampled_actions[:,:,0], -1, 1) # 조향값 클리핑
        sampled_actions[:,:,1] = np.clip(sampled_actions[:,:,1], -1, 1) # 가속값 클리핑

        # 3. 경로 시뮬레이션
        initial_state = np.array([0, 0, 0, current_v]) # [x, y, theta, v], 에이전트 중심 좌표계
        trajectories = np.zeros((N_SAMPLES, HORIZON, 4))
        current_states = np.tile(initial_state, (N_SAMPLES, 1))

        for t in range(HORIZON):
            current_states = self._dynamics(current_states.T, sampled_actions[:, t, :].T).T
            trajectories[:, t, :] = current_states
            
        # 4. 비용 계산
        costs = self._compute_costs(trajectories, lidar_cloud, goal_pos)

        # 5. Path Integral 가중치 계산 및 최적 행동 결정
        beta = np.min(costs)
        weights = np.exp(-1.0 / LAMBDA * (costs - beta))
        weights /= np.sum(weights) + 1e-9 # 정규화

        # 가중 평균을 통해 최적 행동 시퀀스 계산
        optimal_action_sequence = np.sum(weights[:, np.newaxis, np.newaxis] * sampled_actions, axis=0)
        self.last_optimal_action_sequence = optimal_action_sequence

        # 현재 스텝에서 실행할 첫 번째 행동을 반환
        return optimal_action_sequence[0]
    




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

running = True

import random 

# mppi_controller = MPPI() # 기존 코드 대신 아래 코드로 변경
mppi_controller = MPPI(dt=1/1)

try:
    for i in range(10):
        obs, info = env.reset(seed=i + 2)
        
        waypoints = env.agent.navigation.checkpoints

        num_waypoints = len(waypoints)

        k = 5
        print('wayppoint num: ',len(waypoints))
        
        reset = i+2000 
        while len(waypoints)<30:
            obs,info = env.reset(seed= reset)
            reset = random.randint(1,40000)
            waypoints = env.agent.navigation.checkpoints 
            print('i do not have sufficient waypoints ',i,' th')
            print(len(waypoints))
            
        num_waypoints = len(waypoints)
        
        # 에피소드 루프
        while running:
            # --- 목표 지점 계산 (Egocentric) ---
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

            # --- MPPI를 위한 정보 추출 ---
            # obs['state']에서 라이다 데이터와 현재 속도 추출
            state_obs = obs['state']
            current_velocity = state_obs[3] # 정규화된 현재 속도 (실제 속도로 변환 필요 시 env.agent.speed 사용)
            lidar_data = state_obs[33:]     # (240,) 크기의 라이다 데이터

            # --- 행동 결정 ---
            # 1. PD 컨트롤러로 기본적인 주행 방향 제안 (MPPI 샘플링 가이드용)
            pd_action = pd_controller.update(ego_goal_position[1])

            # 2. MPPI 컨트롤러로 라이다 데이터를 고려한 최적 행동 계산
            # 입력: 현재 속도, 라이다 데이터, 에고좌표계 목표, PD 컨트롤러 제안 행동
            action = mppi_controller.compute_action(
                current_v=env.agent.speed, # 정규화되지 않은 실제 속도 사용
                lidar_data=lidar_data,
                goal_pos=ego_goal_position,
                prior_action=pd_action
            )
            
            # --- 목표 웨이포인트 업데이트 ---
            distance_to_target = np.linalg.norm(ego_goal_position)
            if distance_to_target < 5.0:
                k += 1
                if k >= num_waypoints:
                    k = num_waypoints - 1
            
            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward < -0.1: # 충돌 시 보상이 크게 감소하는 것을 이용
                print('crash!!')
            
            # 환경 렌더링
            env.render(
                text={
                    "Agent Position": np.round(env.agent.position, 2),
                    "Action": np.round(action, 2),
                    "Ego Goal Position": np.round(ego_goal_position, 2),
                    "Target Waypoint": k
                }
            )

            if terminated or truncated:
                print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
                break
finally:
    env.close()
