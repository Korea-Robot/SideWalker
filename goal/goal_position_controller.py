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
            action = pd_controller.update(ego_goal_position[1])
            
            # ----------- 목표 웨이포인트 업데이트 ---------------- 
            # 목표지점까지 직선거리 계산 
            distance_to_target = np.linalg.norm(ego_goal_position)
            
            if distance_to_target< 5.0:
                k +=1
                if k>= num_waypoints:
                    k = num_waypoints-1
                    
            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            
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

            # 에피소드 종료 조건 확인
            if terminated or truncated:
                print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
                break
finally:
    # 종료 시 리소스 정리
    env.close()



"""    
1.  **목표 지점 계산 로직 추가**: 메인 루프 안에서 `nav.checkpoints`를 가져와 마지막 웨이포인트를 목표 지점으로 설정하고, 계속 자기위치 기반으로 바로 앞에 가야할 위치를 업데이트 하면서 조종
2. PD controller를 통해  이동하도록 지시 
"""