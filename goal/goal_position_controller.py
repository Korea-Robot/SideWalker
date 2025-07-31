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
    crswalk_density=0.001, #1,
    object_density=0.001, #0.1,
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
pygame.display.set_caption("Control Agent with WASD")
clock = pygame.time.Clock()

running = True

import random 

try:
    # 여러 에피소드 실행
    for i in range(10):
        obs,info = env.reset(seed=i + 1)
        
        waypoints = env.agent.navigation.checkpoints 
        
        print(len(waypoints))
        
        while len(waypoints)<30:
            obs,info = env.reset(seed= i)
            i = random.randint()
            print('i do not have sufficient waypoints ',i,' th')
            
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
            k = 5  # 5번째 웨이포인트를 목표로 설정
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            
            # k 번째 waypoint의 ego coordinate 기준 좌표 
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

            # action = [1,0.3] # 왼쪽 주행. 
            # action = [-1,0.3] # 오른쪽 주행. 
            
            action = [0,1]
            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(ego_goal_position)
            
            print(length = math.sqrt([x**2 for x in ego_goal_position].sum()))
            
            
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


import math 
def PD_controller(ego_goal_position, k):

    action = [0,0]
    length = math.sqrt([x**2 for x in ego_goal_position].sum())
    if length < 3:
        update_k = k+5
    return action , update_k
"""    
1.  **목표 지점 계산 로직 추가**: 메인 루프 안에서 `nav.checkpoints`를 가져와 마지막 웨이포인트를 목표 지점으로 설정하고, 계속 자기위치 기반으로 바로 앞에 가야할 위치를 업데이트 하면서 조종
2. PD controller를 통해  이동하도록 지시 
"""