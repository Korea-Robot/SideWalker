import numpy as np
import os
import pygame
from metaurban.envs import SidewalkStaticMetaUrbanEnv
# from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math

from dataclasses import dataclass

# Action mapping: keyboard to [dx, dy]
ACTION_MAP = {
    pygame.K_w: [0, 1],
    pygame.K_s: [0, -0.5],
    pygame.K_a: [1, 0.5],
    pygame.K_d: [-1, 0.5]
}

# Render mode
render = not os.getenv('TEST_DOC')

# Utility: make a 3D line from a position
def make_line(x_offset, y_offset, height, y_dir=1, color=(1, 105/255, 180/255)):
    points = [(x_offset + x, x * y_dir + y_offset, height * x / 10 + height) for x in range(10)]
    colors = [np.clip(np.array([*color, 1]) * (i + 1) / 11, 0., 1.0) for i in range(10)]
    if y_dir < 0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors



# Environment Configuration
SENSOR_SIZE = (256, 160)

BASE_ENV_CFG = dict(
    use_render=True,
    map='X', 
    manual_control=False, 
    crswalk_density=1, 
    object_density=0.1, 
    walk_on_all_regions=False,
    drivable_area_extension=55, 
    height_scale=1, 
    horizon=100,  # Long horizon              # 이기능이 종료까지의 T의 길이의 말하는것이다. 이게 짧아져도 checkoint, polyline는 똑같음. 값도 똑같음. 정해져잇음.
    
    vehicle_config=dict(enable_reverse=True), # 후진 가능 매우 중요
    
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    
    # scenario setting
    random_spawn_lane_index=False,
    num_scenarios=100,                       # 이건 뭘뜻하는걸까? 아 100번 시나리오 부터 시작한다는뜻
    accident_prob=0,
    # relax_out_of_road_done=True,
    max_lateral_dist=5.0,    
    
    agent_type = 'coco', #['whellcahir']
    
    relax_out_of_road_done=False,  # More strict termination
    # max_lateral_dist=10.0,  # Larger tolerance
    
    agent_observation=ThreeSourceMixObservation,
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)


# Initialize environment
env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
obs = env.reset()

drawer = env.engine.make_line_drawer(thickness=5) # create a line drawer



# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("Control Agent with WASD")
clock = pygame.time.Clock()


def compute_action(relative_position, max_steer=1.0, max_throttle=1.0):
    x, y = relative_position

    # y축 기준 방향 (즉, heading 기준)
    angle = math.atan2(x, y)  # 왼쪽(-), 오른쪽(+)
    steer = -np.clip(angle / (math.pi / 4), -max_steer, max_steer)

    distance = np.linalg.norm([x, y])
    throttle = np.clip(distance / 5.0, 0.0, max_throttle)

    # 목표가 뒤에 있으면 속도 감소 또는 후진
    if y < 0:
        throttle *= -0.5

    return [steer, 1]

import numpy as np
import math

def compute_action(relative_position, max_steer=1.0, max_throttle=1.0):
    """
    에이전트 기준 상대 위치를 [조향, 가감속] action으로 변환합니다.
    """
    local_x, local_y = relative_position

    # 1. 조향(Steer) 계산
    # 목표 지점까지의 각도를 계산합니다. atan2의 결과는 -pi ~ pi 입니다.
    # 이 각도를 기준으로 얼마나 핸들을 꺾을지 결정합니다.
    angle_to_target = math.atan2(local_x, local_y)
    
    # 조향 값은 보통 -1과 1 사이로 정규화합니다.
    # angle_to_target / (math.pi / 4)는 목표가 45도 벗어났을 때 최대 조향을 하도록 만듭니다.
    # - 부호는 시뮬레이터의 조향 방향에 따라 조정될 수 있습니다.
    steer = -np.clip(angle_to_target / (math.pi / 4), -max_steer, max_steer)

    # 2. 가속/감속(Throttle) 계산
    # 목표까지의 거리를 계산합니다.
    distance = np.linalg.norm([local_x, local_y])
    
    # 거리에 비례하여 속도를 결정합니다. 5.0은 비례 상수로, 이 값을 조정해 반응성을 바꿀 수 있습니다.
    throttle = np.clip(distance / 5.0, 0.0, max_throttle)

    # 목표가 뒤에 있다면 (local_y < 0) 브레이크 또는 후진을 위해 음수 값을 적용합니다.
    if local_y < 0:
        throttle = -0.5 # 일정한 속도로 후진하도록 설정

    return [steer, throttle]


import math

def get_angle(x, y):
    angle = math.atan2(y, x)  # 결과는 -pi ~ pi 범위
    return angle


running = True
try:
    for i in range(10):
        env.reset(seed=i+1)
        

        # 60개 처음과 끝이 polyline 과 같음 -2한개가 빠져있음. 왜???

        while running:

            nav = env.agent.navigation
            # navigation에 관련된 상태정보
            
            polyline = nav.reference_trajectory.get_polyline()
            # 앞으로 가야할 상태정보 (절대 위치 기준임)
            
            waypoints = nav.checkpoints  
            if len(waypoints)<10:
                break
            
            # action = [0, 1]  # default no-op

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in ACTION_MAP:
                    action = ACTION_MAP[event.key]

            # Step the environment with selected action

            # Draw polylines every few steps
            
            agent_pos = env.agent.position 
            # 현재 위치 (절대 위치기준임 )
            # == polyline[0], waypoints[0]
            
            print('agent pos : ',agent_pos)
            print('waypoints : ',waypoints[0], ' & ',waypoints[-1])
            
            print('start point : ',nav.start_points) # 다른 좌표계인가? 암튼 값이 다름
            # print('end point : ',nav.end_points)
            print('end :', nav.reference_trajectory.end) #  = waypoints[-1]
            
            
            
            print('polyline : ',polyline[0],polyline[-1]) # waypoints와 동치.
            # print('polyline : ',])        

            print('nav compeltion ', nav.route_completion) # 진행률 표시
            print(len(waypoints),len(polyline)) ### 300개일때와 아닐때 .상대 좌표로 항상 target을 넣어주자.
            
            print(env.agent.out_of_route) # Boolean
            
            print(env.agent.crash_vehicle,env.agent.crash_sidewalk) # Boolean
            print()
            # world coordinate 기준.
            # 매시점 target_pos 에 대한 위치를 구하고 그걸 heading theta를 통해 다시 로봇 좌표계로 변환한다.
            target_pos = waypoints[5]-agent_pos 
            
            if target_pos[0] ==0:
                target_pos[0] = 1e-5
                
            
            # 일단 지금 절대 좌표계 기준으로 내 target position이 존재하고 있다.
            # target position 방향이 존재할것이다. vector  : 절대 좌표계 기준 
            # 또한 로봇이 보고 있는 방향이 존재할것이다. vector : 절대 좌표계 기준
            
            # 그러면 로봇과 목표지점에 대한 각도(-pi ~ pi)를 계산 할 수있다.
            # Cylindrical Coordinate, Orthogonal coordinate?
            
            pi = math.atan2(target_pos[1],target_pos[0])  # 결과는 -pi ~ pi 범위
            print(' Pi !!!!!!!!!!' ,pi)
            heading = env.agent.heading_theta
            theta = pi - heading

            
            print('heading!!!!! : ',heading)
            print('target!!', target_pos)
            rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            
            
            relative_position = rotation_matrix@target_pos
            
            # breakpoint()            
            # print()

            clock.tick(10)  # Control loop speed
            
            
            ##############################
            # --- 수정된 코드 ---  ##########
            ##############################
            
            
            # 월드 좌표계 기준 에이전트의 현재 위치와 진행 방향
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta

            # 월드 좌표계 기준 에이전트에서 목표 지점까지의 벡터
            k = 5
            target_pos_world = waypoints[k] - agent_pos

            # 월드 좌표를 -> 에이전트 좌표로 변환하는 올바른 회전 행렬
            # 에이전트 heading의 "음수" 각도를 사용합니다.
            cos_h = np.cos(-agent_heading)
            sin_h = np.sin(-agent_heading)
            rotation_matrix = np.array([
                [cos_h, -sin_h],
                [sin_h, cos_h]
            ])

            # 회전을 적용하여 에이전트 기준 상대 위치를 얻습니다.
            relative_position = rotation_matrix @ target_pos_world

            # 이제 relative_position = [상대x, 상대y] 가 됩니다.
            # 상대x: 왼쪽(+), 오른쪽(-) 거리
            # 상대y: 앞(+), 뒤(-) 거리

            # 이제 이 상대 위치를 컨트롤러에 올바르게 사용할 수 있습니다.
            action = compute_action(relative_position)
            
            # if abs(theta) > 0.2:
            #     action = [1,1]
            # else:
            #     action = [0,1]
                
            # action = [1,1] 
            # action = [-1,1]
            obs, reward, tm,tc, info = env.step(action)

            if tm or tc:
                break
        
finally:
    env.close()
    pygame.quit()


"""
지금 해야할것은 이제 목표 지점 에대한 goal position을 정확하게 먼저 정의해서 넣어주는것.

cirriculum Learning : 학습 시 거리 증가

그후로는 이제 bad reward, 도로로 간다든지. 인도 벗어난다든지에 대한 정보를 얻을 수있어야하고 그를 통해 reward를 만들어주는것.
먼저 reward코드를 살펴보자.

"""