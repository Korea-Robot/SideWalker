# 필요한 라이브러리들을 가져옵니다.
import numpy as np  # 수치 계산, 특히 행렬 및 벡터 연산을 위해 사용
import os           # 운영체제 관련 기능을 사용하기 위해 (여기서는 사용되지 않음)
import pygame       # 사용자 입력(키보드) 및 창 생성을 위해 사용
import math         # 수학 함수(cos, sin, degrees 등)를 사용하기 위해

# MetaUrban 환경 및 관련 구성 요소들을 가져옵니다.
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 설정 (Configurations) ---

# 키보드 입력을 에이전트의 행동으로 변환하는 딕셔너리입니다.
# 형식: [조향(steer), 가속/감속(throttle)]
# 조향: 양수=좌회전, 음수=우회전
# 가속: 양수=전진, 음수=후진/브레이크
ACTION_MAP = {
    pygame.K_w: [0, 1.0],   # 'W' 키: 직진
    pygame.K_s: [0, -1.0],  # 'S' 키: 후진 또는 브레이크
    pygame.K_a: [0.5, 0.5], # 'A' 키: 좌회전하며 전진
    pygame.K_d: [-0.5, 0.5]  # 'D' 키: 우회전하며 전진
}

# 시뮬레이션 환경에 대한 기본 설정값들을 정의합니다.
SENSOR_SIZE = (256, 160) # 카메라 센서의 해상도 (너비, 높이)
BASE_ENV_CFG = dict(
    # --- 기본 렌더링 및 맵 설정 ---
    use_render=True,            # 시뮬레이션 화면을 렌더링할지 여부
    map='X',                    # 'X'자 모양의 교차로 맵 사용
    manual_control=False,       # 내장된 수동 조작 모드를 비활성화 (Pygame으로 직접 제어)
    crswalk_density=1,          # 횡단보도 밀도
    object_density=0.1,         # 동적 객체(차량 등) 밀도
    walk_on_all_regions=False,  # 모든 지역을 보행 가능하게 할지 여부
    drivable_area_extension=55, # 주행 가능 영역 확장
    height_scale=1,             # 높이 스케일
    horizon=1000,               # 한 에피소드의 최대 길이 (타임스텝)
    
    # --- 에이전트(차량) 설정 ---
    vehicle_config=dict(enable_reverse=True), # 후진 기능 활성화
    
    # --- 시각적 요소 설정 ---
    show_sidewalk=True,         # 인도 표시
    show_crosswalk=True,        # 횡단보도 표시
    random_lane_width=True,     # 차선 폭 랜덤화
    random_agent_model=True,    # 에이전트 모델 랜덤화
    random_lane_num=True,       # 차선 수 랜덤화
    
    # --- 시나리오 및 종료 조건 설정 ---
    random_spawn_lane_index=False, # 시작 차선을 랜덤하게 할지 여부
    num_scenarios=100,             # 생성할 시나리오의 수
    accident_prob=0,               # 사고 발생 확률
    max_lateral_dist=5.0,          # 경로에서 최대로 벗어날 수 있는 거리
    
    agent_type='coco',             # 에이전트의 종류 ('coco' 또는 'wheelchair')
    
    relax_out_of_road_done=False,  # 경로를 벗어났을 때 즉시 종료할지 (엄격한 조건)
    
    # --- 관측(Observation) 및 센서 설정 ---
    agent_observation=ThreeSourceMixObservation, # 여러 센서 데이터를 조합하여 관측
    image_observation=True,                      # 이미지 데이터를 관측에 포함
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50, # 로그 레벨 (50은 에러 메시지만 표시)
)

# --- 유틸리티 함수 (Utility Functions) ---

def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """
    월드(전역) 좌표계의 목표 지점을 에이전트 중심의 자기(egocentric) 좌표계로 변환합니다.
    자기 중심 좌표계는 에이전트의 현재 위치와 방향을 기준으로 합니다.

    :param global_target_pos: 월드 좌표계에서의 목표 지점 [x, y]
    :param agent_pos: 월드 좌표계에서의 에이전트 위치 [x, y]
    :param agent_heading: 에이전트의 현재 진행 방향 (라디안 단위)
    :return: 에이전트 기준 상대 위치 [x, y]. x: 좌(-)/우(+) 거리, y: 전(+)/후(-) 거리
    """
    # 1. 월드 좌표계에서 에이전트로부터 목표 지점까지의 벡터를 계산합니다.
    vec_in_world = global_target_pos - agent_pos

    # 2. 에이전트의 heading의 "음수" 각도를 사용하여 회전 변환을 준비합니다.
    #    월드 좌표계에서 에이전트 좌표계로 바꾸려면, 에이전트의 heading만큼 반대로 회전해야 합니다.
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    # 2D 회전 행렬을 정의합니다.
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])

    # 3. 회전 행렬을 벡터에 적용하여 에이전트 중심 좌표계의 벡터를 얻습니다.
    ego_vector = rotation_matrix @ vec_in_world
    
    return ego_vector


# --- 메인 실행 로직 ---

# MetaUrban 환경을 설정값으로 초기화합니다.
env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)

# Pygame을 초기화하고, 키보드 입력을 받을 작은 창을 생성합니다.
pygame.init()
screen = pygame.display.set_mode((400, 150))
pygame.display.set_caption("Control Agent with WASD")
clock = pygame.time.Clock() # 프레임 속도 제어를 위한 시계 객체

running = True # 메인 루프를 제어하는 플래그
try:
    # 10개의 다른 에피소드를 실행합니다.
    for i in range(10):
        # 새로운 시드(seed)로 환경을 리셋합니다. 시드를 고정하면 항상 같은 맵이 생성됩니다.
        env.reset(seed=i + 1)
        
        # 하나의 에피소드에 대한 메인 루프입니다.
        while running:
            # 기본 행동을 [0, 0] (정지)으로 설정합니다.
            action = [0, 0]

            # Pygame 이벤트 큐를 처리합니다.
            for event in pygame.event.get():
                # 창 닫기 버튼을 누르면 루프를 종료합니다.
                if event.type == pygame.QUIT:
                    running = False
                # 키가 눌렸을 때, 해당 키가 ACTION_MAP에 정의되어 있는지 확인합니다.
                elif event.type == pygame.KEYDOWN and event.key in ACTION_MAP:
                    action = ACTION_MAP[event.key]
            
            # running 플래그가 False가 되면 즉시 루프를 탈출합니다.
            if not running:
                break

            # --- 목표 지점 계산 (자기 중심 좌표계) ---
            ego_goal_position = np.array([0.0, 0.0]) # 기본값으로 초기화
            nav = env.agent.navigation
            waypoints = nav.checkpoints # 에이전트가 따라가야 할 경로점들
            
            # 경로점(waypoints)이 15개 이상 있을 때만 목표 지점을 계산합니다.
            if len(waypoints) > 15:
                k = 15  # 15번째 경로점을 목표로 설정
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                # 월드 좌표계의 목표를 자기 중심 좌표계로 변환합니다.
                ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)


            # 결정된 행동(action)으로 환경을 한 스텝 진행시킵니다.
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 환경을 렌더링하고, 화면 좌측 상단에 디버깅 정보를 표시합니다.
            env.render(
                text={
                    "Agent Position": np.round(env.agent.position, 2),
                    "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                    "Reward": f"{reward:.2f}",
                    "Ego Goal Position": np.round(ego_goal_position, 2)
                }
            )

            # 루프의 속도를 초당 60프레임으로 제한합니다.
            clock.tick(60)

            # 에피소드 종료 조건을 확인합니다. (목표 도달, 충돌, 시간 초과 등)
            if terminated or truncated:
                print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
                break # 현재 에피소드를 종료하고 다음 에피소드로 넘어갑니다.
finally:
    # 프로그램 종료 시, 환경과 Pygame 리소스를 안전하게 정리합니다.
    env.close()
    pygame.quit()