"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
import cv2
import os
import numpy as np
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.obs.image_obs import ImageObservation, ImageStateObservation
import argparse
import torch

def make_metadrive_env_fn(env_cfg):
    env = SidewalkStaticMetaUrbanEnv(dict(
        log_level=50,
        **env_cfg,
    ))
    env = Monitor(env)
    return env

# if __name__ == "__main__":

map_type = 'X'
config = dict(
    crswalk_density=1,
    object_density=0.01,
    use_render=True,
    walk_on_all_regions=False,
    map=map_type,
    manual_control=True,
    drivable_area_extension=55,
    height_scale=1,
    spawn_deliveryrobot_num=2,
    show_mid_block_map=False,
    show_ego_navigation=False,
    debug=False,
    horizon=300,
    on_continuous_line_done=False,
    out_of_route_done=True,
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=True,
        show_line_to_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=True,
        policy_reverse=False,
    ),
    show_sidewalk=True,
    show_crosswalk=True,
    random_spawn_lane_index=False,
    num_scenarios=200000,
    accident_prob=0,
    window_size=(1200, 900),
    relax_out_of_road_done=True,
    max_lateral_dist=1e10,
    
    camera_dist = 0.8,
    camera_height = 1.5,
    camera_pitch = None,
    camera_fov = 66,
    norm_pixel=False,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="saved_imgs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

config.update(
        dict(
            image_observation=True,
            sensors=dict(
                # Perspective-view cameras
                rgb_camera=(RGBCamera,640, 360),
                depth_camera=(DepthCamera, 640, 360),
                semantic_camera=(SemanticCamera, 640,360),
                
                ####### ADDED #######
                # Top-down semantic camera
                # 해상도는 필요에 따라 조절 (e.g., 512, 512)
                top_down_semantic=(SemanticCamera, 512, 512) 
            ),
            agent_observation=ThreeSourceMixObservation,
            interface_panel=[]
        )
    )

# 개선된 데이터셋 수집기 import
from world_dataset import ImitationDatasetCollector
# 데이터셋 수집기 초기화
collector = ImitationDatasetCollector("imitation_dataset")


episode = 1 
env = SidewalkStaticMetaUrbanEnv(config)
try:
    for i in range(100000):
        episode +=1
        
        # 일정 waypoints 이상 없으면 다시 환경 구성 .
        obs,info = env.reset(seed=episode)
        
        waypoints = env.agent.navigation.checkpoints 
        print('waypoint num: ',len(waypoints))

        # 웨이포인트가 충분히 있는지 확인
        while len(waypoints)<30:
            episode+=1
            obs,info = env.reset(seed= episode)
            waypoints = env.agent.navigation.checkpoints 
            print('NOT sufficient waypoints ',episode,' !!!!',len(waypoints))
            
        num_waypoints = len(waypoints)

        # *** 새로운 에피소드 시작 - 이 부분이 누락되어 있었습니다! ***
        collector.start_new_episode(waypoints)

        ######################################################
        # ----- Semantic Map을 통한 Planner & Controller ----- 
        ######################################################

        time_interval = 2
        scenario_t = 0
        
        # 데이터 관측 
    
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):
            obs, reward, tm, tc, info = env.step(action)

            if scenario_t % time_interval == 0:
                # ===== 1. Perspective View 데이터 취득 (기존 코드) =====
                
                
                
                # ===== RGB  =====
                camera = env.engine.get_sensor("rgb_camera")
                
                front  = -3
                height = 1.5
                rgb_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                )
                max_rgb_value = rgb_front.max()
                rgb = rgb_front[..., ::-1]
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)
                
                
                # ===== DEPTH =====
                camera = env.engine.get_sensor("depth_camera")
                depth_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                ).reshape(360, 640, -1)[..., -1] 
                
                # depth_front에 대한 분포는 지금 203~255로 되어있음. 그래서 데이터를 한번 측정해보고 그대로 저장하되 나중에 normalize 필요함.
                # 한장한장하면 너무 상대적이라 위험할지도?
                
                # depth_normalized = depth_front
                
                depth_normalized = cv2.normalize(depth_front, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_img = cv2.bitwise_not(depth_front)[..., None]
                

                
                # ===== SEMANTIC =====
                camera = env.engine.get_sensor("semantic_camera")
                semantic_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                )
                semantic = (semantic_front[..., ::-1] * 255).astype(np.uint8)
                
                
                
                # ===== 2. Top-down View 데이터 취득 ####### ADDED ####### =====
                # Top-down 카메라 센서 가져오기
                top_down_camera = env.engine.get_sensor("top_down_semantic")
                
                # 카메라 위치와 방향 설정
                # position: 에이전트 바로 위 (높이는 커버할 영역에 따라 조절)
                # hpr: Heading, Pitch, Roll. Pitch를 -90으로 하여 바닥을 보게 함
                top_down_semantic_map = top_down_camera.perceive(
                    new_parent_node=env.agent.origin,
                    position=[0, 0, 10], # (x, y, z) - z 값을 조절해 보이는 범위를 설정
                    hpr=[0, -90, 0]      # (heading, pitch, roll)
                )
                
                tdsm = top_down_semantic_map 
                
                
                top_down_semantic_map = (top_down_semantic_map[..., ::-1] * 255).astype(np.uint8)

                
                
                # ===== 3. 이미지 저장 (Perspective + Top-down) =====
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_rgb_perspective.png"), rgb[..., ::-1])
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic_perspective.png"), semantic[..., ::-1])
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_depth_colored_perspective.png"), depth_colored)
                
                ####### ADDED #######
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic_topdown.png"), top_down_semantic_map)
                
                # breakpoint()
                
                k = 7
                #  goal position k step 앞에를 목표로 둔다.
                waypoints = env.agent.navigation.checkpoints 
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                ap = agent_pos
                # print(ap)
                
                # 1. Dynamics Model 
                
                
            scenario_t += 1

            if (tm or tc):
                env.reset(env.current_seed + 1)
                action = [0., 0.]
                scenario_t = 0
finally:
    # 종료 시 리소스 정리
    print('semantic mppi imitation data collect done!')
    env.close()

def collect_data():
    
    return tdsm
    pass