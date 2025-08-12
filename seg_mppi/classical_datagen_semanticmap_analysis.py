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


from utils import PD_Controller,convert_to_egocentric
## PD controller 생성
pd_controller = PD_Controller(kp=0.2,kd=0.0) # 제일 안정적임을 확인 

import cv2, numpy as np, csv

# HSV 임계 (OpenCV: H 0~179)
YELLOW_LO, YELLOW_HI = (15, 80, 80), (35, 255, 255)
PINK1_LO,  PINK1_HI  = (160, 60, 60), (179, 255, 255)  # 상위 분홍
PINK2_LO,  PINK2_HI  = (0,   60, 60), (10,  255, 255)  # 하위 분홍

def _label_from_hsv(hsv):
    h,s,v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    is_y = (YELLOW_LO[0] <= h <= YELLOW_HI[0] and YELLOW_LO[1] <= s <= YELLOW_HI[1] and YELLOW_LO[2] <= v <= YELLOW_HI[2])
    in1  = (PINK1_LO[0]  <= h <= PINK1_HI[0]  and PINK1_LO[1]  <= s <= PINK1_HI[1]  and PINK1_LO[2]  <= v <= PINK1_HI[2])
    in2  = (PINK2_LO[0]  <= h <= PINK2_HI[0]  and PINK2_LO[1]  <= s <= PINK2_HI[1]  and PINK2_LO[2]  <= v <= PINK2_HI[2])
    if is_y: return "yellow"
    if (in1 or in2): return "pink"
    return "other"

def analyze_tdsm_colors(tdsm_bgr_uint8, top_k=20, save_csv_path=None):
    """
    tdsm_bgr_uint8: BGR uint8 이미지(픽셀 전수 체크)
    top_k: 상위 N개만 출력
    save_csv_path: 경로 주면 CSV 저장
    반환: rows(list of dict)
    """
    H,W,_ = tdsm_bgr_uint8.shape
    flat  = tdsm_bgr_uint8.reshape(-1,3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)  # 정확한 고유색 집계
    order = np.argsort(-counts)
    colors, counts = colors[order], counts[order]
    total = int(counts.sum())

    rows = []
    for i in range(len(colors)):
        bgr = colors[i].astype(np.uint8)
        hsv = cv2.cvtColor(bgr.reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]
        label = _label_from_hsv(hsv)
        rows.append({
            "rank": i+1,
            "B": int(bgr[0]), "G": int(bgr[1]), "R": int(bgr[2]),
            "H": int(hsv[0]), "S": int(hsv[1]), "V": int(hsv[2]),
            "count": int(counts[i]),
            "percent": float(100.0 * counts[i] / total),
            "label": label
        })

    # 콘솔 출력(상위 top_k)
    for r in rows[:top_k]:
        print(f"{r['rank']:02d}  {r['percent']:5.1f}%  "
              f"BGR=({r['B']},{r['G']},{r['R']})  "
              f"HSV=({r['H']},{r['S']},{r['V']})  {r['label']}")

    # CSV 저장(옵션)
    if save_csv_path:
        with open(save_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    return rows



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
    
    camera_dist = -2,
    camera_height = 2.2,
    camera_pitch = None,
    camera_fov = 90,
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


running = True

front  = -2
height = 2.2

episode = 3
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
        print(num_waypoints)
        
        print()
        print('start new episode!!')
        print()
        # *** 새로운 에피소드 시작 - 이 부분이 누락되어 있었습니다! ***
        collector.start_new_episode(waypoints)

        ######################################################
        # ----- Semantic Map을 통한 Planner & Controller ----- 
        ######################################################

        time_interval = 2
        scenario_t = 0
        
        # 데이터 관측 
    
        print(HELP_MESSAGE)
        # camera = env.engine.get_sensor("rgb_camera")
        # depth_camera = env.engine.get_sensor("depth_camera")
        # semantic_camera = env.engine.get_sensor("semantic_camera")

        # Top-down 카메라 센서 가져오기
        top_down_camera = env.engine.get_sensor("top_down_semantic")

        # 5번째 웨이포인트를 목표로 설정
        k = 5
        reward = 0
        # 만약에 끼어서 계속 가만히 있는경우 제거하기 위해서.
        start_position = env.agent.position
        stuck_interval = 10
        
        while running:
            if scenario_t % time_interval == 0:
                
                # ===== RGB  =====
                # rgb_front = camera.perceive(
                #     to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                # )
                # max_rgb_value = rgb_front.max()
                # rgb = rgb_front[..., ::-1]

                    
                rgb = obs["image"][..., -1]
                max_rgb_value = rgb.max()
                
                
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)
                
                # ===== DEPTH =====
                # depth_front = depth_camera.perceive(
                #     to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                # ).reshape(360, 640, -1)[..., -1] 

                # o_1 = np.concatenate([o_1, o_1, o_1], axis=-1) # align channel
                depth_front = obs["depth"][..., -1]

                # depth_front에 대한 분포는 지금 203~255로 되어있음. 그래서 데이터를 한번 측정해보고 그대로 저장하되 나중에 normalize 필요함.
                # 한장한장하면 너무 상대적이라 위험할지도?
                
                # depth_normalized = depth_front
                
                depth_normalized = cv2.normalize(depth_front, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_img = cv2.bitwise_not(depth_front)[..., None]
                
                
                # ===== SEMANTIC =====
                # semantic_front = semantic_camera.perceive(
                #     to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, front, height], hpr=[0, 0, 0]
                # )
                semantic_front = obs["semantic"][..., -1]
                semantic = (semantic_front[..., ::-1] * 255).astype(np.uint8)
                
                
                # ===== 2. Top-down View 데이터 취득 ####### ADDED ####### =====
                
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
                
                # tdsm_bgr가 BGR uint8 이미지라면:
                
                rows = analyze_tdsm_colors(top_down_semantic_map, top_k=30,
                                        save_csv_path=os.path.join(args.out_dir, "tdsm_colors.csv"))
                breakpoint()
                
                # ===== 3. 이미지 저장 (Perspective + Top-down) =====
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_rgb_perspective.png"), rgb[..., ::-1])
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic_perspective.png"), semantic[..., ::-1])
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_depth_colored_perspective.png"), depth_colored)
                
                # ###### ADDED #######
                # cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic_topdown.png"), top_down_semantic_map)
                
                # breakpoint()
                
                #  goal position k step 앞에를 목표로 둔다.
                waypoints = env.agent.navigation.checkpoints 
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                ap = agent_pos
                
                # k 번째 waypoint의 ego coordinate 기준 좌표 
                ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

                # ----------- 목표 웨이포인트 업데이트 ---------------- 
                # 목표지점까지 직선거리 계산 
                distance_to_target = np.linalg.norm(ego_goal_position)
                
                # 목표 지점 업데이트
                if distance_to_target< 5.0:
                    k +=1
                    if k>= num_waypoints:
                        k = num_waypoints-1

                # 에이전트 상태 정보 수집
                agent_state = {
                    "position": env.agent.position,
                    "heading": env.agent.heading_theta,
                    "velocity": env.agent.speed,
                    "angular_velocity": getattr(env.agent, 'angular_velocity', 0.0)
                }
                
                print('goal : ',ego_goal_position)
                print(k)
                # print(agent_state)
                
                action = pd_controller.update(ego_goal_position[1]) # yaw방향에 대해서만 추측함. throttle은 고정 
                
                # 데이터 수집
                collector.collect_sample(
                    obs, action, agent_state, ego_goal_position, reward, scenario_t
                )   
                
                #### Planning & Controller 
                
                # Method 1 : A* + PD controller
                
                # Method 2 : MPPI (model predictive path integral)
                
                # 1. Dynamics Model 
                
                
                # tdsm과 ego_goal_position 이용해서 planner를 만들어야함.
                # 여기서 top-down semantic map에서 중앙에는 로봇이 존재함. 
                # 그리고 장애물은 전부 분홍색이 아님. 
                # 따라서 가려는 goal_position 쪽에 분홍색이 아닌 semantic 색깔이 존재하면 그것은 장애물임.
                # 먼저 semantic의 rgb 표현을 확인해봐야한다. 
                
                
                # global target을 위에 a star waypoint로 변환 .
                
                
                
                
                
                
                
                
                
                
                
                
            obs, reward, tm, tc, info = env.step(action)
            
            scenario_t +=1

            # 에피소드 종료 조건
            if tm or tc or scenario_t >= 800 or reward <0:
                episode_info = {
                    "seed": episode,
                    "terminated": tm,
                    "truncated": tc,
                    "crashed": reward < 0,
                    "episode_length": scenario_t,
                    "success": bool(np.linalg.norm(waypoints-env.agent.position) < 1)
                }


                # 에이전트 상태 정보 수집
                agent_state = {
                    "position": env.agent.position,
                    "heading": env.agent.heading_theta,
                    "velocity": env.agent.speed,
                }
                
                # 데이터 수집
                collector.collect_sample(
                    obs, action, agent_state, ego_goal_position, reward, scenario_t
                )  
                
                break
        print(f"Episode completed with {scenario_t} steps")


        # 에피소드 하나 종료
        # try:
        collector.finish_episode(episode_info)
        
        # 각 에피소드의 길이가 64 이상이면 주기적으로 저장
        if scenario_t > 64: 
            # 주기적으로 데이터셋 정보 저장 (매 10 에피소드마다)
            if collector.episode_counter % 10 == 0:
                collector.save_dataset_info()
                collector.create_train_test_split()
                
                print(f"\nDataset update - Episode {collector.episode_counter}:")
                print(f"  Total episodes: {collector.episode_counter}")
                print(f"  Total samples: {collector.dataset_info['total_samples']}")
                print(f"  Dataset saved to: {collector.dataset_root}")
        else:
            print(f"Episode {episode} too short ({scenario_t} steps), skipping...")
            
        # except Exception as e:
        #     print(f"Error saving episode {episode}: {e}")
        #     continue
        
finally:
    # 종료 시 리소스 정리
    print('semantic mppi imitation data collect done!')
    env.close()

def collect_data():
    
    return tdsm
    pass