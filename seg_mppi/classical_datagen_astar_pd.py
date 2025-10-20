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


# =================== Utilities for planning ===================

import math
from collections import deque

# (1) top-down semantic -> occupancy (0=free, 1=obstacle)

# (2) pixel <-> ego(meters) 변환 (픽셀-미터 스케일)
def estimate_px_per_meter(img_h, cam_height_m=10.0, fov_deg=90.0):
    """
    top-down 카메라를 높이 cam_height에서 pitch -90도로 내려보는 형태로 사용.
    수직 시야폭(미터) ~= 2 * h * tan(FOV/2), px_per_m = img_h / vertical_meters.
    """
    vertical_m = 2.0 * cam_height_m * math.tan(math.radians(fov_deg) / 2.0)
    return img_h / max(1e-6, vertical_m)

def ego_to_pixel(ego_xy_m, px_per_m, img_h, img_w):
    """
    ego frame: +x=앞(이미지 위쪽), +y=좌(이미지 왼쪽)라고 가정.
    이미지 좌표: (row 증가=아래, col 증가=오른쪽).
    """
    x_fwd, y_left = float(ego_xy_m[0]), float(ego_xy_m[1])
    row = int(img_h//2 - x_fwd * px_per_m)
    col = int(img_w//2 - y_left * px_per_m)
    return row, col

def pixel_to_ego(row, col, px_per_m, img_h, img_w):
    x_fwd = (img_h//2 - row) / px_per_m
    y_left = (img_w//2 - col) / px_per_m
    return np.array([x_fwd, y_left], dtype=np.float32)

# (3) A* on grid (8-이웃)
def astar(grid, start_rc, goal_rc):
    """
    grid: 0/1 occupancy (1=blocked).
    start_rc, goal_rc: (row, col)
    """
    h, w = grid.shape
    sr, sc = start_rc; gr, gc = goal_rc
    if grid[gr, gc] == 1:
        # goal이 막혔으면 주변 free로 이동
        rad = 3
        found = False
        for r in range(gr-rad, gr+rad+1):
            for c in range(gc-rad, gc+rad+1):
                if 0<=r<h and 0<=c<w and grid[r,c]==0:
                    gr, gc = r, c; found=True; break
            if found: break

    def hcost(r, c):  # 휴리스틱(유클리드)
        return math.hypot(r - gr, c - gc)

    nbr8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    g = { (sr,sc): 0.0 }
    parent = { (sr,sc): None }
    import heapq
    pq = [(hcost(sr,sc), 0.0, sr, sc)]
    visited = set()

    while pq:
        f, gcur, r, c = heapq.heappop(pq)
        if (r,c) in visited: continue
        visited.add((r,c))
        if (r,c)==(gr,gc):
            # 경로 복원
            path = []
            cur = (r,c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for dr,dc in nbr8:
            nr, nc = r+dr, c+dc
            if not(0<=nr<h and 0<=nc<w): continue
            if grid[nr,nc]==1: continue
            step = math.hypot(dr,dc)
            ng = gcur + step
            if (nr,nc) not in g or ng < g[(nr,nc)]:
                g[(nr,nc)] = ng
                parent[(nr,nc)] = (r,c)
                heapq.heappush(pq, (ng + hcost(nr,nc), ng, nr, nc))
    return None  # 경로 없음

def draw_astar_overlay(img_bgr, path, start_rc, goal_rc, local_rc, save_path):
    """
    img_bgr: uint8 BGR (cv2용)
    path: [(row, col), ...]  from astar()
    start_rc, goal_rc, local_rc: (row, col)
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # 경로 polyline 그리기 (cv2는 (x,y)=(col,row) 순서)
    if path is not None and len(path) >= 2:
        pts = np.array([[c, r] for (r, c) in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # 시작/목표/로컬타겟 표시
    def dot(rc, color, R=4):
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < h and 0 <= c < w:
            cv2.circle(vis, (c, r), R, color, thickness=-1)

    dot(start_rc, (0, 255, 255))   # start=yellow
    dot(goal_rc,  (0, 165, 255))   # goal=orange
    dot(local_rc, (255, 0, 0))     # local target=blue

    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)


# ======= TDSM 분석 및 점유맵 생성 =======
import numpy as np, cv2, os

def build_occupancy_from_tdsm_colors(tdsm_bgr, tol=2, k_close=3, k_open=3):
    """
    tdsm_bgr: BGR uint8 (H,W,3)
    free_colors: 정확 주행가능 색 2개만 사용
      - pink/magenta: BGR=(244,35,232)
      - yellow      : BGR=(55,176,189)
    tol: 채널별 허용 오차(안티앨리어싱 보정). 0~3 권장.
    반환: occ (uint8, 0=free, 1=obstacle)
    """
    H, W = tdsm_bgr.shape[:2]
    img = tdsm_bgr.astype(np.int16)

    c1 = np.array([244, 35, 232], dtype=np.int16)  # pink/magenta
    c2 = np.array([ 55,176, 189], dtype=np.int16)  # yellow

    m1 = np.all(np.abs(img - c1.reshape(1,1,3)) <= tol, axis=2)  # True=free
    m2 = np.all(np.abs(img - c2.reshape(1,1,3)) <= tol, axis=2)

    free = (m1 | m2).astype(np.uint8)

    # 모폴로지로 빈틈/노이즈 정리 (선택)
    if k_close > 0:
        k = np.ones((k_close, k_close), np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, k)
    if k_open > 0:
        k = np.ones((k_open, k_open), np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, k)

    occ = (1 - free).astype(np.uint8)  # 0=free, 1=obstacle
    return occ


def draw_astar_overlay_with_occ(tdsm_bgr, occ, path, start_rc, goal_rc, local_rc, save_path):
    """
    tdsm_bgr: BGR uint8
    occ: 0/1 (1=obstacle)
    path: [(row,col), ...]
    """
    vis = tdsm_bgr.copy()
    # 장애물 반투명 오버레이
    occ_color = np.zeros_like(vis); occ_color[occ==1] = (0,0,255)
    vis = cv2.addWeighted(vis, 1.0, occ_color, 0.35, 0)

    # 경로 그리기(흰색, 두께 3)
    if path is not None and len(path) >= 2:
        pts = np.array([[c, r] for (r, c) in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False, color=(255,255,255), thickness=3, lineType=cv2.LINE_AA)

    # 점 표시(조금 크게)
    def dot(rc, color, R=5):
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < vis.shape[0] and 0 <= c < vis.shape[1]:
            cv2.circle(vis, (c, r), R, color, thickness=-1, lineType=cv2.LINE_AA)

    dot(start_rc, (0,255,255))   # 시작: 노랑
    dot(goal_rc,  (0,165,255))   # 목표: 주황
    dot(local_rc, (255,0,0))     # 로컬: 파랑

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)


# if __name__ == "__main__":

map_type = 'X'
config = dict(
    crswalk_density=1,
    object_density=0.7,
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
parser.add_argument("--out_dir", type=str, default="saved_imgs_origin")
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
                # (steer axis=0, speed axis=1)

                # =================== Method 1: A* + PD ===================
                # =================== A* + PD (간소화 버전) ===================
                H, W = top_down_semantic_map.shape[:2]   # BGR uint8
                tdsm_bgr = top_down_semantic_map

                # (1) 정확 색 기반 점유맵 (pink + yellow만 free)
                occ = build_occupancy_from_tdsm_colors(tdsm_bgr, tol=2, k_close=3, k_open=3)

                # (2) 픽셀-미터 스케일 (top-down z=10, fov=90와 일치)
                PX_PER_M = estimate_px_per_meter(img_h=H, cam_height_m=10.0, fov_deg=90.0)

                # (3) 에고 목표를 픽셀로
                goal_row, goal_col = ego_to_pixel(ego_goal_position, PX_PER_M, H, W)
                goal_row = int(np.clip(goal_row, 0, H-1))
                goal_col = int(np.clip(goal_col, 0, W-1))

                # (4) A*
                start_rc = (H//2, W//2)
                path = astar(occ, start_rc, (goal_row, goal_col))

                # (5) 로컬 타겟(경로 없으면 goal로 대체)
                if path is None or len(path) < 2:
                    local_rc = (goal_row, goal_col)
                else:
                    lookahead = min(12, len(path)-1)
                    local_rc = path[lookahead]

                # (6) 오버레이 저장(확실히 보이게)
                def draw_astar_overlay_with_occ(tdsm_bgr, occ, path, start_rc, goal_rc, local_rc, save_path):
                    vis = tdsm_bgr.copy()
                    # 장애물 반투명 빨강
                    occ_color = np.zeros_like(vis); occ_color[occ==1] = (0,0,255)
                    vis = cv2.addWeighted(vis, 1.0, occ_color, 0.35, 0)
                    # 경로(흰색 두께3)
                    if path is not None and len(path) >= 2:
                        pts = np.array([[c, r] for (r, c) in path], dtype=np.int32).reshape(-1,1,2)
                        cv2.polylines(vis, [pts], False, (255,255,255), 3, cv2.LINE_AA)
                    # 점
                    for rc, col in [(start_rc,(0,255,255)), (goal_rc,(0,165,255)), (local_rc,(255,0,0))]:
                        r, c = int(rc[0]), int(rc[1])
                        if 0<=r<vis.shape[0] and 0<=c<vis.shape[1]:
                            cv2.circle(vis, (c,r), 5, col, -1, cv2.LINE_AA)
                    os.makedirs(args.out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_t_{scenario_t:06d}_astar_overlay.png"), vis)

                draw_astar_overlay_with_occ(tdsm_bgr, occ, path, start_rc, (goal_row,goal_col), local_rc, 
                                            save_path=os.path.join(args.out_dir, "tmp.png"))

                # (7) 로컬 타겟을 에고(m)로 → PD → action
                local_ego = pixel_to_ego(local_rc[0], local_rc[1], PX_PER_M, H, W)
                y_left_error = float(local_ego[1])

                u = pd_controller.update(y_left_error)
                if isinstance(u, (list, tuple, np.ndarray)):  # 스칼라 보장
                    u = float(np.ravel(u)[0])
                steer_cmd = float(np.clip(u, -1.0, 1.0))
                speed_cmd = 1.0
                action = [steer_cmd, speed_cmd]
                
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