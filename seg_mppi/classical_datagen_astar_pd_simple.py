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


import numpy as np
import cv2
import math
import heapq
from collections import deque

def build_occupancy_from_tdsm_colors(tdsm_bgr, tol=10, k_close=5, k_open=3):
    """
    더 관대한 색상 매칭과 더 강한 morphology 연산으로 occupancy map 생성
    """
    H, W = tdsm_bgr.shape[:2]
    img = tdsm_bgr.astype(np.int16)
    
    # 주행 가능 영역 색상들 (더 넓은 범위로 설정)
    # Pink/Magenta: BGR=(244,35,232) - 도로
    # Yellow: BGR=(55,176,189) - 인도/보행로
    free_colors = [
        np.array([244, 35, 232], dtype=np.int16),  # pink/magenta
        np.array([55, 176, 189], dtype=np.int16),  # yellow
        np.array([128, 64, 128], dtype=np.int16),  # 추가 도로색 (회색)
    ]
    
    free_mask = np.zeros((H, W), dtype=bool)
    
    for color in free_colors:
        mask = np.all(np.abs(img - color.reshape(1,1,3)) <= tol, axis=2)
        free_mask |= mask
    
    free = free_mask.astype(np.uint8)
    
    # 더 강한 morphology 연산으로 연결성 개선
    if k_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if k_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, kernel)
    
    # 추가: dilation으로 주행 영역을 약간 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    free = cv2.dilate(free, kernel, iterations=1)
    
    occ = (1 - free).astype(np.uint8)  # 0=free, 1=obstacle
    return occ, free

def estimate_px_per_meter(img_h, cam_height_m=10.0, fov_deg=90.0):
    """픽셀 당 미터 변환 계수 계산"""
    vertical_m = 2.0 * cam_height_m * math.tan(math.radians(fov_deg) / 2.0)
    return img_h / max(1e-6, vertical_m)

def ego_to_pixel(ego_xy_m, px_per_m, img_h, img_w):
    """ego coordinate를 pixel coordinate로 변환"""
    x_fwd, y_left = float(ego_xy_m[0]), float(ego_xy_m[1])
    row = int(img_h//2 - x_fwd * px_per_m)
    col = int(img_w//2 - y_left * px_per_m)
    return row, col

def pixel_to_ego(row, col, px_per_m, img_h, img_w):
    """pixel coordinate를 ego coordinate로 변환"""
    x_fwd = (img_h//2 - row) / px_per_m
    y_left = (img_w//2 - col) / px_per_m
    return np.array([x_fwd, y_left], dtype=np.float32)

def find_nearest_free_cell(grid, target_rc, max_radius=10):
    """막힌 목표점 주변에서 가장 가까운 자유 공간 찾기"""
    h, w = grid.shape
    tr, tc = target_rc
    
    if 0 <= tr < h and 0 <= tc < w and grid[tr, tc] == 0:
        return target_rc
    
    # BFS로 가장 가까운 자유 공간 찾기
    queue = deque([(tr, tc, 0)])
    visited = set()
    
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) in visited or dist > max_radius:
            continue
        visited.add((r, c))
        
        if 0 <= r < h and 0 <= c < w and grid[r, c] == 0:
            return (r, c)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                queue.append((nr, nc, dist + 1))
    
    return target_rc  # fallback

def improved_astar(grid, start_rc, goal_rc):
    """개선된 A* 알고리즘"""
    h, w = grid.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    
    # 시작점과 목표점이 막혀있으면 가까운 자유 공간으로 이동
    if grid[sr, sc] == 1:
        sr, sc = find_nearest_free_cell(grid, (sr, sc))
    if grid[gr, gc] == 1:
        gr, gc = find_nearest_free_cell(grid, (gr, gc))
    
    def heuristic(r, c):
        return math.hypot(r - gr, c - gc)
    
    # 8방향 이동 (대각선 비용 더 정확하게)
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),  # 직선
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), 
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))  # 대각선
    ]
    
    g_score = {(sr, sc): 0.0}
    parent = {(sr, sc): None}
    pq = [(heuristic(sr, sc), 0.0, sr, sc)]
    visited = set()
    
    while pq:
        f, g_current, r, c = heapq.heappop(pq)
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        if (r, c) == (gr, gc):
            # 경로 복원
            path = []
            current = (r, c)
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        
        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == 1:
                continue
            
            tentative_g = g_current + cost
            
            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                parent[(nr, nc)] = (r, c)
                f_score = tentative_g + heuristic(nr, nc)
                heapq.heappush(pq, (f_score, tentative_g, nr, nc))
    
    return None  # 경로를 찾을 수 없음

def create_enhanced_visualization(tdsm_bgr, occ, free_mask, path, start_rc, goal_rc, local_rc):
    """향상된 시각화 함수"""
    vis = tdsm_bgr.copy()
    h, w = vis.shape[:2]
    
    # 1. 주행 가능 영역을 반투명 초록색으로 표시
    free_overlay = np.zeros_like(vis)
    free_overlay[free_mask == 1] = (0, 255, 0)  # 초록색
    vis = cv2.addWeighted(vis, 0.7, free_overlay, 0.3, 0)
    
    # 2. 장애물을 반투명 빨간색으로 표시
    obstacle_overlay = np.zeros_like(vis)
    obstacle_overlay[occ == 1] = (0, 0, 255)  # 빨간색
    vis = cv2.addWeighted(vis, 1.0, obstacle_overlay, 0.4, 0)
    
    # 3. 경로를 굵은 흰색 선으로 그리기 (여러 번 그려서 더 굵게)
    if path is not None and len(path) >= 2:
        pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
        
        # 배경용 두꺼운 검은 선
        cv2.polylines(vis, [pts], False, (0, 0, 0), 8, cv2.LINE_AA)
        # 메인 흰색 선
        cv2.polylines(vis, [pts], False, (255, 255, 255), 5, cv2.LINE_AA)
        
        # 경로의 방향성을 보여주는 화살표들
        for i in range(0, len(path) - 5, 5):  # 5칸씩 건너뛰며 화살표 그리기
            p1 = path[i]
            p2 = path[i + 3] if i + 3 < len(path) else path[-1]
            
            # 화살표 그리기
            pt1 = (int(p1[1]), int(p1[0]))  # (col, row) -> (x, y)
            pt2 = (int(p2[1]), int(p2[0]))
            cv2.arrowedLine(vis, pt1, pt2, (255, 255, 0), 2, tipLength=0.3)
    
    # 4. 주요 포인트들을 큰 원으로 표시
    def draw_point(rc, color, label, radius=8):
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < h and 0 <= c < w:
            # 외곽선
            cv2.circle(vis, (c, r), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            # 메인 색상
            cv2.circle(vis, (c, r), radius, color, -1, cv2.LINE_AA)
            # 텍스트 레이블
            cv2.putText(vis, label, (c + radius + 5, r + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    draw_point(start_rc, (0, 255, 255), "START")      # 노란색
    draw_point(goal_rc, (0, 165, 255), "GOAL")        # 주황색  
    draw_point(local_rc, (255, 0, 255), "LOCAL")      # 자홍색
    
    # 5. 정보 텍스트 추가
    info_text = [
        f"Path length: {len(path) if path else 0} points",
        f"Grid size: {h}x{w}",
        f"Free cells: {np.sum(free_mask)}",
        f"Obstacle cells: {np.sum(occ)}"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(vis, text, (10, 30 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    return vis

def process_semantic_map_and_plan(tdsm_bgr, ego_goal_position, save_path=None):
    """
    메인 처리 함수: semantic map에서 A* 경로 계획 및 시각화
    
    Args:
        tdsm_bgr: Top-down semantic map (BGR format)
        ego_goal_position: 목표 위치 [x_forward, y_left] in meters
        save_path: 저장할 경로 (None이면 저장하지 않음)
    
    Returns:
        vis: 시각화된 이미지
        path: A* 경로 [(row, col), ...]
        local_target_ego: 로컬 타겟의 ego coordinate
    """
    H, W = tdsm_bgr.shape[:2]
    
    print(f"Processing semantic map: {H}x{W}")
    
    # 1. Occupancy map 생성
    occ, free_mask = build_occupancy_from_tdsm_colors(tdsm_bgr, tol=10, k_close=5, k_open=3)
    
    print(f"Free cells: {np.sum(free_mask)}, Obstacle cells: {np.sum(occ)}")
    
    # 2. 픽셀-미터 변환 계수
    PX_PER_M = estimate_px_per_meter(img_h=H, cam_height_m=10.0, fov_deg=90.0)
    print(f"Pixels per meter: {PX_PER_M:.2f}")
    
    # 3. 목표점을 픽셀 좌표로 변환
    goal_row, goal_col = ego_to_pixel(ego_goal_position, PX_PER_M, H, W)
    goal_row = int(np.clip(goal_row, 0, H-1))
    goal_col = int(np.clip(goal_col, 0, W-1))
    
    print(f"Goal in ego: {ego_goal_position}, Goal in pixels: ({goal_row}, {goal_col})")
    
    # 4. A* 실행
    start_rc = (H//2, W//2)  # 이미지 중앙 (에이전트 위치)
    path = improved_astar(occ, start_rc, (goal_row, goal_col))
    
    if path is None:
        print("WARNING: No path found!")
        local_rc = (goal_row, goal_col)
        local_target_ego = ego_goal_position
    else:
        print(f"Path found with {len(path)} points")
        # 로컬 타겟 선택 (lookahead)
        lookahead_distance = 15  # 픽셀 단위
        lookahead_idx = min(lookahead_distance, len(path) - 1)
        local_rc = path[lookahead_idx]
        
        # 로컬 타겟을 ego coordinate로 변환
        local_target_ego = pixel_to_ego(local_rc[0], local_rc[1], PX_PER_M, H, W)
    
    print(f"Local target in ego: {local_target_ego}")
    
    # 5. 시각화 생성
    vis = create_enhanced_visualization(tdsm_bgr, occ, free_mask, path, 
                                       start_rc, (goal_row, goal_col), local_rc)
    
    # 6. 저장
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
        print(f"Visualization saved to: {save_path}")
    
    return vis, path, local_target_ego

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="saved_imgs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# 개선된 데이터셋 수집기 import
from world_dataset import ImitationDatasetCollector
# 데이터셋 수집기 초기화
collector = ImitationDatasetCollector("imitation_dataset")

running = True
front  = -2
height = 2.2
episode = 3

from semantic_env_config import EnvConfig
env_config = EnvConfig()
env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)

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

                rgb = obs["image"][..., -1]
                max_rgb_value = rgb.max()
                
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)
                depth_front = obs["depth"][..., -1]

                depth_normalized = cv2.normalize(depth_front, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_img = cv2.bitwise_not(depth_front)[..., None]
                
                semantic_front = obs["semantic"][..., -1]
                semantic = (semantic_front[..., ::-1] * 255).astype(np.uint8)
                
                
                # ===== Top-down View 데이터 취득 =====
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

                vis, path, local_target_ego = process_semantic_map_and_plan(
                    tdsm_bgr, ego_goal_position, 
                    save_path=os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_t_{scenario_t:06d}_astar.png")
                )

                # PD 컨트롤러에 사용
                y_left_error = float(local_target_ego[1])
                action = pd_controller.update(y_left_error)
                
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

""" 
## Problem

이 코드에서 문제점이 있다면 장애물을 너무 가까스로 지나가서 자꾸 부딪힌다는것이야. 
inflation layer를 주고 astar에 관련한 것은 이미지를 저장하는것은 옵션으로만 넣고 default로는 끔.
대신 영상을 볼 수 있는 옵션을 default로 킴. 
이에 대한 minimal한 코드 작성

"""