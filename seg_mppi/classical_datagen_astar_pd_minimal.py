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

def build_occupancy_from_tdsm_colors(tdsm_bgr, tol=10, k_close=5, k_open=3, inflation_radius=15):
    """
    더 관대한 색상 매칭과 더 강한 morphology 연산으로 occupancy map 생성
    inflation_radius를 추가하여 장애물 주변에 안전 버퍼 생성
    """
    H, W = tdsm_bgr.shape[:2]
    img = tdsm_bgr.astype(np.int16)
    
    # 주행 가능 영역 색상들 (더 넓은 범위로 설정)
    free_colors = [
        np.array([244, 35, 232], dtype=np.int16),  # pink/magenta
        np.array([55, 176, 189], dtype=np.int16),  # yellow
        # np.array([128, 64, 128], dtype=np.int16),  # 추가 도로색 (회색)
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
    
    # 기본 occupancy map 생성
    occ = (1 - free).astype(np.uint8)  # 0=free, 1=obstacle
    
    # **INFLATION LAYER 추가** - 장애물 주변에 안전 버퍼 생성
    if inflation_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (2*inflation_radius+1, 2*inflation_radius+1))
        occ_inflated = cv2.dilate(occ, kernel, iterations=1)
        
        # inflation으로 인해 줄어든 자유공간을 다시 계산
        free_inflated = (1 - occ_inflated).astype(np.uint8)
    else:
        occ_inflated = occ
        free_inflated = free
    
    return occ_inflated, free_inflated

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

def find_nearest_free_cell(grid, target_rc, max_radius=15):
    """막힌 목표점 주변에서 가장 가까운 자유 공간 찾기 (반경 증가)"""
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
    """개선된 A* 알고리즘 - inflation이 적용된 grid 사용"""
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
    
    # 2. 장애물을 반투명 빨간색으로 표시 (inflation 적용된 영역)
    obstacle_overlay = np.zeros_like(vis)
    obstacle_overlay[occ == 1] = (0, 0, 255)  # 빨간색
    vis = cv2.addWeighted(vis, 1.0, obstacle_overlay, 0.4, 0)
    
    # 3. 경로를 굵은 흰색 선으로 그리기
    if path is not None and len(path) >= 2:
        pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
        
        # 배경용 두꺼운 검은 선
        cv2.polylines(vis, [pts], False, (0, 0, 0), 12, cv2.LINE_AA)
        # 메인 흰색 선
        cv2.polylines(vis, [pts], False, (0, 255, 255), 10, cv2.LINE_AA)
        
        # 경로의 방향성을 보여주는 화살표들
        for i in range(0, len(path) - 5, 5):
            p1 = path[i]
            p2 = path[i + 3] if i + 3 < len(path) else path[-1]
            
            pt1 = (int(p1[1]), int(p1[0]))
            pt2 = (int(p2[1]), int(p2[0]))
            cv2.arrowedLine(vis, pt1, pt2, (255, 255, 0), 2, tipLength=0.3)
    
    # 4. 주요 포인트들을 큰 원으로 표시
    def draw_point(rc, color, label, radius=8):
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < h and 0 <= c < w:
            cv2.circle(vis, (c, r), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (c, r), radius, color, -1, cv2.LINE_AA)
            cv2.putText(vis, label, (c + radius + 5, r + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    draw_point(start_rc, (0, 255, 255), "START")      
    draw_point(goal_rc, (0, 165, 255), "GOAL")        
    draw_point(local_rc, (255, 0, 255), "LOCAL")      
    
    return vis

def process_semantic_map_and_plan(tdsm_bgr, ego_goal_position, inflation_radius=15, 
                                show_visualization=True, save_path=None):
    """
    메인 처리 함수: semantic map에서 A* 경로 계획 및 시각화
    
    Args:
        tdsm_bgr: Top-down semantic map (BGR format)
        ego_goal_position: 목표 위치 [x_forward, y_left] in meters
        inflation_radius: 장애물 팽창 반경 (픽셀 단위)
        show_visualization: 시각화 표시 여부 (기본 True)
        save_path: 저장할 경로 (None이면 저장하지 않음, 기본 None)
    
    Returns:
        vis: 시각화된 이미지 (show_visualization=True일 때만)
        path: A* 경로 [(row, col), ...]
        local_target_ego: 로컬 타겟의 ego coordinate
    """
    H, W = tdsm_bgr.shape[:2]
    
    # 1. Occupancy map 생성 (inflation layer 포함)
    occ, free_mask = build_occupancy_from_tdsm_colors(tdsm_bgr, tol=10, k_close=5, k_open=3,
                                                     inflation_radius=inflation_radius)
    
    # 2. 픽셀-미터 변환 계수
    PX_PER_M = estimate_px_per_meter(img_h=H, cam_height_m=10.0, fov_deg=90.0)
    
    # 3. 목표점을 픽셀 좌표로 변환
    goal_row, goal_col = ego_to_pixel(ego_goal_position, PX_PER_M, H, W)
    goal_row = int(np.clip(goal_row, 0, H-1))
    goal_col = int(np.clip(goal_col, 0, W-1))
    
    # 4. A* 실행 (inflation이 적용된 occupancy map 사용)
    start_rc = (H//2, W//2)
    path = improved_astar(occ, start_rc, (goal_row, goal_col))
    
    if path is None:
        print("WARNING: No path found!")
        local_rc = (goal_row, goal_col)
        local_target_ego = ego_goal_position
    else:
        # 로컬 타겟 선택 (lookahead)
        lookahead_distance = 15
        lookahead_idx = min(lookahead_distance, len(path) - 1)
        local_rc = path[lookahead_idx]
        
        # 로컬 타겟을 ego coordinate로 변환
        local_target_ego = pixel_to_ego(local_rc[0], local_rc[1], PX_PER_M, H, W)
    
    # 5. 시각화 생성 및 표시
    vis = None
    if show_visualization:
        vis = create_enhanced_visualization(tdsm_bgr, occ, free_mask, path, 
                                           start_rc, (goal_row, goal_col), local_rc)
        
        # OpenCV 창으로 시각화 표시
        cv2.imshow('A* Path Planning with Inflation', vis)
        cv2.waitKey(1)  # 1ms 대기
        
        # 저장 (옵션)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis)
    
    return vis, path, local_target_ego

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="saved_imgs")
parser.add_argument("--show_vis", action="store_true", default=True, help="Show real-time visualization")
parser.add_argument("--save_images", action="store_true", default=False, help="Save visualization images")
parser.add_argument("--inflation_radius", type=int, default=15, help="Obstacle inflation radius in pixels")
args = parser.parse_args()

if args.save_images:
    os.makedirs(args.out_dir, exist_ok=True)

# 개선된 데이터셋 수집기 import
from world_dataset import ImitationDatasetCollector
collector = ImitationDatasetCollector("imitation_dataset")

running = True
episode = 3

from semantic_env_config import EnvConfig
env_config = EnvConfig()
env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)

try:
    for i in range(100000):
        episode += 1
        
        obs, info = env.reset(seed=episode)
        waypoints = env.agent.navigation.checkpoints 
        
        # 웨이포인트가 충분히 있는지 확인
        while len(waypoints) < 30:
            episode += 1
            obs, info = env.reset(seed=episode)
            waypoints = env.agent.navigation.checkpoints 
        
        num_waypoints = len(waypoints)
        print(f'Episode {episode}: {num_waypoints} waypoints')
        
        collector.start_new_episode(waypoints)

        time_interval = 2
        scenario_t = 0
        
        # Top-down 카메라 센서 가져오기
        top_down_camera = env.engine.get_sensor("top_down_semantic")
        
        k = 5  # 5번째 웨이포인트를 목표로 설정
        reward = 0
        
        while running:
            if scenario_t % time_interval == 0:
                # 센서 데이터 처리
                rgb = obs["image"][..., -1]
                max_rgb_value = rgb.max()
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)
                
                # Top-down semantic map 획득
                top_down_semantic_map = top_down_camera.perceive(
                    new_parent_node=env.agent.origin,
                    position=[0, 0, 10],
                    hpr=[0, -90, 0]
                )
                
                tdsm_bgr = (top_down_semantic_map[..., ::-1] * 255).astype(np.uint8)
                
                # 목표 설정
                waypoints = env.agent.navigation.checkpoints 
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                
                ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
                
                # 목표 웨이포인트 업데이트
                distance_to_target = np.linalg.norm(ego_goal_position)
                if distance_to_target < 5.0:
                    k += 1
                    if k >= num_waypoints:
                        k = num_waypoints - 1

                # 에이전트 상태 정보
                agent_state = {
                    "position": env.agent.position,
                    "heading": env.agent.heading_theta,
                    "velocity": env.agent.speed,
                    "angular_velocity": getattr(env.agent, 'angular_velocity', 0.0)
                }
                
                # **개선된 A* with Inflation Layer 사용**
                save_path = None
                if args.save_images:
                    save_path = os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_t_{scenario_t:06d}_astar.png")
                
                vis, path, local_target_ego = process_semantic_map_and_plan(
                    tdsm_bgr, ego_goal_position, 
                    inflation_radius=args.inflation_radius,
                    show_visualization=args.show_vis,
                    save_path=save_path
                )

                # PD 컨트롤러 사용
                y_left_error = float(local_target_ego[1])
                action = pd_controller.update(y_left_error)
                
                # 데이터 수집
                collector.collect_sample(
                    obs, action, agent_state, ego_goal_position, reward, scenario_t
                )
                
            obs, reward, tm, tc, info = env.step(action)
            scenario_t += 1

            # 에피소드 종료 조건
            if tm or tc or scenario_t >= 800 or reward < 0:
                episode_info = {
                    "seed": episode,
                    "terminated": tm,
                    "truncated": tc,
                    "crashed": reward < 0,
                    "episode_length": scenario_t,
                    "success": bool(np.linalg.norm(waypoints[-1] - env.agent.position) < 1)
                }

                agent_state = {
                    "position": env.agent.position,
                    "heading": env.agent.heading_theta,
                    "velocity": env.agent.speed,
                }
                
                collector.collect_sample(
                    obs, action, agent_state, ego_goal_position, reward, scenario_t
                )  
                break
        
        print(f"Episode completed with {scenario_t} steps")
        collector.finish_episode(episode_info)
        
        # 주기적으로 데이터셋 저장
        if scenario_t > 64 and collector.episode_counter % 10 == 0:
            collector.save_dataset_info()
            collector.create_train_test_split()
            print(f"\nDataset update - Episode {collector.episode_counter}:")
            print(f"  Total episodes: {collector.episode_counter}")
            print(f"  Total samples: {collector.dataset_info['total_samples']}")
            
finally:
    print('Semantic MPPI imitation data collection done!')
    cv2.destroyAllWindows()  # OpenCV 창 닫기
    env.close()

# python script.py --inflation_radius 10 --save_images --show_vis