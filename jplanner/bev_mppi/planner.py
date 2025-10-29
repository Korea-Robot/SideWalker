#!/usr/bin/env python3
# planner.py

import math
import torch

class SubgoalPlanner:
    """
    역할: MPPI가 따라갈 중간 목표 지점(subgoal)을 *Costmap을 인지하여* 생성합니다.

    로봇 전방의 여러 후보 지점을 샘플링하고,
    (1) 최종 목표까지의 거리
    (2) Costmap 상의 장애물 비용
    을 모두 고려하여 최적의 "당근"을 선택합니다.
    """
    
    def __init__(self, logger, device,
                 lookahead_distance, goal_threshold,
                 num_subgoal_samples, subgoal_goal_cost_w, subgoal_obs_cost_w,
                 grid_resolution, grid_origin_x, grid_origin_y, cells_x, cells_y):
        """
        Args:
            logger: ROS 2 노드의 로거
            device: 'cuda' 또는 'cpu'
            lookahead_distance (float): 서브골을 샘플링할 기본 거리
            goal_threshold (float): 최종 목표 도달 임계값
            num_subgoal_samples (int): 매 스텝마다 평가할 후보 서브골 개수 (e.g., 50)
            subgoal_goal_cost_w (float): 서브골 평가 시 목표 근접성 가중치
            subgoal_obs_cost_w (float): 서브골 평가 시 장애물 비용 가중치
            ...grid_params: Costmap 좌표 변환을 위한 파라미터
        """
        self.logger = logger
        self.device = device
        
        # 플래너 파라미터
        self.lookahead_distance = lookahead_distance
        self.goal_threshold = goal_threshold
        self.num_subgoal_samples = num_subgoal_samples
        self.goal_w = subgoal_goal_cost_w
        self.obs_w = subgoal_obs_cost_w

        # Costmap 파라미터
        self.grid_resolution = grid_resolution
        self.grid_origin_x = grid_origin_x
        self.grid_origin_y = grid_origin_y
        self.cells_x = cells_x
        self.cells_y = cells_y

        # 디버깅 변수
        self.last_subgoal_x = 0.0
        self.last_subgoal_y = 0.0
        self.is_final_subgoal = False
        
        # 후보 샘플링을 위한 각도 텐서 (미리 계산)
        # -60도 ~ +60도 사이의 N개 샘플
        self.sample_angles_rel = torch.linspace(
            # -math.pi / 3.0, math.pi / 3.0,
            -math.pi / 4.0, math.pi / 4.0, 
            self.num_subgoal_samples, 
            device=self.device, dtype=torch.float32
        )

    def world_to_grid_idx_torch(self, x, y):
        """월드 좌표(m) 텐서를 그리드 인덱스(r, c) 텐서로 변환"""
        grid_c = ((x - self.grid_origin_x) / self.grid_resolution).long()
        grid_r = ((y - self.grid_origin_y) / self.grid_resolution).long()
        return grid_r, grid_c

    def get_subgoal(self, current_x, current_y, main_target_x, main_target_y, costmap_tensor):
        """
        현재 위치, 최종 목표, *Costmap*을 기반으로 최적의 서브골을 계산합니다.

        Args:
            current_x, current_y (float): 로봇의 현재 글로벌 위치
            main_target_x, main_target_y (float): 추종해야 할 *최종* 웨이포인트의 글로벌 위치
            costmap_tensor (torch.Tensor): MPPI가 사용하는 BEV Costmap

        Returns:
            tuple: ((subgoal_x, subgoal_y), is_final_subgoal)
        """
        
        # 1. 로봇과 *최종* 목표 지점 사이의 거리와 방향 계산
        dx_global = main_target_x - current_x
        dy_global = main_target_y - current_y
        distance_to_main_target = math.sqrt(dx_global**2 + dy_global**2)
        
        # 2. 목표 도달 확인
        if distance_to_main_target < self.goal_threshold:
            self.last_subgoal_x = main_target_x
            self.last_subgoal_y = main_target_y
            self.is_final_subgoal = True
            return (main_target_x, main_target_y), True

        # --- Cost-Aware 서브골 탐색 ---

        # 3. Costmap이 준비되지 않았다면, (안전하지 않지만) 기존의 Naive 로직으로 대체
        if costmap_tensor is None:
            self.logger.warn("SubgoalPlanner: Costmap not ready, falling back to naive (unsafe) subgoal.", throttle_duration_sec=1.0)
            return self._get_naive_subgoal(current_x, current_y, dx_global, dy_global, distance_to_main_target, main_target_x, main_target_y)

        try:
            # 4. 후보 서브골 생성 (N개 샘플)
            # 최종 목표 방향을 기준으로 +- 60도 범위의 후보 각도 생성
            angle_to_main_goal = torch.atan2(
                torch.tensor(dy_global, device=self.device), 
                torch.tensor(dx_global, device=self.device)
            )
            candidate_angles = angle_to_main_goal + self.sample_angles_rel
            
            # (N, 2) 크기의 후보 서브골 좌표 (글로벌)
            cand_x = current_x + torch.cos(candidate_angles) * self.lookahead_distance
            cand_y = current_y + torch.sin(candidate_angles) * self.lookahead_distance
            
            # 5. 후보 서브골 비용 평가 (N개 동시)
            
            # 5-1. Cost 1: 최종 목표와의 거리 (낮을수록 좋음)
            cost_goal = torch.sqrt(
                (cand_x - main_target_x)**2 + (cand_y - main_target_y)**2
            )
            
            # 5-2. Cost 2: Costmap 상의 장애물 비용 (낮을수록 좋음)
            grid_r, grid_c = self.world_to_grid_idx_torch(cand_x, cand_y)
            
            # 맵 경계 체크
            out_of_bounds = (grid_c < 0) | (grid_c >= self.cells_x) | \
                            (grid_r < 0) | (grid_r >= self.cells_y)
            
            grid_r_clamped = torch.clamp(grid_r, 0, self.cells_y - 1)
            grid_c_clamped = torch.clamp(grid_c, 0, self.cells_x - 1)
            
            cost_obstacle = costmap_tensor[grid_r_clamped, grid_c_clamped] / 255.0 # 0.0 ~ 1.0
            cost_obstacle[out_of_bounds] = 10.0 # 맵 밖은 매우 높은 비용 부여

            # 5-3. 총 비용
            total_cost = (self.goal_w * cost_goal) + (self.obs_w * cost_obstacle)

            # 6. 최적의 서브골 선택
            best_idx = torch.argmin(total_cost)
            best_subgoal_x = cand_x[best_idx].item()
            best_subgoal_y = cand_y[best_idx].item()

            self.last_subgoal_x = best_subgoal_x
            self.last_subgoal_y = best_subgoal_y
            self.is_final_subgoal = False
            return (best_subgoal_x, best_subgoal_y), False

        except Exception as e:
            self.logger.error(f"SubgoalPlanner failed: {e}\n{traceback.format_exc()}")
            # 실패 시 안전을 위해 기존 Naive 로직 사용
            return self._get_naive_subgoal(current_x, current_y, dx_global, dy_global, distance_to_main_target, main_target_x, main_target_y)


    def _get_naive_subgoal(self, current_x, current_y, dx_global, dy_global, distance_to_main_target, main_target_x, main_target_y):
        """Costmap이 없을 때 사용하는 기존의 직선 기반 서브골 계산"""
        
        # 3. "Long Horizon" 문제 해결 (기존 로직)
        if distance_to_main_target > self.lookahead_distance:
            unit_vec_x = dx_global / distance_to_main_target
            unit_vec_y = dy_global / distance_to_main_target
            subgoal_x = current_x + unit_vec_x * self.lookahead_distance
            subgoal_y = current_y + unit_vec_y * self.lookahead_distance
            
            self.last_subgoal_x = subgoal_x
            self.last_subgoal_y = subgoal_y
            self.is_final_subgoal = False
            return (subgoal_x, subgoal_y), False
            
        # 4. 목표가 lookahead 거리보다는 가깝지만, goal_threshold보다는 먼 경우
        else:
            self.last_subgoal_x = main_target_x
            self.last_subgoal_y = main_target_y
            self.is_final_subgoal = False
            return (main_target_x, main_target_y), False

