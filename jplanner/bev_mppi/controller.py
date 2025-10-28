# controller.py

import torch
import math
from dynamics_predictor import DynamicsPredictor

class MPPIController:
    """
    역할: MPPI 알고리즘의 핵심 로직을 모두 포함합니다.

    샘플링, 비용 계산, 가중치 부여, 제어 시퀀스 업데이트를 수행합니다.

    DynamicsPredictor를 내부적으로 사용합니다.

    ROS에 독립적이며, torch 텐서만 입출력으로 받습니다.
    MPPI (Model Predictive Path Integral) 컨트롤러의 핵심 로직.
    ROS로부터 독립적으로 설계되어, 텐서 연산에만 집중합니다.
    """
    
    def __init__(self, logger, device,
                 K, T, dt, lambda_, sigma_v, sigma_w,
                 min_v, max_v, max_w,
                 goal_cost_w, obstacle_cost_w, control_cost_w,
                 grid_resolution, grid_origin_x, grid_origin_y, cells_x, cells_y,
                 num_samples_to_plot):
        """
        모든 MPPI 및 비용 계산 관련 파라미터를 초기화합니다.
        
        Args:
            logger: ROS 2 노드의 로거
            device: 'cuda' 또는 'cpu'
            K, T, dt, lambda_: MPPI 알고리즘 파라미터
            sigma_v, sigma_w: 제어 노이즈
            min_v, max_v, max_w: 로봇 제어 한계
            ..._cost_w: 비용 함수 가중치
            ...grid...: 비용 계산을 위한 BEV 맵 정보
            num_samples_to_plot: 시각화를 위해 반환할 샘플 궤적 수
        """
        self.logger = logger
        self.device = device
        
        # MPPI 파라미터
        self.K = K
        self.T = T
        self.dt = dt
        self.lambda_ = lambda_
        
        # 로봇 한계
        self.min_v = min_v
        self.max_v = max_v
        self.max_w = max_w
        
        # 비용 가중치
        self.goal_cost_w = goal_cost_w
        self.obstacle_cost_w = obstacle_cost_w
        self.control_cost_w = control_cost_w
        
        # BEV 맵 파라미터
        self.grid_resolution = grid_resolution
        self.grid_origin_x = grid_origin_x
        self.grid_origin_y = grid_origin_y
        self.cells_x = cells_x
        self.cells_y = cells_y

        # 시각화 파라미터
        self.num_samples_to_plot = num_samples_to_plot
        
        # --- MPPI 핵심 변수 ---
        
        # 1. 동역학 모델
        self.predictor = DynamicsPredictor(self.device, self.dt)
        
        # 2. 평균 제어 시퀀스 (v, w). (T, 2)
        self.U = torch.zeros(self.T, 2, device=self.device, dtype=torch.float32)
        
        # 3. 제어 노이즈 공분산 (v, w)
        self.Sigma = torch.tensor([[sigma_v**2, 0.0],
                                    [0.0, sigma_w**2]], device=self.device, dtype=torch.float32)
        
        # 4. 노이즈 샘플링을 위한 분포
        self.noise_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2, device=self.device), self.Sigma
        )

    def motion_model(self, states, controls):
        """동역학 예측기 래퍼"""
        return self.predictor.motion_model(states, controls)

    def reset(self):
        """제어 시퀀스(U)를 0으로 리셋 (e.g., 로봇 정지 시)"""
        self.U.zero_()

    def world_to_grid_idx_torch(self, x, y):
        """월드 좌표(m) 텐서를 그리드 인덱스(r, c) 텐서로 변환"""
        grid_c = ((x - self.grid_origin_x) / self.grid_resolution).long()
        grid_r = ((y - self.grid_origin_y) / self.grid_resolution).long()
        return grid_r, grid_c

    def compute_heuristic_prior(self, local_goal_tensor):
        """'Cold Start'를 위해 목표 지향적인 휴리스틱 제어 시퀀스를 생성"""
        self.logger.info("Prior is zero. Generating new goal-directed prior.")
        angle_to_goal = torch.atan2(local_goal_tensor[1], local_goal_tensor[0])
        w = torch.clamp(angle_to_goal * 2.0, -self.max_w, self.max_w)
        v_val = self.max_v * 0.5
        if torch.abs(angle_to_goal) > (math.pi / 4.0):
             v_val = 0.0
        control_prior = torch.tensor([v_val, w.item()], device=self.device, dtype=torch.float32)
        return control_prior.expand(self.T, 2)

    def compute_costs(self, trajectories, local_goal_tensor, perturbed_controls, costmap_tensor):
        """K개의 궤적에 대한 비용을 계산 (병렬 처리)"""
        
        # 1. 목표 지점 비용 (Goal Cost)
        final_states_xy = trajectories[:, -1, :2] # (K, 2)
        goal_cost = torch.linalg.norm(final_states_xy - local_goal_tensor, dim=1)
        
        # 2. 장애물 비용 (Obstacle Cost)
        traj_x = trajectories[..., 0] # (K, T)
        traj_y = trajectories[..., 1] # (K, T)
        grid_r, grid_c = self.world_to_grid_idx_torch(traj_x, traj_y)

        out_of_bounds = (grid_c < 0) | (grid_c >= self.cells_x) | \
                        (grid_r < 0) | (grid_r >= self.cells_y)
        
        grid_r_clamped = torch.clamp(grid_r, 0, self.cells_y - 1)
        grid_c_clamped = torch.clamp(grid_c, 0, self.cells_x - 1)
        
        # Runner로부터 costmap_tensor를 직접 받아 사용
        obstacle_costs_per_step = costmap_tensor[grid_r_clamped, grid_c_clamped] / 255.0
        obstacle_costs_per_step[out_of_bounds] = 1.0 # 맵 밖은 장애물
        obstacle_cost = torch.sum(obstacle_costs_per_step, dim=1) # (K,)
        
        # 3. 제어 비용 (Control Cost)
        control_cost = torch.sum(torch.linalg.norm(perturbed_controls, dim=2), dim=1) # (K,)
        
        # 4. 총 비용 계산
        total_cost = (
            self.goal_cost_w * goal_cost +
            self.obstacle_cost_w * obstacle_cost +
            self.control_cost_w * control_cost
        )
        return total_cost # (K,)

    def run_mppi(self, local_goal_tensor, costmap_tensor):
        """
        MPPI 컨트롤러의 1스텝 실행.
        
        Args:
            local_goal_tensor (torch.Tensor): (2,) 로컬 목표 [x, y]
            costmap_tensor (torch.Tensor): (H, W) BEV Costmap
            
        Returns:
            tuple: (v, w), optimal_trajectory, sampled_trajectories
                   (None, None, None)일 경우 실패
        """
        
        if costmap_tensor is None:
            self.logger.warn("MPPI: Costmap is not ready.")
            return None, None, None # 실패
            
        # 1. Prior(U)가 0인지 (Cold Start) 확인
        if torch.all(self.U == 0.0):
            self.U = self.compute_heuristic_prior(local_goal_tensor)
        
        # 2. (K)개의 노이즈가 추가된 제어 시퀀스(v, w) 샘플 생성
        noise = self.noise_dist.sample((self.K, self.T))
        perturbed_controls = self.U.unsqueeze(0) + noise # (K, T, 2)
        perturbed_controls[..., 0].clamp_(self.min_v, self.max_v)
        perturbed_controls[..., 1].clamp_(-self.max_w, self.max_w)

        # 3. (K)개의 궤적 시뮬레이션 (롤아웃)
        trajectories = torch.zeros(self.K, self.T, 3, device=self.device, dtype=torch.float32)
        current_states = torch.zeros(self.K, 3, device=self.device, dtype=torch.float32) 
        for t in range(self.T):
            next_states = self.motion_model(current_states, perturbed_controls[:, t, :])
            trajectories[:, t, :] = next_states
            current_states = next_states
        
        # 4. (K)개의 궤적에 대한 비용 계산
        costs = self.compute_costs(
            trajectories, local_goal_tensor, perturbed_controls, costmap_tensor
        ) # (K,)

        # 5. 비용 기반 가중치 계산 (Softmax)
        costs_normalized = costs - torch.min(costs)
        weights = torch.exp(-1.0 / self.lambda_ * costs_normalized)
        weights /= (torch.sum(weights) + 1e-9) # (K,)

        # 6. 가중 평균을 사용하여 평균 제어 시퀀스(U) 업데이트
        weighted_noise = torch.einsum('k,ktu->tu', weights, noise)
        self.U = self.U + weighted_noise

        # 7. ★ 시각화를 위한 궤적 준비 ★
        
        # 7-1. 최적 궤적 (U 롤아웃)
        optimal_traj_local = torch.zeros(self.T, 3, device=self.device, dtype=torch.float32)
        current_state_optimal = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        for t in range(self.T):
            control_optimal = self.U[t, :].unsqueeze(0)
            next_state_optimal = self.motion_model(current_state_optimal, control_optimal)
            optimal_traj_local[t, :] = next_state_optimal.squeeze()
            current_state_optimal = next_state_optimal
            
        # 7-2. 샘플 궤적 (다운샘플링)
        if self.K > self.num_samples_to_plot:
            indices = torch.randint(0, self.K, (self.num_samples_to_plot,))
            sampled_trajs_local_subset = trajectories[indices, ...]
        else:
            sampled_trajs_local_subset = trajectories

        # 8. 제어 시퀀스 시프트 및 제어 명령 반환
        best_control = self.U[0, :] # (2,)
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1, :] = 0.0 # 마지막 스텝은 0으로 리셋

        # Runner에게 (제어명령, 최적궤적, 샘플궤적) 반환
        return (best_control[0].item(), best_control[1].item()), \
               optimal_traj_local, \
               sampled_trajs_local_subset
