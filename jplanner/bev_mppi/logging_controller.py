import torch
import math
import time  # 로깅 및 시간 측정을 위해 추가
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
        # 이걸 policy pi 로 바꾸고 싶다. 일단은 heuristic하게 동작시켜보자. 
        self.U = torch.zeros(self.T, 2, device=self.device, dtype=torch.float32)
        # self.T.shape  : 
        # self.U.shape (2)
        # U[0] : linear  velocity
        # U[1] : angular velocity 
        
        # 3. 제어 노이즈 공분산 (v, w)
        self.Sigma = torch.tensor([[sigma_v**2, 0.0],
                                   [0.0, sigma_w**2]], device=self.device, dtype=torch.float32)
        # self.Sigma.shape : variance of 
        
        # 4. 노이즈 샘플링을 위한 분포
        self.noise_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2, device=self.device), self.Sigma
        )

        

    def motion_model(self, states, controls) -> torch.Tensor:
        """동역학 예측기 래퍼"""
        return self.predictor.motion_model(states, controls)

    def reset(self):
        """제어 시퀀스(U)를 0으로 리셋 (e.g., 로봇 정지 시)"""
        self.U.zero_()

    # world coordinates to grid index tensor 
    def world_to_grid_idx_torch(self, x, y):
        """월드 좌표(m) 텐서를 그리드 인덱스(r, c) 텐서로 변환"""
        grid_c = ((x - self.grid_origin_x) / self.grid_resolution).long()
        grid_r = ((y - self.grid_origin_y) / self.grid_resolution).long()
        return grid_r, grid_c

    # heuristic prior about self.T
    def compute_heuristic_prior(self, local_goal_tensor):
        """'Cold Start'를 위해 목표 지향적인 휴리스틱 제어 시퀀스를 생성"""
        self.logger.info("Prior is zero. Generating new goal-directed prior.")
        angle_to_goal = torch.atan2(local_goal_tensor[1], local_goal_tensor[0])
        w = torch.clamp(angle_to_goal * 1.0, -self.max_w, self.max_w)
        v_val = self.max_v * 0.5
        if torch.abs(angle_to_goal) > (math.pi / 4.0):
            v_val = 0.0
            
        
        # (신규) 휴리스틱 계산값 로깅
        self.logger.debug(f"[Controller] Heuristic: angle_to_goal={angle_to_goal.item():.2f}, w={w.item():.2f}, v={v_val:.2f}")
            
        # control_prior = torch.tensor([v_val, w.item()], device=self.device, dtype=torch.float32)
        
        # 괄호를 한 겹 더 추가하여 (1, 2) 크기로 만듦
        control_prior = torch.tensor([[v_val, w.item()]], device=self.device, dtype=torch.float32) # (1,2)
        
        return control_prior.expand(self.T, 2) # (1, 2) -> (T, 2)로 확장

    def compute_costs(self, trajectories, local_goal_tensor, perturbed_controls, costmap_tensor) -> torch.Tensor: # (K,)
        """K개의 궤적에 대한 비용을 계산 (병렬 처리)"""
        
        # (신규) 비용 계산 내부 시간 측정 시작
        cost_start_time = time.perf_counter()
        
        # 1. 목표 지점 비용 (Goal Cost)
        t_goal_start = time.perf_counter()
        final_states_xy = trajectories[:, -1, :2] # (K, 2)
        goal_cost = torch.linalg.norm(final_states_xy - local_goal_tensor, dim=1)
        t_goal_end = time.perf_counter()
        
        # 2. 장애물 비용 (Obstacle Cost)
        t_obs_start = time.perf_counter()
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
        t_obs_end = time.perf_counter()
        
        # 3. 제어 비용 (Control Cost)
        t_ctrl_start = time.perf_counter()
        control_cost = torch.sum(torch.linalg.norm(perturbed_controls, dim=2), dim=1) # (K,)
        t_ctrl_end = time.perf_counter()
        
        # 4. 총 비용 계산
        total_cost = (
            self.goal_cost_w * goal_cost +
            self.obstacle_cost_w * obstacle_cost +
            self.control_cost_w * control_cost
        )
        
        cost_end_time = time.perf_counter()

        # (신규) 비용 계산 통계 및 시간 로깅 (DEBUG 레벨)
        self.logger.debug(
            f"[Controller] compute_costs (Total: {(cost_end_time - cost_start_time)*1000.0:.2f} ms)\n"
            f"  ├─ Time (ms): Goal={((t_goal_end - t_goal_start)*1000.0):.2f}, "
            f"Obstacle={((t_obs_end - t_obs_start)*1000.0):.2f}, "
            f"Control={((t_ctrl_end - t_ctrl_start)*1000.0):.2f}\n"
            f"  ├─ Goal Cost (Raw):     min={goal_cost.min().item():.2f}, max={goal_cost.max().item():.2f}, mean={goal_cost.mean().item():.2f}\n"
            f"  ├─ Obstacle Cost (Raw): min={obstacle_cost.min().item():.2f}, max={obstacle_cost.max().item():.2f}, mean={obstacle_cost.mean().item():.2f}\n"
            f"  └─ Total Cost (Weighted): min={total_cost.min().item():.2f}, max={total_cost.max().item():.2f}, mean={total_cost.mean().item():.2f}"
        )
        
        return total_cost # (K,)

    def run_mppi(self, local_goal_tensor, costmap_tensor):
        """
        MPPI 컨트롤러의 1스텝 실행.
        
        Args:
            local_goal_tensor (torch.Tensor): (2,) 로컬 목표 [x, y]
            costmap_tensor (torch.Tensor): (H, W) BEV Costmap [200,300]
            
        Returns:
            tuple: (v, w), optimal_trajectory, sampled_trajectories
                   (None, None, None)일 경우 실패
        """
        
        # (신규) MPPI 전체 실행 시간 측정 시작
        mppi_total_start = time.perf_counter()
        
        if costmap_tensor is None:
            self.logger.warn("MPPI: Costmap is not ready.")
            return None, None, None # 실패
            
        # 1. Prior(U)가 0인지 (Cold Start) 확인
        t_start_prior = time.perf_counter()
        if torch.all(self.U == 0.0):
            self.U = self.compute_heuristic_prior(local_goal_tensor)
        t_end_prior = time.perf_counter()
        
        # 2. (K)개의 노이즈가 추가된 제어 시퀀스(v, w) 샘플 생성
        t_start_sample = time.perf_counter()
        noise = self.noise_dist.sample((self.K, self.T)) # (K,T,2) : K 경로 개수,  T 시간, 2 인풋사이즈
        perturbed_controls = self.U.unsqueeze(0) + noise # (K, T, 2) : 노이즈가 포함된 경로 
        perturbed_controls[..., 0].clamp_(self.min_v, self.max_v) 
        perturbed_controls[..., 1].clamp_(-self.max_w, self.max_w) # constraints
        t_end_sample = time.perf_counter()

        # 3. (K)개의 궤적 시뮬레이션 (롤아웃)
        t_start_rollout = time.perf_counter()
        trajectories = torch.zeros(self.K, self.T, 3, device=self.device, dtype=torch.float32)
        # coordinates x,y,theta 
        
        current_states = torch.zeros(self.K, 3, device=self.device, dtype=torch.float32) 
        for t in range(self.T):
            next_states = self.motion_model(current_states, perturbed_controls[:, t, :])
            trajectories[:, t, :] = next_states
            current_states = next_states
        t_end_rollout = time.perf_counter()
        
        # 4. (K)개의 궤적에 대한 비용 계산
        t_start_cost = time.perf_counter()
        costs = self.compute_costs(
            trajectories, local_goal_tensor, perturbed_controls, costmap_tensor
        ) # (K,)
        t_end_cost = time.perf_counter()

        # 5. 비용 기반 가중치 계산 (Softmax)
        t_start_weight = time.perf_counter()
        costs_normalized = costs - torch.min(costs)
        weights = torch.exp(-1.0 / self.lambda_ * costs_normalized)
        weights /= (torch.sum(weights) + 1e-9) # (K,)
        t_end_weight = time.perf_counter()

        # (신규) 비용 및 가중치 통계 로깅
        self.logger.debug(
            f"[Controller] Costs (raw): min={costs.min().item():.2f}, max={costs.max().item():.2f}, mean={costs.mean().item():.2f}\n"
            f"  └─ Weights:             min={weights.min().item():.6f}, max={weights.max().item():.6f}, sum={weights.sum().item():.2f}"
        )

        # 6. 가중 평균을 사용하여 평균 제어 시퀀스(U) 업데이트
        t_start_update = time.perf_counter()
        weighted_noise = torch.einsum('k,ktu->tu', weights, noise)
        self.U = self.U + weighted_noise
        t_end_update = time.perf_counter()
        
        # (신규) 업데이트 통계 로깅
        self.logger.debug(
            f"[Controller] Update: weighted_noise_norm={torch.linalg.norm(weighted_noise).item():.4f}, "
            f"New U[0,:]=[{self.U[0, 0].item():.2f}, {self.U[0, 1].item():.2f}]"
        )


        # 7. ★ 시각화를 위한 궤적 준비 ★
        t_start_viz = time.perf_counter()
        
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
        t_end_viz = time.perf_counter()

        # 8. 제어 시퀀스 시프트 및 제어 명령 반환
        best_control = self.U[0, :] # (2,)
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1, :] = 0.0 # 마지막 스텝은 0으로 리셋

        mppi_total_end = time.perf_counter()

        # (신규) 반환될 제어 명령 로깅
        self.logger.debug(f"[Controller] Returning Control: v={best_control[0].item():.3f}, w={best_control[1].item():.3f}")

        # (신규) MPPI 단계별 시간 측정 결과 로깅
        self.logger.debug(
            f"[Controller] run_mppi Timings (Total: {(mppi_total_end - mppi_total_start)*1000.0:.2f} ms)\n"
            f"  ├─ 1. Prior Check: {(t_end_prior - t_start_prior)*1000.0:.2f} ms\n"
            f"  ├─ 2. Sampling:    {(t_end_sample - t_start_sample)*1000.0:.2f} ms\n"
            f"  ├─ 3. Rollout:     {(t_end_rollout - t_start_rollout)*1000.0:.2f} ms\n"
            f"  ├─ 4. Cost Comp:   {(t_end_cost - t_start_cost)*1000.0:.2f} ms (Details logged in compute_costs)\n"
            f"  ├─ 5. Weights:     {(t_end_weight - t_start_weight)*1000.0:.2f} ms\n"
            f"  ├─ 6. Update:      {(t_end_update - t_start_update)*1000.0:.2f} ms\n"
            f"  └─ 7. Viz Prep:    {(t_end_viz - t_start_viz)*1000.0:.2f} ms"
        )

        # Runner에게 (제어명령, 최적궤적, 샘플궤적) 반환
        return (best_control[0].item(), best_control[1].item()), \
               optimal_traj_local, \
               sampled_trajs_local_subset

