#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import math
import traceback
import time
import threading # Matplotlib 연동을 위해 추가

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- Matplotlib 시각화 라이브러리 ---
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# ---------------------------------

# ==============================================================================
# --- ROS2 Node ---
# ==============================================================================

class MPPIBevPlanner(Node):
    def __init__(self):
        super().__init__('mppi_bev_planner_viz_node') # 노드 이름 변경

        # --- ROS 2 파라미터 선언 ---
        
        # 1. BEV Map 파라미터 (bev_map.py와 동일해야 함)
        self.declare_parameter('grid_resolution', 0.1)  # meters per cell
        self.declare_parameter('grid_size_x', 15.0)     # total width in meters
        self.declare_parameter('grid_size_y', 15.0)     # total height in meters
        self.declare_parameter('inflation_radius', 0.3) # meters
        
        # 2. 로봇 제어 파라미터
        self.declare_parameter('max_linear_velocity', 0.6)  # m/s
        self.declare_parameter('min_linear_velocity', 0.0)  # m/s (후진 방지)
        self.declare_parameter('max_angular_velocity', 1.2) # rad/s
        self.declare_parameter('goal_threshold', 0.3)       # m

        # 3. MPPI 알고리즘 파라미터
        self.declare_parameter('mppi_k', 1000)      # K: 샘플 궤적 수
        self.declare_parameter('mppi_t', 40)        # T: 예측 시간 스텝 ( horizon )
        self.declare_parameter('mppi_dt', 0.1)      # dt: 예측 시간 간격 (control_timer와 맞추는 것이 좋음)
        self.declare_parameter('mppi_lambda', 1.0)  # Lambda: 온도 파라미터 (클수록 스무딩)
        self.declare_parameter('mppi_sigma_v', 0.1) # 선속도 노이즈 표준편차
        self.declare_parameter('mppi_sigma_w', 0.2) # 각속도 노이즈 표준편차

        # 4. MPPI 비용 함수 가중치 (확장 포인트)
        self.declare_parameter('goal_cost_weight', 5.0)     # 목표 지점 비용 가중치
        self.declare_parameter('obstacle_cost_weight', 100.0) # 장애물 비용 가중치
        self.declare_parameter('control_cost_weight', 0.1)  # 제어 비용 가중치
        
        # 5. 시각화 파라미터
        self.declare_parameter('num_samples_to_plot', 50) # 시각화할 샘플 궤적 수

        # --- 파라미터 값 가져오기 ---
        # BEV
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        # Robot
        self.max_v = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.min_v = self.get_parameter('min_linear_velocity').get_parameter_value().double_value
        self.max_w = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.goal_threshold = self.get_parameter('goal_threshold').get_parameter_value().double_value
        # MPPI
        self.K = self.get_parameter('mppi_k').get_parameter_value().integer_value
        self.T = self.get_parameter('mppi_t').get_parameter_value().integer_value
        self.dt = self.get_parameter('mppi_dt').get_parameter_value().double_value
        self.lambda_ = self.get_parameter('mppi_lambda').get_parameter_value().double_value
        sigma_v = self.get_parameter('mppi_sigma_v').get_parameter_value().double_value
        sigma_w = self.get_parameter('mppi_sigma_w').get_parameter_value().double_value
        # Cost Weights
        self.goal_cost_w = self.get_parameter('goal_cost_weight').get_parameter_value().double_value
        self.obstacle_cost_w = self.get_parameter('obstacle_cost_weight').get_parameter_value().double_value
        self.control_cost_w = self.get_parameter('control_cost_weight').get_parameter_value().double_value
        # Viz
        self.num_samples_to_plot = self.get_parameter('num_samples_to_plot').get_parameter_value().integer_value


        # --- Grid 설정 (BEV 맵 처리를 위해) ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        # --- ROS2 Setup ---
        self.bev_sub = self.create_subscription(
            PointCloud2, '/bev_map', self.bev_map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 ---
        self.current_pose = None    # [x, y, yaw] (글로벌 좌표계)
        self.inflated_grid = None   # (cells_y, cells_x)
        self.costmap_tensor = None  # Costmap의 Torch 텐서 버전 (GPU 캐시용)
        
        # --- 웨이포인트 ---
        d1 = (0.0, 0.0)
        d2 = (2.7, 0)
        d3 = (2.433, 2.274)
        d4 = (-0.223, 2.4)
        d5 = (-2.55, 5.0)
        self.waypoints = [d1, d2, d3, d1, d4, d5]
        self.waypoint_index = 0
        
        # --- MPPI 핵심 변수 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        self.U = torch.zeros(self.T, 2, device=self.device, dtype=torch.float32)
        self.Sigma = torch.tensor([[sigma_v**2, 0.0],
                                    [0.0, sigma_w**2]], device=self.device, dtype=torch.float32)
        self.noise_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2, device=self.device), self.Sigma
        )

        # --- Matplotlib 시각화 데이터 및 잠금 ---
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []                     # 로봇의 전체 궤적 (글로벌)
        self.obstacle_points_local = np.array([])     # BEV 장애물 (로컬)
        self.latest_local_goal = np.array([])         # 로컬 목표 지점 (로컬)
        self.latest_optimal_trajectory_local = np.array([]) # MPPI 최적 궤적 (로컬)
        self.latest_sampled_trajectories_local = np.array([]) # MPPI 샘플 궤적 다발 (로컬)

        # 제어 루프 타이머
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        self.get_logger().info("✅ MPPI BEV Planner (with Matplotlib) has started.")
        self.get_logger().info(f"  Samples K={self.K}, Horizon T={self.T}, dt={self.dt}")

    # --- Odometry 콜백 ---
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        with self.plot_data_lock:
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y]) # 시각화용 궤적 저장

    # --- BEV 맵 콜백 (Costmap 생성) ---
    def bev_map_callback(self, msg: PointCloud2):
        """
        /bev_map 토픽을 구독하여 Costmap을 생성하고 팽창시킴.
        시각화를 위해 장애물 포인트도 저장함.
        """
        try:
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            obstacle_points_local = [] # ★ 시각화용 장애물 리스트
            
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                grid_c, grid_r = self.world_to_grid_idx_numpy(x, y) # Numpy용
                
                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    grid[grid_r, grid_c] = 255
                    obstacle_points_local.append([x, y]) # ★ 시각화용으로 저장
            
            self.inflated_grid = cv2.dilate(grid, self.inflation_kernel)
            self.costmap_tensor = torch.from_numpy(self.inflated_grid).to(self.device).float()

            # ★ 시각화용 데이터 업데이트
            with self.plot_data_lock:
                self.obstacle_points_local = np.array(obstacle_points_local)

        except Exception as e:
            self.get_logger().error(f"BEV map processing error: {e}\n{traceback.format_exc()}")

    # --- 좌표 변환 헬퍼 (Numpy) ---
    def world_to_grid_idx_numpy(self, x, y):
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    # --- 로봇 정지 ---
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.U.zero_() # MPPI 제어 시퀀스 리셋
        
        # ★ 시각화 데이터 클리어
        with self.plot_data_lock:
            self.latest_local_goal = np.array([])
            self.latest_optimal_trajectory_local = np.array([])
            self.latest_sampled_trajectories_local = np.array([])


    # ==============================================================================
    # --- MPPI 핵심 로직 (Torch) ---
    # ==============================================================================

    def motion_model(self, states, controls):
        v = controls[:, 0]
        w = controls[:, 1]
        yaw = states[:, 2]

        x_next = states[:, 0] + v * torch.cos(yaw) * self.dt
        y_next = states[:, 1] + v * torch.sin(yaw) * self.dt
        yaw_next = yaw + w * self.dt
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        return torch.stack([x_next, y_next, yaw_next], dim=1)

    def world_to_grid_idx_torch(self, x, y):
        grid_c = ((x - self.grid_origin_x) / self.grid_resolution).long()
        grid_r = ((y - self.grid_origin_y) / self.grid_resolution).long()
        return grid_r, grid_c

    def compute_costs(self, trajectories, local_goal_tensor, perturbed_controls):
        # 1. 목표 지점 비용 (Goal Cost)
        final_states_xy = trajectories[:, -1, :2] # (K, 2)
        goal_cost = torch.linalg.norm(final_states_xy - local_goal_tensor, dim=1)
        
        # 2. 장애물 비용 (Obstacle Cost)
        traj_x = trajectories[..., 0] # (K, T)
        traj_y = trajectories[..., 1] # (K, T)
        grid_r, grid_c = self.world_to_grid_idx_torch(traj_x, traj_y)

        out_of_bounds = (grid_c < 0) | (grid_c >= self.cells_x) | (grid_r < 0) | (grid_r >= self.cells_y)
        grid_r_clamped = torch.clamp(grid_r, 0, self.cells_y - 1)
        grid_c_clamped = torch.clamp(grid_c, 0, self.cells_x - 1)
        obstacle_costs_per_step = self.costmap_tensor[grid_r_clamped, grid_c_clamped] / 255.0
        obstacle_costs_per_step[out_of_bounds] = 1.0
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

    def compute_heuristic_prior(self, local_goal_tensor):
        self.get_logger().info("Prior is zero. Generating new goal-directed prior.")
        angle_to_goal = torch.atan2(local_goal_tensor[1], local_goal_tensor[0])
        w = torch.clamp(angle_to_goal * 2.0, -self.max_w, self.max_w)
        v_val = self.max_v * 0.5
        if torch.abs(angle_to_goal) > (math.pi / 4.0):
             v_val = 0.0
        control_prior = torch.tensor([v_val, w.item()], device=self.device, dtype=torch.float32)
        return control_prior.expand(self.T, 2)

    def run_mppi(self, local_goal_tensor):
        """
        MPPI 컨트롤러의 핵심 로직.
        Args:
            local_goal_tensor: (2,) [x, y] 로컬 목표 (Torch 텐서)
        """
        
        # 0. 준비
        if self.costmap_tensor is None:
            self.get_logger().warn("MPPI: Costmap is not ready.")
            return self.stop_robot()
            
        start_time = time.time()
        
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
        costs = self.compute_costs(trajectories, local_goal_tensor, perturbed_controls) # (K,)

        # 5. 비용 기반 가중치 계산 (Softmax)
        costs_normalized = costs - torch.min(costs)
        weights = torch.exp(-1.0 / self.lambda_ * costs_normalized)
        weights /= (torch.sum(weights) + 1e-9) # (K,)

        # 6. 가중 평균을 사용하여 평균 제어 시퀀스(U) 업데이트
        weighted_noise = torch.einsum('k,ktu->tu', weights, noise)
        self.U = self.U + weighted_noise

        # 7. ★★★ 시각화 데이터 저장 ★★★
        
        # 7-1. 방금 계산한 *최적의* 제어 시퀀스(U)를 롤아웃하여 '최적 궤적' 생성
        optimal_traj_local = torch.zeros(self.T, 3, device=self.device, dtype=torch.float32)
        current_state_optimal = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        for t in range(self.T):
            control_optimal = self.U[t, :].unsqueeze(0)
            next_state_optimal = self.motion_model(current_state_optimal, control_optimal)
            optimal_traj_local[t, :] = next_state_optimal.squeeze()
            current_state_optimal = next_state_optimal
            
        # 7-2. Plot Lock을 잡고 데이터 복사 (CPU로)
        with self.plot_data_lock:
            self.latest_local_goal = local_goal_tensor.cpu().numpy()
            self.latest_optimal_trajectory_local = optimal_traj_local.cpu().numpy()
            
            # 7-3. (K)개의 궤적 중 (N)개만 랜덤 샘플링하여 저장
            if self.K > self.num_samples_to_plot:
                indices = torch.randint(0, self.K, (self.num_samples_to_plot,))
                self.latest_sampled_trajectories_local = trajectories[indices, ...].cpu().numpy()
            else:
                self.latest_sampled_trajectories_local = trajectories.cpu().numpy()
        # ★★★ 시각화 데이터 저장 끝 ★★★
        

        # 8. 제어 시퀀스 시프트 (다음 스텝 준비)
        best_control = self.U[0, :] # (2,)
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1, :] = 0.0 # 마지막 스텝은 0으로 리셋

        # 9. 최적의 제어 명령 반환
        twist = Twist()
        twist.linear.x = best_control[0].item()
        twist.angular.z = best_control[1].item()
        
        elapsed_time = (time.time() - start_time) * 1000 # ms
        self.get_logger().info(f"MPPI: v={twist.linear.x:.2f}, w={twist.angular.z:.2f} | Time: {elapsed_time:.1f}ms")

        return twist

    # --- 메인 제어 루프 ---
    def control_callback(self):
        
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("🎉 All waypoints reached! Stopping.")
                self.stop_robot()
                self.control_timer.cancel()
                return

            current_x, current_y, current_yaw = self.current_pose
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp[0], target_wp[1]

            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                self.stop_robot() 
                return

            # 글로벌 목표 -> 로컬 목표 변환
            dx_global = target_x - current_x
            dy_global = target_y - current_y
            local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            # MPPI 실행 (★ 텐서를 넘겨주도록 수정)
            local_goal_tensor = torch.tensor(
                [local_target_x, local_target_y], device=self.device, dtype=torch.float32
            )
            twist_cmd = self.run_mppi(local_goal_tensor)
            
            self.cmd_pub.publish(twist_cmd)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.stop_robot()
            
    def destroy_node(self):
        self.get_logger().info("Shutting down... Stopping robot.")
        self.stop_robot()
        super().destroy_node()

# ==============================================================================
# --- Matplotlib 시각화 함수 ---
# (A* 코드의 시각화 로직을 MPPI 데이터에 맞게 수정한 버전)
# ==============================================================================

def update_plot(frame, node: MPPIBevPlanner, ax, traj_line,
                current_point, heading_line, goal_point,
                reached_wps_plot, pending_wps_plot, obstacle_scatter,
                optimal_traj_line, sampled_traj_lines):
    
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        # MPPI 데이터 (로컬)
        optimal_traj_local = node.latest_optimal_trajectory_local.copy()
        sampled_trajs_local = node.latest_sampled_trajectories_local.copy()
        goal_local = node.latest_local_goal.copy()
        obstacles_local = node.obstacle_points_local.copy()
        # 글로벌 웨이포인트
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj or pose is None:
        return []

    # --- 글로벌 웨이포인트 업데이트 ---
    reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:]
    if reached_wps.size > 0:
        reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else:
        reached_wps_plot.set_data([], [])
    if pending_wps.size > 0:
        pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else:
        pending_wps_plot.set_data([], [])

    # --- 로봇 궤적 및 자세 업데이트 ---
    traj_arr = np.array(traj)
    traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0])

    current_x, current_y, current_yaw = pose
    current_point.set_data([-current_y], [current_x])
    heading_len = 0.5
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

    # --- 로컬 플랜 및 장애물 -> 글로벌 변환 ---
    rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                            [math.sin(current_yaw),  math.cos(current_yaw)]])
    
    # 장애물 포인트 (로컬 -> 글로벌)
    if obstacles_local.size > 0:
        obstacles_global = (rot_matrix @ obstacles_local.T).T + np.array([current_x, current_y])
        obstacle_scatter.set_offsets(np.c_[-obstacles_global[:, 1], obstacles_global[:, 0]])
    else:
        obstacle_scatter.set_offsets(np.empty((0, 2)))
    
    # 로컬 골 (로컬 -> 글로벌)
    if goal_local.size > 0:
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])
    else:
        goal_point.set_data([], [])

    # ★ 최적 궤적 (로컬 -> 글로벌)
    if optimal_traj_local.size > 0:
        optimal_traj_global = (rot_matrix @ optimal_traj_local[:, :2].T).T + np.array([current_x, current_y])
        optimal_traj_line.set_data(-optimal_traj_global[:, 1], optimal_traj_global[:, 0])
    else:
        optimal_traj_line.set_data([], [])

    # ★ 샘플링된 궤적 다발 (로컬 -> 글로벌)
    if sampled_trajs_local.size > 0:
        for i, line in enumerate(sampled_traj_lines):
            if i < len(sampled_trajs_local):
                traj_local = sampled_trajs_local[i] # (T, 3)
                traj_global = (rot_matrix @ traj_local[:, :2].T).T + np.array([current_x, current_y])
                line.set_data(-traj_global[:, 1], traj_global[:, 0])
            else:
                line.set_data([], []) # 남는 라인 아티스트 클리어
    else:
        for line in sampled_traj_lines:
            line.set_data([], [])

    artists = [traj_line, current_point, heading_line, goal_point,
               reached_wps_plot, pending_wps_plot, obstacle_scatter, optimal_traj_line]
    artists.extend(sampled_traj_lines)
    
    return artists


def main(args=None):
    rclpy.init(args=args)
    node = MPPIBevPlanner()

    # ROS 2 노드를 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # --- Matplotlib 설정 (A* 코드와 거의 동일) ---
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_title('Real-time MPPI BEV Planner', fontsize=14)
    ax.set_xlabel('-Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.5
    x_max, y_max = wps_array.max(axis=0) + 1.5
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(-y_max, -y_min)
    
    # --- 플롯 아티스트 생성 ---
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    
    obstacle_scatter = ax.scatter([], [], c='red', s=2, alpha=0.4, label='BEV Obstacles')
    
    # --- MPPI 전용 아티스트 ---
    optimal_traj_line, = ax.plot([], [], 'm-', lw=2.5, zorder=10,
                                 label=f'Optimal Trajectory (U)')
    
    sampled_traj_lines = []
    for i in range(node.num_samples_to_plot):
        label = 'Sampled Trajectories (K)' if i == 0 else None
        line, = ax.plot([], [], 'c-', lw=0.5, alpha=0.2, zorder=5, label=label)
        sampled_traj_lines.append(line)
    
    ax.legend(loc='upper right', fontsize=9)
    
    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax, traj_line,
               current_point, heading_line, goal_point,
               reached_wps_plot, pending_wps_plot, obstacle_scatter,
               optimal_traj_line, sampled_traj_lines),
        interval=100, blit=True
    )

    try:
        plt.show() # 메인 스레드에서 Matplotlib 실행 (블로킹)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Matplotlib and ROS node.")
        # Matplotlib이 닫히면 ROS 노드 종료
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == '__main__':
    main()
