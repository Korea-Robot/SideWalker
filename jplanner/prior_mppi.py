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

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# ==============================================================================
# --- ROS2 Node ---
# ==============================================================================

class MPPIBevPlanner(Node):
    def __init__(self):
        super().__init__('mppi_bev_planner_node')

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
        self.inflated_grid = None   # (cells_y, cells_x) (MPPI의 Costmap으로 사용)
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

        # 평균 제어 시퀀스 (v, w). (T, 2)
        self.U = torch.zeros(self.T, 2, device=self.device, dtype=torch.float32)
        
        # 제어 노이즈 공분산 (v, w)
        self.Sigma = torch.tensor([[sigma_v**2, 0.0],
                                    [0.0, sigma_w**2]], device=self.device, dtype=torch.float32)
        
        # 노이즈 샘플링을 위한 분포
        self.noise_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2, device=self.device), self.Sigma
        )

        # 제어 루프 타이머
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        self.get_logger().info("✅ MPPI BEV Planner Node (Full) has started.")
        self.get_logger().info(f"  Samples K={self.K}, Horizon T={self.T}, dt={self.dt}")
        self.get_logger().info(f"  Cost Weights: Goal={self.goal_cost_w}, Obstacle={self.obstacle_cost_w}")

    # --- Odometry 콜백 ---
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose = [x, y, yaw]

    # --- BEV 맵 콜백 (Costmap 생성) ---
    def bev_map_callback(self, msg: PointCloud2):
        """
        /bev_map 토픽을 구독하여 Costmap을 생성하고 팽창시킴.
        """
        try:
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                grid_c, grid_r = self.world_to_grid_idx_numpy(x, y) # Numpy용
                
                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    grid[grid_r, grid_c] = 255
            
            self.inflated_grid = cv2.dilate(grid, self.inflation_kernel)

            # MPPI가 사용할 수 있도록 Costmap을 Torch 텐서로 변환 (GPU로)
            self.costmap_tensor = torch.from_numpy(self.inflated_grid).to(self.device).float()

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
        # MPPI 제어 시퀀스도 리셋 (Cold Start 유발)
        self.U.zero_()

    # ==============================================================================
    # --- MPPI 핵심 로직 (Torch) ---
    # ==============================================================================

    def motion_model(self, states, controls):
        """
        로봇의 다음 상태를 예측 (K개의 궤적에 대해 병렬 처리)
        Args:
            states: (K, 3) 텐서 [x, y, yaw]
            controls: (K, 2) 텐서 [v, w]
        Returns:
            (K, 3) 텐서: 다음 스텝의 [x, y, yaw]
        """
        v = controls[:, 0]
        w = controls[:, 1]
        yaw = states[:, 2]

        x_next = states[:, 0] + v * torch.cos(yaw) * self.dt
        y_next = states[:, 1] + v * torch.sin(yaw) * self.dt
        yaw_next = yaw + w * self.dt
        
        # Yaw를 -pi ~ +pi 범위로 정규화
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        return torch.stack([x_next, y_next, yaw_next], dim=1)

    def world_to_grid_idx_torch(self, x, y):
        """
        월드 좌표(m) 텐서를 그리드 인덱스(r, c) 텐서로 변환
        Args:
            x: (K, T) 텐서
            y: (K, T) 텐서
        Returns:
            grid_r, grid_c (K, T) 텐서
        """
        grid_c = ((x - self.grid_origin_x) / self.grid_resolution).long()
        grid_r = ((y - self.grid_origin_y) / self.grid_resolution).long()
        return grid_r, grid_c

    def compute_costs(self, trajectories, local_goal_tensor, perturbed_controls):
        """
        K개의 궤적에 대한 비용을 계산 (병렬 처리)
        
        Args:
            trajectories: (K, T, 3) 텐서 [x, y, yaw]
            local_goal_tensor: (2,) 텐서 [x, y]
            perturbed_controls: (K, T, 2) 텐서 [v, w]
            
        Returns:
            (K,) 텐서: 각 궤적의 총 비용
            
        --- 
        ★★★ 확장 포인트 ★★★
        향후 Semantic BEV Map (label_bev_tensor)이 있다면,
        이 함수에 인자로 추가하고,
        '3. 장애물 비용 (Obstacle Cost)' 섹션에서 
        label_bev_tensor를 샘플링하여 
        'pedestrian_cost', 'car_cost' 등을 추가하면 됩니다.
        ---
        """
        
        # 1. 목표 지점 비용 (Goal Cost)
        # 궤적의 *마지막* 지점과 로컬 목표 지점 간의 거리
        final_states_xy = trajectories[:, -1, :2] # (K, 2)
        goal_cost = torch.linalg.norm(final_states_xy - local_goal_tensor, dim=1)
        
        # 2. 장애물 비용 (Obstacle Cost)
        # 궤적의 모든 (x, y) 좌표를 그리드 인덱스로 변환
        traj_x = trajectories[..., 0] # (K, T)
        traj_y = trajectories[..., 1] # (K, T)
        grid_r, grid_c = self.world_to_grid_idx_torch(traj_x, traj_y)

        # 그리드 범위 밖으로 나간 궤적에 페널티
        out_of_bounds_x = (grid_c < 0) | (grid_c >= self.cells_x)
        out_of_bounds_y = (grid_r < 0) | (grid_r >= self.cells_y)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y

        # 유효한 인덱스만 클램핑 (범위 밖 샘플링 방지)
        grid_r_clamped = torch.clamp(grid_r, 0, self.cells_y - 1)
        grid_c_clamped = torch.clamp(grid_c, 0, self.cells_x - 1)

        # Costmap에서 비용 샘플링
        obstacle_costs_per_step = self.costmap_tensor[grid_r_clamped, grid_c_clamped] # (K, T)
        
        # Costmap 값은 0~255이므로 0~1로 정규화
        obstacle_costs_per_step = obstacle_costs_per_step / 255.0
        
        # 범위 밖으로 나간 스텝에 대해 높은 비용 부여 (1.0 = 최대 장애물 비용)
        obstacle_costs_per_step[out_of_bounds] = 1.0

        # 시간에 대해 비용을 합산
        obstacle_cost = torch.sum(obstacle_costs_per_step, dim=1) # (K,)
        
        # 3. 제어 비용 (Control Cost)
        # 부드러운 제어를 위해 제어 입력(v, w) 자체에도 작은 비용 부여
        control_cost = torch.sum(torch.linalg.norm(perturbed_controls, dim=2), dim=1) # (K,)
        
        # 4. 총 비용 계산
        total_cost = (
            self.goal_cost_w * goal_cost +
            self.obstacle_cost_w * obstacle_cost +
            self.control_cost_w * control_cost
        )
        
        return total_cost # (K,)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 새롭게 추가된 함수 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    def compute_heuristic_prior(self, local_goal_tensor):
        """
        'Cold Start' (U가 0일 때)를 위해,
        로컬 목표 지점을 향해 (단순하게) 주행하는
        휴리스틱 제어 시퀀스(T, 2)를 생성합니다. (P 제어기)
        
        Args:
            local_goal_tensor: (2,) 텐서 [x, y]
        Returns:
            (T, 2) 텐서
        """
        self.get_logger().info("Prior is zero. Generating new goal-directed prior.")
        
        # 목표 지점까지의 각도 (로봇 기준)
        goal_x = local_goal_tensor[0]
        goal_y = local_goal_tensor[1]
        angle_to_goal = torch.atan2(goal_y, goal_x) # 0-dim 텐서

        # 간단한 P 제어기
        # 목표 각도에 비례하는 각속도 (max_w로 제한)
        w = torch.clamp(angle_to_goal * 2.0, -self.max_w, self.max_w) # 2.0은 임의의 P gain
        
        # 목표가 정면 근처에 있을 때만 전진
        v_val = self.max_v * 0.5 # 예: 최대 속도의 절반
        if torch.abs(angle_to_goal) > (math.pi / 4.0): # 목표가 45도 이상 빗나가면
             v_val = 0.0 # 일단 회전부터

        # (T, 2) 텐서 생성: T 스텝 내내 이 제어를 유지한다고 가정
        control_prior = torch.tensor([v_val, w.item()], device=self.device, dtype=torch.float32)
        prior_U = control_prior.expand(self.T, 2)
        
        return prior_U

    def run_mppi(self, local_goal):
        """
        MPPI 컨트롤러의 핵심 로직.
        Args:
            local_goal: (x, y) 로봇 기준 좌표계의 목표 지점
        Returns:
            Twist: 계산된 최적의 제어 명령
        """
        
        # 0. 준비
        if self.costmap_tensor is None:
            self.get_logger().warn("MPPI: Costmap is not ready.")
            return self.stop_robot()
            
        start_time = time.time()
        
        local_goal_tensor = torch.tensor(local_goal, device=self.device, dtype=torch.float32) # (2,)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ 수정된 부분 (Goal-Directed Prior) ★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # 현재 Prior(U)가 0인지 (즉, stop_robot() 직후인지) 확인
        if torch.all(self.U == 0.0):
            # 0이라면, 목표지향적인 새로운 Prior를 생성
            self.U = self.compute_heuristic_prior(local_goal_tensor)
        # 0이 아니라면, 이전 스텝의 'Warm Start' 값을 그대로 사용 (torch.roll된 상태)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # 1. (K)개의 노이즈가 추가된 제어 시퀀스(v, w) 샘플 생성
        # (K, T, 2) 형상의 노이즈 텐서 생성
        noise = self.noise_dist.sample((self.K, self.T))
        
        # 현재 평균 제어 시퀀스(U)에 노이즈 추가 (K, T, 2)
        perturbed_controls = self.U.unsqueeze(0) + noise
        
        # 제어 입력(v, w)을 로봇의 한계 내로 클램핑
        perturbed_controls[..., 0] = torch.clamp(
            perturbed_controls[..., 0], self.min_v, self.max_v
        )
        perturbed_controls[..., 1] = torch.clamp(
            perturbed_controls[..., 1], -self.max_w, self.max_w
        )

        # 2. (K)개의 궤적 시뮬레이션 (롤아웃)
        trajectories = torch.zeros(self.K, self.T, 3, device=self.device, dtype=torch.float32)
        
        # 모든 K개의 궤적은 (0, 0, 0)에서 시작 (로봇 기준 좌표계)
        current_states = torch.zeros(self.K, 3, device=self.device, dtype=torch.float32) 

        for t in range(self.T):
            current_controls = perturbed_controls[:, t, :] # (K, 2)
            next_states = self.motion_model(current_states, current_controls) # (K, 3)
            trajectories[:, t, :] = next_states
            current_states = next_states

        # 3. (K)개의 궤적에 대한 비용 계산
        costs = self.compute_costs(trajectories, local_goal_tensor, perturbed_controls) # (K,)

        # 4. 비용 기반 가중치 계산 (Softmax)
        costs_normalized = costs - torch.min(costs) # 수치 안정성
        weights = torch.exp(-1.0 / self.lambda_ * costs_normalized)
        weights = weights / (torch.sum(weights) + 1e-9) # (K,)

        # 5. 가중 평균을 사용하여 평균 제어 시퀀스(U) 업데이트
        # weights: (K,), noise: (K, T, 2) -> (T, 2)
        weighted_noise = torch.einsum('k,ktu->tu', weights, noise)
        self.U = self.U + weighted_noise

        # 6. 제어 시퀀스 시프트 (다음 스텝 준비)
        # 가장 첫 번째 제어(U[0])를 사용하고, U를 한 칸씩 당김
        best_control = self.U[0, :] # (2,)
        
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1, :] = 0.0 # 마지막 스텝은 0으로 리셋

        # 7. 최적의 제어 명령 반환
        twist = Twist()
        twist.linear.x = best_control[0].item()
        twist.angular.z = best_control[1].item()
        
        elapsed_time = (time.time() - start_time) * 1000 # ms
        self.get_logger().info(f"MPPI: v={twist.linear.x:.2f}, w={twist.angular.z:.2f} | Time: {elapsed_time:.1f}ms")

        return twist

    # --- 메인 제어 루프 ---
    def control_callback(self):
        
        # 1. 데이터 준비 확인
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            return

        try:
            # 2. 모든 웨이포인트 도달 확인
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("🎉 All waypoints reached! Stopping.")
                self.stop_robot()
                self.control_timer.cancel()
                return

            # 3. 글로벌 목표 및 현재 상태
            current_x, current_y, current_yaw = self.current_pose
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp[0], target_wp[1]

            # 4. 목표 도달 여부 확인 (글로벌 좌표계)
            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                self.stop_robot() # 다음 웨이포인트 전에 잠시 정지 (이때 U가 0이 됨)
                return

            # 5. 글로벌 목표 -> 로컬 목표 변환
            dx_global = target_x - current_x
            dy_global = target_y - current_y
            local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            # 6. MPPI 실행
            twist_cmd = self.run_mppi((local_target_x, local_target_y))
            
            # 7. 제어 명령 발행
            self.cmd_pub.publish(twist_cmd)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.stop_robot()
            
    def destroy_node(self):
        self.get_logger().info("Shutting down... Stopping robot.")
        self.stop_robot()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MPPIBevPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
