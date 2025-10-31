#!/usr/bin/env python3
# runner.py

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
import threading
from types import SimpleNamespace

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- 모듈화된 코드 임포트 ---
from optimized_controller import MPPIController
# from visualizer import setup_visualization
from bold_visualizer import setup_visualization
# -----------------------------

class MPPIBevPlanner(Node):
    """
    (수정) MPPI 플래너를 실행하는 메인 ROS 2 노드.
    (신규) Semantic BEV Map을 구독하여 시맨틱 비용 맵을 생성하고,
    이를 MPPI 컨트롤러에 전달하여 비용 함수에 반영합니다.
    """
    def __init__(self):
        super().__init__('mppi_bev_planner_viz_node')

        # --- 1. ROS 2 파라미터 선언 ---
        # (기존 파라미터...)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size_x', 40.0)
        self.declare_parameter('grid_size_y', 30.0)
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('max_linear_velocity', 0.6)
        # ... (기존 파라미터들) ...
        self.declare_parameter('goal_cost_weight', 95.0)
        self.declare_parameter('obstacle_cost_weight', 244.0)
        self.declare_parameter('control_cost_weight', 0.1)
        self.declare_parameter('num_samples_to_plot', 50)
        
        # (신규) 충돌 감지기 파라미터
        self.declare_parameter('collision_check_distance', 0.5) 
        self.declare_parameter('collision_check_width', 0.25)   
        self.declare_parameter('collision_cost_threshold', 250.0) 

        # (신규) ★ 시맨틱 비용 파라미터 ★
        self.declare_parameter('semantic_bev_topic', '/semantic_bev_map')
        # Cityscapes 기준 예시: 1(인도), 11(사람), 13(차)
        self.declare_parameter('prefer_labels', [1])       # 선호하는 라벨 (예: 인도)
        self.declare_parameter('avoid_labels', [11, 12, 13, 14, 15, 17, 18]) # 회피 라벨 (사람, 차 등)
        self.declare_parameter('cost_for_prefer', 1.0)     # 선호 라벨 비용 (낮을수록 좋음)
        self.declare_parameter('cost_for_avoid', 255.0)    # 회피 라벨 비용 (장애물과 동일)
        self.declare_parameter('cost_for_default', 20.0)   # 그 외 라벨 (예: 도로)
        self.declare_parameter('semantic_cost_weight', 150.0) # 시맨틱 비용의 전체 가중치

        # --- 2. 파라미터 값 가져오기 ---
        # (기존 파라미터...)
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        # ... (기존 파라미터들) ...
        self.obstacle_cost_w = self.get_parameter('obstacle_cost_weight').get_parameter_value().double_value
        self.control_cost_w = self.get_parameter('control_cost_weight').get_parameter_value().double_value
        self.num_samples_to_plot = self.get_parameter('num_samples_to_plot').get_parameter_value().integer_value
        
        # (신규) 충돌 감지기 파라미터
        self.collision_check_distance = self.get_parameter('collision_check_distance').get_parameter_value().double_value
        self.collision_check_width = self.get_parameter('collision_check_width').get_parameter_value().double_value
        self.collision_cost_threshold = self.get_parameter('collision_cost_threshold').get_parameter_value().double_value

        # (신규) ★ 시맨틱 비용 파라미터 가져오기 ★
        semantic_bev_topic = self.get_parameter('semantic_bev_topic').value
        prefer_labels_list = self.get_parameter('prefer_labels').get_parameter_value().integer_array_value
        avoid_labels_list = self.get_parameter('avoid_labels').get_parameter_value().integer_array_value
        self.cost_for_prefer = self.get_parameter('cost_for_prefer').get_parameter_value().double_value
        self.cost_for_avoid = self.get_parameter('cost_for_avoid').get_parameter_value().double_value
        self.cost_for_default = self.get_parameter('cost_for_default').get_parameter_value().double_value
        self.semantic_cost_weight = self.get_parameter('semantic_cost_weight').get_parameter_value().double_value
        
        # 빠른 조회를 위해 Set으로 변환
        self.prefer_labels_set = set(prefer_labels_list)
        self.avoid_labels_set = set(avoid_labels_list)

        # --- 3. Grid 및 BEV 설정 ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        # (충돌 감지기 ROI 계산 - 기존과 동일)
        self.robot_grid_c = int((0.0 - self.grid_origin_x) / self.grid_resolution)
        self.robot_grid_r = int((0.0 - self.grid_origin_y) / self.grid_resolution)
        check_dist_cells = int(self.collision_check_distance / self.grid_resolution)
        check_width_cells = int(self.collision_check_width / self.grid_resolution)
        self.roi_r_start = max(0, self.robot_grid_r - check_width_cells // 2)
        self.roi_r_end = min(self.cells_y, self.robot_grid_r + check_width_cells // 2)
        self.roi_c_start = max(0, self.robot_grid_c) # 로봇 위치부터
        self.roi_c_end = min(self.cells_x, self.robot_grid_c + check_dist_cells) # 전방으로
        
        self.get_logger().info(
            f"Collision checker ROI (grid indices):\n"
            f"  Rows (width): {self.roi_r_start} to {self.roi_r_end}\n"
            f"  Cols (dist):  {self.roi_c_start} to {self.roi_c_end}"
        )

        
        # --- 4. ROS2 Setup ---
        self.bev_sub = self.create_subscription(
            PointCloud2, '/bev_map', self.bev_map_callback, 10) # 장애물 BEV
        
        # (신규) ★ 시맨틱 BEV 구독자 ★
        self.sem_bev_sub = self.create_subscription(
            PointCloud2, semantic_bev_topic, self.semantic_bev_callback, 10)
            
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 5. 상태 변수 ---
        self.current_pose = None    # [x, y, yaw] (글로벌 좌표계)
        self.costmap_tensor = None  # (장애물) Costmap의 Torch 텐서
        self.semantic_costmap_tensor = None # (신규) ★ 시맨틱 Costmap의 Torch 텐서 ★
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        self.collision_detected_last_step = False
        self.is_shutting_down = False 
        
        # --- 6. 웨이포인트 (수정) ---
        # (기존과 동일 - (x, y)와 yaw 분리)
        wp_data = [
            {'pos': (0.2548, -0.1488), 'ori': (0.9997, 0.0059, 0.0071, 0.0208)},
            # ... (나머지 웨이포인트) ...
            {'pos': (41.2895, -28.0243), 'ori': (0.7745, 0.0003, 0.0046, 0.6325)},
        ]
        self.waypoints = [] 
        self.waypoint_yaws = [] 
        for wp in wp_data:
            pos = wp['pos']
            ori = wp['ori']
            q = SimpleNamespace(w=ori[0], x=ori[1], y=ori[2], z=ori[3])
            yaw = self.quaternion_to_yaw(q)
            self.waypoints.append((pos[0], pos[1])) 
            self.waypoint_yaws.append(yaw)          
        
        self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints (x, y) and {len(self.waypoint_yaws)} yaws.")
        self.waypoint_index = 0
        
        # --- 7. Matplotlib 시각화 데이터 및 잠금 ---
        # (기존과 동일)
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.obstacle_points_local = np.array([])
        self.latest_local_goal = np.array([])
        self.latest_optimal_trajectory_local = np.array([])
        self.latest_sampled_trajectories_local = np.array([])

        # --- 8. ★ MPPI 컨트롤러 모듈 생성 (수정) ★ ---
        self.controller = MPPIController(
            logger=self.get_logger(),
            device=self.device,
            K=self.K, T=self.T, dt=self.dt, lambda_=self.lambda_,
            sigma_v=sigma_v, sigma_w=sigma_w,
            min_v=self.min_v, max_v=self.max_v, max_w=self.max_w,
            goal_cost_w=self.goal_cost_w,
            obstacle_cost_w=self.obstacle_cost_w,
            control_cost_w=self.control_cost_w,
            semantic_cost_w=self.semantic_cost_weight, # (신규) ★ 시맨틱 가중치 전달 ★
            grid_resolution=self.grid_resolution,
            grid_origin_x=self.grid_origin_x,
            grid_origin_y=self.grid_origin_y,
            cells_x=self.cells_x,
            cells_y=self.cells_y,
            num_samples_to_plot=self.num_samples_to_plot
        )

        # --- 9. 제어 루프 타이머 ---
        # (기존과 동일)
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        # --- 10. 로깅 타이머 ---
        # (기존과 동일)
        self.last_control_callback_time_ms = 0.0
        self.last_mppi_run_time_ms = 0.0
        self.last_bev_map_callback_time_ms = 0.0
        self.last_sem_bev_callback_time_ms = 0.0 # (신규) 시맨틱 콜백 시간
        self.current_status = "Initializing" 
        self.logging_timer = self.create_timer(1.0, self.logging_callback) 
        
        self.get_logger().info("✅ MPPI BEV Planner (with Semantic Cost) has started.")

    
    def logging_callback(self):
        """1초마다 현재 상태와 성능을 로깅합니다."""
        
        with self.plot_data_lock:
            status = self.current_status
            mppi_time = self.last_mppi_run_time_ms
            control_time = self.last_control_callback_time_ms
            bev_time = self.last_bev_map_callback_time_ms
            sem_bev_time = self.last_sem_bev_callback_time_ms # (신규)
            
            other_control_time = control_time - mppi_time
        
        loop_slack_ms = (self.dt * 1000.0) - mppi_time 

        log_msg = (
            f"\n--- MPPI Status (1s Heartbeat) ---\n"
            f"  Status: {status}\n"
            f"  Loop Slack: {loop_slack_ms:6.1f} ms (Target: {self.dt * 1000.0:.0f} ms)\n"
            f"  Performance (Last call, ms):\n"
            f"    ├─ MPPI.run_mppi(): {mppi_time:8.2f} ms\n"
            f"    ├─ Other Control Logic: {other_control_time:4.2f} ms\n"
            f"    ├─ Total Control Callback: {control_time:5.2f} ms\n"
            f"    ├─ Obstacle BEV Callback: {bev_time:6.2f} ms\n" # (수정)
            f"    └─ Semantic BEV Callback: {sem_bev_time:6.2f} ms" # (신규)
        )
        self.get_logger().info(log_msg)


    # --- ROS 콜백 함수들 ---

    def quaternion_to_yaw(self, q):
        # ... (기존과 동일)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        # ... (기존과 동일)
        return math.atan2(math.sin(angle), math.cos(angle))

    def odom_callback(self, msg: Odometry):
        # ... (기존과 동일)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        with self.plot_data_lock: 
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y])

    def bev_map_callback(self, msg: PointCloud2):
        """ (장애물) BEV PointCloud를 (장애물) Costmap 텐서로 변환합니다. """
        start_time = time.perf_counter() 
        try:
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            obstacle_points_local = []
            
            # (최적화) x, y 필드만 읽음
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                grid_c, grid_r = self.world_to_grid_idx_numpy(x, y)
                
                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    grid[grid_r, grid_c] = 255
                    obstacle_points_local.append([x, y])
            
            inflated_grid_np = cv2.dilate(grid, self.inflation_kernel)
            
            self.costmap_tensor = torch.from_numpy(inflated_grid_np).to(self.device).float()
            with self.plot_data_lock:
                self.obstacle_points_local = np.array(obstacle_points_local)

        except Exception as e:
            self.get_logger().error(f"Obstacle BEV map processing error: {e}\n{traceback.format_exc()}")
        finally:
            end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_bev_map_callback_time_ms = (end_time - start_time) * 1000.0

    # (신규) ★ 시맨틱 BEV 콜백 ★
    def semantic_bev_callback(self, msg: PointCloud2):
        """
        Semantic BEV PointCloud를 (시맨틱) Costmap 텐서로 변환합니다.
        - msg: (x, y, z, rgb, label) 필드를 가진 포인트 클라우드
        - output: self.semantic_costmap_tensor (dense, cells_y x cells_x)
        """
        start_time = time.perf_counter()
        try:
            # 1. 기본 비용(default_cost)으로 채워진 조밀한(dense) 그리드 생성
            sem_grid_np = np.full(
                (self.cells_y, self.cells_x), 
                self.cost_for_default, 
                dtype=np.float32
            )
            
            # 2. (최적화) x, y, label 필드만 읽음 (label은 5번째 필드)
            # semantic_bev_node.py가 'label'을 5번째 float32로 저장함
            for point in pc2.read_points(msg, field_names=('x', 'y', 'label'), skip_nans=True):
                x, y, label_float = point[0], point[1], point[2]
                label = int(label_float) # float -> int
                
                # 3. 월드 좌표 -> 그리드 인덱스
                grid_c, grid_r = self.world_to_grid_idx_numpy(x, y)

                # 4. 그리드 범위 내인지 확인
                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    # 5. 라벨에 따라 비용 할당
                    cost = self.cost_for_default
                    if label in self.prefer_labels_set:
                        cost = self.cost_for_prefer
                    elif label in self.avoid_labels_set:
                        cost = self.cost_for_avoid
                    
                    # 6. 그리드에 비용 "페인팅"
                    sem_grid_np[grid_r, grid_c] = cost

            # 7. NumPy 그리드를 GPU 텐서로 변환하여 저장
            self.semantic_costmap_tensor = torch.from_numpy(sem_grid_np).to(self.device).float()

        except Exception as e:
            self.get_logger().error(f"Semantic BEV map processing error: {e}\n{traceback.format_exc()}")
        finally:
            end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_sem_bev_callback_time_ms = (end_time - start_time) * 1000.0


    def world_to_grid_idx_numpy(self, x, y):
        # ... (기존과 동일)
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    def stop_robot(self):
        # ... (기존과 동일 - is_shutting_down 플래그 포함)
        if self.is_shutting_down:
            return
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.controller.reset()
        with self.plot_data_lock:
            self.latest_local_goal = np.array([])
            self.latest_optimal_trajectory_local = np.array([])
            self.latest_sampled_trajectories_local = np.array([])

    
    def check_for_imminent_collision(self) -> bool:
        """ (장애물) Costmap을 기반으로 즉각적인 충돌 확인 """
        if self.costmap_tensor is None:
            return False 
            
        try:
            # (장애물) Costmap을 사용
            danger_zone = self.costmap_tensor[
                self.roi_r_start : self.roi_r_end,
                self.roi_c_start : self.roi_c_end
            ]
            
            if torch.any(danger_zone >= self.collision_cost_threshold):
                return True
                
        except Exception as e:
            self.get_logger().error(f"Collision check error: {e}\n{traceback.format_exc()}")
            return True 
            
        return False

    # --- 메인 제어 루프 (수정) ---

    def control_callback(self):
        """
        메인 제어 루프.
        (수정) (장애물) Costmap과 (시맨틱) Costmap을 모두 컨트롤러에 전달
        """
        
        if self.is_shutting_down:
            return
            
        control_start_time = time.perf_counter()
        
        # (수정) ★ 맵 2개(장애물, 시맨틱)와 Odom을 모두 기다림 ★
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...", throttle_duration_sec=1.0)
            with self.plot_data_lock: self.current_status = "Waiting for Odometry"
            return
            
        if self.costmap_tensor is None:
            self.get_logger().warn("Waiting for Obstacle BEV map...", throttle_duration_sec=1.0)
            with self.plot_data_lock: self.current_status = "Waiting for Obstacle Map"
            return
            
        if self.semantic_costmap_tensor is None:
            self.get_logger().warn("Waiting for Semantic BEV map...", throttle_duration_sec=1.0)
            with self.plot_data_lock: self.current_status = "Waiting for Semantic Map"
            return
        # -----------------------------------------------------------

        try:
            # --- 0. 즉각적인 충돌 감지 (장애물 맵 기준) ---
            if self.check_for_imminent_collision():
                if not self.collision_detected_last_step:
                    self.get_logger().warn("🛑 IMMINENT OBSTACLE DETECTED! Stopping robot.")
                
                self.stop_robot()
                with self.plot_data_lock: self.current_status = "OBSTACLE STOP"
                self.collision_detected_last_step = True
                return 
            
            if self.collision_detected_last_step:
                self.get_logger().info("✅ Obstacle clear. Resuming navigation.")
                self.collision_detected_last_step = False
            # ---------------------------------

            # 1. 웨이포인트 도달 확인
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("🎉 All waypoints reached! Stopping.")
                # ... (이하 동일)
                with self.plot_data_lock: self.current_status = "All waypoints reached" 
                self.stop_robot()
                self.control_timer.cancel()
                self.logging_timer.cancel() 
                return

            with self.plot_data_lock:
                self.current_status = f"Running to WP {self.waypoint_index+1}/{len(self.waypoints)}"

            # 2. 현재 상태 및 목표 설정
            current_x, current_y, current_yaw = self.current_pose
            target_wp_xy = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp_xy[0], target_wp_xy[1]
            target_yaw = self.waypoint_yaws[self.waypoint_index]

            # 3. 2단계 로직 (위치 접근 -> Yaw 정렬)
            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # --- 1단계: 위치 접근 (MPPI) ---
            if distance_to_goal > self.goal_threshold:
                # 4. 글로벌 목표 -> 로컬 목표 변환
                # ... (기존과 동일)
                dx_global = target_x - current_x
                dy_global = target_y - current_y
                local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
                local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
                
                local_goal_tensor = torch.tensor(
                    [local_target_x, local_target_y], device=self.device, dtype=torch.float32
                )
                
                # 5. ★ MPPI 컨트롤러 실행 (수정) ★
                mppi_start_time = time.perf_counter()
                
                control_tuple, opt_traj_gpu, sampled_trajs_gpu = self.controller.run_mppi(
                    local_goal_tensor, 
                    self.costmap_tensor,            # (수정) 1. 장애물 맵
                    self.semantic_costmap_tensor  # (신규) 2. 시맨틱 맵
                )
                
                mppi_end_time = time.perf_counter()
                mppi_run_time_ms = (mppi_end_time - mppi_start_time) * 1000.0
                
                with self.plot_data_lock:
                    self.last_mppi_run_time_ms = mppi_run_time_ms
                
                # 6. 컨트롤러 실행 결과 처리
                if control_tuple is None: 
                    self.get_logger().warn("MPPI controller failed. Stopping.")
                    with self.plot_data_lock:
                        self.current_status = "Controller Failed (Maps?)" 
                    self.stop_robot()
                    return
                
                # 7. 시각화 데이터 업데이트
                # ... (기존과 동일)
                with self.plot_data_lock:
                    self.latest_local_goal = local_goal_tensor.cpu().numpy()
                    self.latest_optimal_trajectory_local = opt_traj_gpu.cpu().numpy()
                    self.latest_sampled_trajectories_local = sampled_trajs_gpu.cpu().numpy()
                
                # 8. 제어 명령 발행
                # ... (기존과 동일)
                v, w = control_tuple
                twist_cmd = Twist()
                twist_cmd.linear.x = v
                twist_cmd.angular.z = w
                self.cmd_pub.publish(twist_cmd)
            
            # --- 2단계: Yaw 정렬 (P제어) ---
            else:
                # ... (기존과 동일)
                yaw_error = self.normalize_angle(target_yaw - current_yaw)
                
                if abs(yaw_error) > self.yaw_threshold:
                    with self.plot_data_lock:
                        self.current_status = f"Aligning Yaw at WP {self.waypoint_index+1}"
                        self.latest_local_goal = np.array([])
                        self.latest_optimal_trajectory_local = np.array([])
                        self.latest_sampled_trajectories_local = np.array([])
                    
                    v = 0.0
                    w = self.yaw_p_gain * yaw_error
                    w = np.clip(w, -self.max_w, self.max_w)
                    
                    twist_cmd = Twist()
                    twist_cmd.linear.x = v
                    twist_cmd.angular.z = w
                    self.cmd_pub.publish(twist_cmd)
                
                else:
                    self.get_logger().info(f"✅ Waypoint {self.waypoint_index} (Position & Yaw) reached!")
                    self.waypoint_index += 1
                    self.stop_robot() 
                    return

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            with self.plot_data_lock:
                self.current_status = "ERROR in control loop" 
            self.stop_robot()
        finally:
            control_end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_control_callback_time_ms = (control_end_time - control_start_time) * 1000.0

            
    def destroy_node(self):
        # ... (기존과 동일)
        self.get_logger().info("Shutting down... Stopping robot.")
        self.is_shutting_down = True 
        if self.control_timer:
            self.control_timer.cancel()
        if self.logging_timer: 
            self.logging_timer.cancel()
        self.stop_robot()
        super().destroy_node()

# --- main 함수 ---

def main(args=None):
    # ... (기존과 동일)
    rclpy.init(args=args)
    node = MPPIBevPlanner()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        setup_visualization(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        node.get_logger().info("Matplotlib closed, shutting down ROS node.")
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == '__main__':
    main()


