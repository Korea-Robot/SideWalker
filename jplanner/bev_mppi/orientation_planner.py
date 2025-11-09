





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

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- 모듈화된 코드 임포트 ---
from optimized_controller import MPPIController
# from visualizer import setup_visualization
from visualizer_orient import setup_visualization
#from bold_visualizer import setup_visualization
# -----------------------------


"""
역할: 메인 ROS 2 노드입니다.

모든 ROS 통신(Sub/Pub/Timer/Params), 콜백, 상태 관리를 담당합니다.

controller와 visualizer 모듈을 임포트하여 오케스트레이션(조율)합니다.

control_callback에서 controller.run_mppi를 호출하고, 반환된 데이터를 plot_data_lock을 통해 시각화 데이터로 넘겨줍니다.
"""
class MPPIBevPlanner(Node):
    """
    MPPI 플래너를 실행하는 메인 ROS 2 노드.
    ROS 통신, 상태 관리, 그리고 컨트롤러/시각화 모듈의 조율을 담당.
    """

    # --- (신규) 정적 헬퍼 함수 ---
    # __init__에서 웨이포인트 파싱을 위해 먼저 정의
    def quaternion_to_yaw_from_parts(self, w, x, y, z):
        """쿼터니언(w, x, y, z) 구성요소로부터 Yaw 각도를 계산합니다."""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def __init__(self):
        super().__init__('mppi_bev_planner_viz_node')

        # --- 1. ROS 2 파라미터 선언 ---
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size_x', 50.0)
        self.declare_parameter('grid_size_y', 30.0)
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('max_linear_velocity', 0.9)
        self.declare_parameter('min_linear_velocity', 0.15)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_threshold', 0.5)

        # (신규) 방향 정렬 파라미터
        self.declare_parameter('yaw_threshold', 0.4) # [rad] 약 5.7도
        self.declare_parameter('yaw_p_gain', 0.5)    # 방향 정렬 P 제어 게인
        self.declare_parameter('min_align_angular_velocity', 0.1) # [rad/s] 최소 회전 속도

        self.declare_parameter('mppi_k', 5000)
        self.declare_parameter('mppi_t', 40)
        self.declare_parameter('mppi_dt', 0.1)
        self.declare_parameter('mppi_lambda', 1.0)
        self.declare_parameter('mppi_sigma_v', 0.1)
        self.declare_parameter('mppi_sigma_w', 0.3)
        self.declare_parameter('goal_cost_weight', 95.0)
        self.declare_parameter('obstacle_cost_weight', 244.0)
        self.declare_parameter('control_cost_weight', 0.1)
        self.declare_parameter('num_samples_to_plot', 50)

        # 충돌 감지기 파라미터
        self.declare_parameter('collision_check_distance', 0.0)
        self.declare_parameter('collision_check_width', 0.25)
        self.declare_parameter('collision_cost_threshold', 250.0)

        # --- 2. 파라미터 값 가져오기 ---
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.max_v = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.min_v = self.get_parameter('min_linear_velocity').get_parameter_value().double_value
        self.max_w = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.goal_threshold = self.get_parameter('goal_threshold').get_parameter_value().double_value

        # (신규) 방향 정렬 파라미터 가져오기
        self.yaw_threshold = self.get_parameter('yaw_threshold').get_parameter_value().double_value
        self.yaw_p_gain = self.get_parameter('yaw_p_gain').get_parameter_value().double_value
        self.min_align_w = self.get_parameter('min_align_angular_velocity').get_parameter_value().double_value

        self.K = self.get_parameter('mppi_k').get_parameter_value().integer_value
        self.T = self.get_parameter('mppi_t').get_parameter_value().integer_value
        self.dt = self.get_parameter('mppi_dt').get_parameter_value().double_value
        self.lambda_ = self.get_parameter('mppi_lambda').get_parameter_value().double_value
        sigma_v = self.get_parameter('mppi_sigma_v').get_parameter_value().double_value
        sigma_w = self.get_parameter('mppi_sigma_w').get_parameter_value().double_value
        self.goal_cost_w = self.get_parameter('goal_cost_weight').get_parameter_value().double_value
        self.obstacle_cost_w = self.get_parameter('obstacle_cost_weight').get_parameter_value().double_value
        self.control_cost_w = self.get_parameter('control_cost_weight').get_parameter_value().double_value
        self.num_samples_to_plot = self.get_parameter('num_samples_to_plot').get_parameter_value().integer_value

        self.collision_check_distance = self.get_parameter('collision_check_distance').get_parameter_value().double_value
        self.collision_check_width = self.get_parameter('collision_check_width').get_parameter_value().double_value
        self.collision_cost_threshold = self.get_parameter('collision_cost_threshold').get_parameter_value().double_value

        # --- 3. Grid 및 BEV 설정 ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )

        # 충돌 감지를 위한 그리드 셀 계산
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
            PointCloud2, '/bev_map', self.bev_map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 5. 상태 변수 ---
        self.current_pose = None    # [x, y, yaw] (글로벌 좌표계)
        self.costmap_tensor = None  # Costmap의 Torch 텐서 버전 (GPU 캐시용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        self.collision_detected_last_step = False

        # (신규) 주행 상태 변수
        # "NAVIGATING": MPPI 주행
        # "ALIGNING":   방향 정렬 (제자리 회전)
        # "PAUSING":    1초 대기
        # "DONE":       모든 주행 완료
        self.current_task = "NAVIGATING"
        self.pause_start_time = None

        # --- 6. (신규) 웨이포인트 (x, y, yaw) ---
        # 사용자로부터 받은 데이터를 파싱합니다.
        wp_data = [
            {'w': 0.6999738, 'x': 0.005913, 'y': 0.007110, 'z': 0.020897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},
            {'w': 0.6999738, 'x': 0.005913, 'y': 0.007110, 'z': -0.720897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},
            {'w': 0.9999738, 'x': 0.005913, 'y': 0.007110, 'z': 0.720897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},

            {'w': 1.392549, 'x': 0.001612, 'y': 0.005855, 'z': 0.019710, 'pos_x': 6.337625, 'pos_y': -0.486741},
            {'w': 0.716120, 'x': -0.017303, 'y': -0.007357, 'z': -0.697722, 'pos_x': 7.698485, 'pos_y': -31.281488},
            {'w': 0.999245, 'x': -0.005607, 'y': 0.003672, 'z': -0.038267, 'pos_x': 23.286015, 'pos_y': -33.319051},
            {'w': 0.729621, 'x': -0.014739, 'y': -0.009012, 'z': -0.683632, 'pos_x': 26.809536, 'pos_y': -67.154194},
            {'w': -0.361488, 'x': 0.010711, 'y': 0.006852, 'z': 0.932289, 'pos_x': 24.974758, 'pos_y': -75.537039},
            {'w': 0.819278, 'x': -0.043234, 'y': -0.054672, 'z': -0.569142, 'pos_x': 27.027404, 'pos_y': -99.656741},
            {'w': -0.097394, 'x': -0.007766, 'y': 0.010799, 'z': 0.995156, 'pos_x': 26.906550, 'pos_y': -99.646736},
            {'w': 0.815350, 'x': -0.005566, 'y': 0.006831, 'z': 0.578900, 'pos_x': 33.534878, 'pos_y': -54.895761},
            {'w': 0.774516, 'x': 0.000336, 'y': 0.004612, 'z': 0.632536, 'pos_x': 41.289568, 'pos_y': -28.024376}
        ]

        # (x, y, yaw) 튜플 리스트로 변환
        base_waypoints = []
        for wp in wp_data:
            yaw = self.quaternion_to_yaw_from_parts(wp['w'], wp['x'], wp['y'], wp['z'])
            base_waypoints.append((wp['pos_x'], wp['pos_y'], yaw))

        # (d1 ~ d10에 해당하는 튜플 리스트, 인덱스 0~9)
        d = base_waypoints

        # 기존 시퀀스 (d4,d3,d2,d1,...)에 맞춰 (x, y, yaw) 튜플로 재구성
        self.waypoints = [
            d[3], d[2], d[1], d[0], d[1], d[2], d[3],
            d[4], d[5], d[6], d[7], d[8], d[9]
        ]

        self.waypoints = [
            d[0], d[1], d[2], d[3],
            d[4], d[5], d[6], d[7], d[8], d[9]
        ]
        
        self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints with (x, y, yaw).")

        self.waypoint_index = 0

        # --- 7. Matplotlib 시각화 데이터 및 잠금 ---
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []                     # 로봇의 전체 궤적 (글로벌)
        self.obstacle_points_local = np.array([])     # BEV 장애물 (로컬)
        self.latest_local_goal = np.array([])         # 로컬 목표 지점 (로컬)
        self.latest_optimal_trajectory_local = np.array([]) # MPPI 최적 궤적 (로컬)
        self.latest_sampled_trajectories_local = np.array([]) # MPPI 샘플 궤적 다발 (로컬)

        # --- 8. ★ MPPI 컨트롤러 모듈 생성 ★ ---
        self.controller = MPPIController(
            logger=self.get_logger(),
            device=self.device,
            K=self.K, T=self.T, dt=self.dt, lambda_=self.lambda_,
            sigma_v=sigma_v, sigma_w=sigma_w,
            min_v=self.min_v, max_v=self.max_v, max_w=self.max_w,
            goal_cost_w=self.goal_cost_w,
            obstacle_cost_w=self.obstacle_cost_w,
            control_cost_w=self.control_cost_w,
            grid_resolution=self.grid_resolution,
            grid_origin_x=self.grid_origin_x,
            grid_origin_y=self.grid_origin_y,
            cells_x=self.cells_x,
            cells_y=self.cells_y,
            num_samples_to_plot=self.num_samples_to_plot
        )

        # --- 9. 제어 루프 타이머 ---
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        # --- 10. 1초 로깅 타이머 및 상태 변수 ---
        self.last_control_callback_time_ms = 0.0
        self.last_mppi_run_time_ms = 0.0
        self.last_bev_map_callback_time_ms = 0.0
        self.current_status = "Initializing" # 현재 노드 상태
        self.logging_timer = self.create_timer(1.0, self.logging_callback) # 1초 타이머
        # -------------------------------------------------

        self.get_logger().info("✅ MPPI BEV Planner (Modularized, with Alignment) has started.")

    # --- (신규) 1초 로깅 콜백 ---

    def logging_callback(self):
        """1초마다 현재 상태와 성능을 로깅합니다."""

        # 스레드 안전하게 성능 데이터 복사
        with self.plot_data_lock:
            status = self.current_status
            mppi_time = self.last_mppi_run_time_ms
            control_time = self.last_control_callback_time_ms
            bev_time = self.last_bev_map_callback_time_ms

            # 참고: control_time (e.g., 25ms)은 mppi_time (e.g., 20ms)보다 항상 큽니다.
            other_control_time = control_time - mppi_time

        # 제어 루프(dt) 대비 MPPI 연산이 얼마나 여유가 있는지
        loop_slack_ms = (self.dt * 1000.0) - mppi_time

        log_msg = (
            f"\n--- MPPI Status (1s Heartbeat) ---\n"
            f"  Status: {status}\n"
            f"  Loop Slack: {loop_slack_ms:6.1f} ms (Target: {self.dt * 1000.0:.0f} ms)\n"
            f"  Performance (Last call, ms):\n"
            f"    ├─ MPPI.run_mppi(): {mppi_time:8.2f} ms\n"
            f"    ├─ Other Control Logic: {other_control_time:4.2f} ms\n"
            f"    ├─ Total Control Callback: {control_time:5.2f} ms\n"
            f"    └─ BEV Map Callback: {bev_time:9.2f} ms"
        )
        self.get_logger().info(log_msg)


    # --- ROS 콜백 함수들 ---

    def quaternion_to_yaw_from_msg(self, q):
        """(이름 변경) Odometry 메시지(q)로부터 Yaw 각도를 계산합니다."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # (수정) 이름이 변경된 헬퍼 함수 사용
        yaw = self.quaternion_to_yaw_from_msg(msg.pose.pose.orientation)

        with self.plot_data_lock: # 시각화 스레드와 공유
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y])

    def bev_map_callback(self, msg: PointCloud2):
        start_time = time.perf_counter()
        try:
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            obstacle_points_local = []

            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                grid_c, grid_r = self.world_to_grid_idx_numpy(x, y)

                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    grid[grid_r, grid_c] = 255
                    obstacle_points_local.append([x, y])

            inflated_grid_np = cv2.dilate(grid, self.inflation_kernel)

            # ★ MPPI 컨트롤러와 시각화 모듈을 위한 데이터 업데이트
            self.costmap_tensor = torch.from_numpy(inflated_grid_np).to(self.device).float()
            with self.plot_data_lock:
                self.obstacle_points_local = np.array(obstacle_points_local)

        except Exception as e:
            self.get_logger().error(f"BEV map processing error: {e}\n{traceback.format_exc()}")
        finally:
            end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_bev_map_callback_time_ms = (end_time - start_time) * 1000.0


    def world_to_grid_idx_numpy(self, x, y):
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    def stop_robot(self):
        """로봇을 정지시키고 컨트롤러 상태를 리셋합니다."""

        # (신규) 노드가 종료 중일 때 publish를 시도하지 않도록 컨텍스트 확인
        if not rclpy.ok():
             self.get_logger().warn("stop_robot() called during shutdown, skipping publish.")
             return

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        # ★ 컨트롤러의 제어 시퀀스(U) 리셋
        self.controller.reset()

        # ★ 시각화 데이터 클리어
        with self.plot_data_lock:
            self.latest_local_goal = np.array([])
            self.latest_optimal_trajectory_local = np.array([])
            self.latest_sampled_trajectories_local = np.array([])

    # --- (신규) 충돌 감지 함수 ---

    def check_for_imminent_collision(self) -> bool:
        """
        미리 계산된 ROI를 사용해 costmap_tensor에서 즉각적인 충돌을 확인합니다.
        """
        if self.costmap_tensor is None:
            return False

        try:
            danger_zone = self.costmap_tensor[
                self.roi_r_start : self.roi_r_end,
                self.roi_c_start : self.roi_c_end
            ]

            if torch.any(danger_zone >= self.collision_cost_threshold):
                return True

        except Exception as e:
            self.get_logger().error(f"Collision check error: {e}\n{traceback.format_exc()}")
            return True # 에러 발생 시 안전을 위해 멈춤

        return False

    # --- (신규) 앵글 정규화 헬퍼 ---

    def normalize_angle(self, angle):
        """각도를 [-pi, pi] 범위로 정규화합니다."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # --- 메인 제어 루프 (로직 수정) ---

    def control_callback(self):
        """
        (수정됨)
        메인 제어 루프. 상태 머신(NAVIGATING, ALIGNING, PAUSING)을 기반으로 동작합니다.
        """
        control_start_time = time.perf_counter()

        # 0. 필수 데이터 확인
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            with self.plot_data_lock:
                self.current_status = "Waiting for Odometry"
            return

        # (신규) 모든 주행이 완료되었으면 정지 상태 유지
        if self.current_task == "DONE":
            self.stop_robot()
            return

        # (신규) 즉각적인 충돌 감지 (비상 정지)
        try:
            if self.check_for_imminent_collision():
                if not self.collision_detected_last_step:
                    self.get_logger().warn("🛑 IMMINENT COLLISION DETECTED! Stopping robot.")

                self.stop_robot()
                with self.plot_data_lock:
                    self.current_status = "COLLISION STOP"
                self.collision_detected_last_step = True
                return

            if self.collision_detected_last_step:
                self.get_logger().info("✅ Collision clear. Resuming navigation.")
                self.collision_detected_last_step = False

            # --- 1. (신규) 목표 및 현재 오차 계산 ---
            # (이 로직은 모든 상태에서 공통으로 사용됩니다)

            # 현재 웨이포인트(x, y, yaw) 가져오기
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y, target_yaw = target_wp[0], target_wp[1], target_wp[2]

            current_x, current_y, current_yaw = self.current_pose

            # 목표까지의 거리 및 방향 오차 계산
            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            yaw_error = self.normalize_angle(target_yaw - current_yaw)

            # --- 2. (신규) 주행 상태 머신 ---

            if self.current_task == "NAVIGATING":
                # "NAVIGATING": MPPI로 목표 위치까지 주행
                with self.plot_data_lock:
                    self.current_status = f"Running to WP {self.waypoint_index+1}/{len(self.waypoints)}"

                # 2-1. 위치에 도달했는지 확인
                if distance_to_goal < self.goal_threshold:
                    self.get_logger().info(f"WP {self.waypoint_index+1} position reached. Aligning orientation...")
                    self.current_task = "ALIGNING" # 다음 상태로 변경
                    self.stop_robot()
                    return # 이번 사이클은 종료

                # 2-2. (기존 MPPI 로직) 위치에 도달하지 못했으면 MPPI 계속 수행

                # 글로벌 목표 -> 로컬 목표 변환
                dx_global = target_x - current_x
                dy_global = target_y - current_y
                local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
                local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)

                local_goal_tensor = torch.tensor(
                    [local_target_x, local_target_y], device=self.device, dtype=torch.float32
                )

                # MPPI 컨트롤러 실행
                mppi_start_time = time.perf_counter()

                control_tuple, opt_traj_gpu, sampled_trajs_gpu = self.controller.run_mppi(
                    local_goal_tensor,
                    self.costmap_tensor # 최신 Costmap 텐서를 전달
                )

                mppi_end_time = time.perf_counter()
                mppi_run_time_ms = (mppi_end_time - mppi_start_time) * 1000.0

                with self.plot_data_lock:
                    self.last_mppi_run_time_ms = mppi_run_time_ms

                if control_tuple is None:
                    self.get_logger().warn("MPPI controller failed (Costmap not ready?). Stopping.")
                    with self.plot_data_lock:
                        self.current_status = "Controller Failed (Costmap?)"
                    self.stop_robot()
                    return

                # 시각화 데이터 업데이트
                with self.plot_data_lock:
                    self.latest_local_goal = local_goal_tensor.cpu().numpy()
                    self.latest_optimal_trajectory_local = opt_traj_gpu.cpu().numpy()
                    self.latest_sampled_trajectories_local = sampled_trajs_gpu.cpu().numpy()

                # 제어 명령 발행
                v, w = control_tuple
                twist_cmd = Twist()
                twist_cmd.linear.x = v
                twist_cmd.angular.z = w
                self.cmd_pub.publish(twist_cmd)

            elif self.current_task == "ALIGNING":
                # "ALIGNING": 목표 방향으로 제자리 회전
                with self.plot_data_lock:
                    self.current_status = f"Aligning at WP {self.waypoint_index+1}"

                # 3-1. 방향이 정렬되었는지 확인
                if abs(yaw_error) < self.yaw_threshold:
                    self.get_logger().info("Orientation aligned. Pausing for 1 second...")
                    self.current_task = "PAUSING" # 다음 상태로 변경
                    self.pause_start_time = self.get_clock().now() # 현재 시간 저장
                    self.stop_robot()
                    return # 이번 사이클은 종료

                # 3-2. (신규) 방향이 정렬되지 않았으면 P제어로 회전
                # (MPPI를 사용하지 않고 단순 회전 명령 발행)
                w = self.yaw_p_gain * yaw_error
                # 최대/최소 속도 클램핑
                w = np.clip(w, -self.max_w, self.max_w)

                # 임계값 밖에서는 최소 속도 보장 (stiction 방지)
                if abs(w) < self.min_align_w:
                    w = self.min_align_w * np.sign(w)

                twist_cmd = Twist()
                twist_cmd.linear.x = 0.0
                twist_cmd.angular.z = w
                self.cmd_pub.publish(twist_cmd)
                return # MPPI 로직 스킵

            elif self.current_task == "PAUSING":
                # "PAUSING": 1초간 대기
                with self.plot_data_lock:
                    self.current_status = f"Pausing at WP {self.waypoint_index+1}"

                elapsed_time_ns = (self.get_clock().now() - self.pause_start_time).nanoseconds
                elapsed_time_sec = elapsed_time_ns / 1e9

                # 4-1. 1초가 경과했는지 확인
                if elapsed_time_sec >= 1.0:
                    self.get_logger().info(f"✅ Waypoint {self.waypoint_index+1} complete! Moving to next.")
                    self.waypoint_index += 1 # 다음 웨이포인트 인덱스로
                    self.current_task = "NAVIGATING" # 다시 주행 상태로
                    self.pause_start_time = None

                    # 4-2. (신규) 모든 웨이포인트를 완료했는지 확인
                    if self.waypoint_index >= len(self.waypoints):
                        self.get_logger().info("🎉 All waypoints reached! Stopping.")
                        with self.plot_data_lock:
                            self.current_status = "All waypoints reached"
                        self.current_task = "DONE" # 최종 상태로 변경
                        self.stop_robot()
                        self.control_timer.cancel() # 타이머 중지
                        self.logging_timer.cancel()
                    return # 이번 사이클 종료

                else:
                    # 4-3. 아직 1초가 안 됐으면 정지 상태 유지
                    self.stop_robot()
                    return # MPPI 로직 스킵

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
        self.get_logger().info("Shutting down... Stopping robot.")
        if self.control_timer:
            self.control_timer.cancel()
        if self.logging_timer:
            self.logging_timer.cancel()
        self.stop_robot()
        super().destroy_node()

# --- main 함수 ---

def main(args=None):
    rclpy.init(args=args)
    node = MPPIBevPlanner()

    # ROS 2 노드(rclpy.spin)를 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 메인 스레드에서 Matplotlib 시각화 실행
    # setup_visualization 함수는 plt.show()로 인해 블로킹됨
    try:
        setup_visualization(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        # Matplotlib 창이 닫히면 ROS 노드 종료
        node.get_logger().info("Matplotlib closed, shutting down ROS node.")
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == '__main__':
    main()
