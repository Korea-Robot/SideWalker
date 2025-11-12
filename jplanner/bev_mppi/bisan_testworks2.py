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

# --- (신규) 뎁스 카메라 및 CV Bridge ---
from sensor_msgs.msg import Image
import cv_bridge
# ------------------------------------

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- 모듈화된 코드 임포트 ---
from optimized_controller import MPPIController
from visualizer import setup_visualization
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
    def __init__(self):
        super().__init__('mppi_bev_planner_viz_node')

        # --- 1. ROS 2 파라미터 선언 ---
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size_x', 50.0)
        self.declare_parameter('grid_size_y', 30.0)
        self.declare_parameter('inflation_radius', 0.2)
        self.declare_parameter('max_linear_velocity', 0.9)
        self.declare_parameter('min_linear_velocity', 0.15)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('mppi_k', 7000)
        self.declare_parameter('mppi_t', 40)
        self.declare_parameter('mppi_dt', 0.1)
        self.declare_parameter('mppi_lambda', 1.0)
        self.declare_parameter('mppi_sigma_v', 0.1)
        self.declare_parameter('mppi_sigma_w', 0.3)
        self.declare_parameter('goal_cost_weight', 95.0)
        self.declare_parameter('obstacle_cost_weight', 244.0)
        self.declare_parameter('control_cost_weight', 0.1)
        self.declare_parameter('num_samples_to_plot', 50)

        # (신규) 충돌 감지기 파라미터 (Depth Camera 기반)
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw') # 사용할 뎁스 카메라 토픽
        self.declare_parameter('depth_collision_threshold', 0.5) # [m] 50cm
        self.declare_parameter('depth_roi_width_percent', 0.4)   # 이미지 중앙 40% 폭
        self.declare_parameter('depth_roi_height_percent', 0.4)  # 이미지 중앙 40% 높이
        self.declare_parameter('depth_min_pixels_for_collision', 50) # 임계값 이하 픽셀이 50개 이상이면 정지

        # --- 2. 파라미터 값 가져오기 ---
        # (가독성을 위해 .get_parameter()...를 변수로 저장)
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.max_v = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.min_v = self.get_parameter('min_linear_velocity').get_parameter_value().double_value
        self.max_w = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.goal_threshold = self.get_parameter('goal_threshold').get_parameter_value().double_value
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

        # (신규) 충돌 감지기 파라미터 가져오기
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.depth_collision_threshold = self.get_parameter('depth_collision_threshold').get_parameter_value().double_value
        self.depth_roi_width_p = self.get_parameter('depth_roi_width_percent').get_parameter_value().double_value
        self.depth_roi_height_p = self.get_parameter('depth_roi_height_percent').get_parameter_value().double_value
        self.min_pixels_for_collision = self.get_parameter('depth_min_pixels_for_collision').get_parameter_value().integer_value

        # --- 3. Grid 및 BEV 설정 ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        # (제거됨) 기존 BEV 기반 충돌 감지 로직
        
        # --- 4. ROS2 Setup ---
        self.bridge = cv_bridge.CvBridge() # (신규)
        self.depth_sub = self.create_subscription( # (신규)
            Image, self.depth_topic, self.depth_callback, 10
        )
        self.bev_sub = self.create_subscription(
            PointCloud2, '/bev_map', self.bev_map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 5. 상태 변수 ---
        self.current_pose = None   # [x, y, yaw] (글로벌 좌표계)
        self.costmap_tensor = None # Costmap의 Torch 텐서 버전 (GPU 캐시용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.imminent_collision_from_depth = False # (신규) 뎁스 콜백이 갱신
        
        self.get_logger().info(f"Using device: {self.device}")
        
        # (신규) 충돌 상태
        self.collision_detected_last_step = False
        
        # --- 6. 웨이포인트 ---
        # 6F 
        d1 = (-5.6,0.48)
        d2 = (-4.66,7.05)
        d3 = (2.844,6.9)
        d4 = (2.85,-0.68)
        d5 = (-5.0,0.132)
        # self.waypoints = [d1, d2, d3, d4,d5, d1,d2,d3, d4,d5, d1,d2,d3, d4,d5, d1,d2]


        # 1F loop
        d1 = (-0.3,1.88)
        d2 = (5.58,19.915)
        d3 = (2.606,36.25)
        d4 = (-9.88,38.336)
        d5 = (-21.88,29.57)

        # 1029 6F
        d1 = (0.09,-0.08)
        d2 = (6.60,0.84)
        d3 = (7.92,-7.85)
        d4 = (0.74,-8.18)
        d5 = d1


        self.waypoints = [d1, d2, d3, d4, d5,d1]

        # 1F large map 

        # d1 = ( 1.18, -0.14)
        d2 = ( 17.73, 1.23)
        d3 = (22.17,11.71)
        d4 = ( 21.39, 19.28)
        d5 = ( 22.16, 29.43)
        d6 = ( 42.10, 28.57)
        d7 = ( 39.79, 17.11)
        d8 = ( 21.21, 17.41)
        self.waypoints = [d3,d4, d5,d6,d7,d8,d3,d2]

        d1  = ( 0.25,  -0.15 )  # start point 
        d2  = ( 6.34,  -0.49 )  # water 
        d3  = ( 7.70,  -31.28) # point  
        d4  = ( 23.29, -33.32) # point 
        d5  = ( 26.81, -67.15) # point 
        d6  = ( 24.97, -75.54) # bolad
        d7  = ( 27.03, -99.66) # sidewalk
        d8  = ( 26.91, -99.65) # trafficlight
        d9  = ( 33.53, -54.90) # to kenopi 
        d10  = ( 41.29, -28.02) # kenopi

        
        self.waypoints = [d4,d3,d2,d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]
        # self.waypoints = [d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]


        d1 = (0,0)          # start 
        d2 = (4.46,0.26)    # point
        d3 = (9.75,-30.78) # point
        d4 = (24.16,-30.74) # point
        d5 = (29.65,-97.64) # traffic light 
        d6 = (32.42,-96.53) 
        d7 = (61.57,-101.34) # forest enterance
        d8 = (60.59,-67.95) # middle of forest
        d9 = (53.99,-22.33) # end of forest
        d10 = (32.87,-28.13)

        self.waypoints = [d3,d2,d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]
        # self.waypoints = [d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]
        self.waypoints = [d5,d6,d7,d8,d9,d10]

        d1 = (1.0,1.0)          # start 
        d2 = (4.46,0.26)    # point
        d3 = (12.75,-30.78) # point
        d4 = (24.16,-30.74) # point
        d5 = (29.65,-97.64) # traffic light 
        d6 = (32.42,-96.53) 
        d7 = (61.57,-101.34) # forest enterance
        d8 = (60.59,-67.95) # middle of forest
        d9 = (53.99,-22.33) # end of forest
        d10 = (32.87,-28.13)

        d11 = (33.65,-77.64)
        d12 = (45.77,-22.33)

        self.waypoints = [d3,d2,d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]*1
        # self.waypoints = [d3,d2,d1,d2,d3,d4, ]
        # self.waypoints = [d1,d2,d3,d4, d5,d6,d7,d8,d9,d10]
        # self.waypoints = [d5,d6,d7,d8,d9,d10]
        self.waypoints = [d4,d11,d12,d10]*5

        # --- (신규) Waypoint 통계 변수 ---
        self.total_waypoints = len(self.waypoints)
        self.summary_logged = False # 요약 로그 중복 출력을 방지하기 위함
        self.get_logger().info(f"✅ Mission loaded with {self.total_waypoints} waypoints.")
        # -----------------------------------
        
        self.waypoint_index = 0
        
        # --- 7. Matplotlib 시각화 데이터 및 잠금 ---
        # (시각화 스레드와 ROS 스레드 간의 데이터 교환용)
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []                # 로봇의 전체 궤적 (글로벌)
        self.obstacle_points_local = np.array([])   # BEV 장애물 (로컬)
        self.latest_local_goal = np.array([])       # 로컬 목표 지점 (로컬)
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

        # --- 10. (신규) 1초 로깅 타이머 및 상태 변수 ---
        self.last_control_callback_time_ms = 0.0
        self.last_mppi_run_time_ms = 0.0
        self.last_bev_map_callback_time_ms = 0.0
        self.current_status = "Initializing" # 현재 노드 상태
        self.logging_timer = self.create_timer(1.0, self.logging_callback) # 1초 타이머
        # -------------------------------------------------

        self.get_logger().info("✅ MPPI BEV Planner (Modularized) has started.")

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
        # mppi_time이 20ms이고 dt가 100ms이면, 80ms의 여유(slack)가 있음
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

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        with self.plot_data_lock: # 시각화 스레드와 공유
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y])

    def bev_map_callback(self, msg: PointCloud2):
        start_time = time.perf_counter() # (신규) 시간 측정 시작
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
            # (신규) 시간 측정 종료 및 저장
            end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_bev_map_callback_time_ms = (end_time - start_time) * 1000.0


    # --- (신규) 뎁스 카메라 콜백 ---
    
    def depth_callback(self, msg: Image):
        """
        뎁스 카메라 이미지를 처리하여 즉각적인 충돌 위험을 감지합니다.
        """
        try:
            # 1. ROS 이미지를 OpenCV(Numpy)로 변환
            # 뎁스 인코딩이 '16UC1' (mm 단위) 또는 '32FC1' (m 단위)일 수 있음
            if msg.encoding == '16UC1':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # mm를 m 단위의 float로 변환
                cv_image = cv_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}", throttle_duration_sec=5.0)
                return

            # 2. 이미지 중앙의 ROI(관심 영역) 정의
            h, w = cv_image.shape
            roi_w = int(w * self.depth_roi_width_p)
            roi_h = int(h * self.depth_roi_height_p)
            
            roi_x_start = (w - roi_w) // 2
            roi_y_start = (h - roi_h) // 2
            
            depth_roi = cv_image[roi_y_start : roi_y_start + roi_h, 
                                 roi_x_start : roi_x_start + roi_w]

            # 3. ROI 내 유효한(nan/inf가 아닌) 픽셀만 필터링
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0.01)] # 1cm 이상

            if valid_depths.size == 0:
                # ROI 내에 유효한 픽셀이 없음 (아마도 너무 멀거나 가까움)
                with self.plot_data_lock:
                    self.imminent_collision_from_depth = False
                return

            # 4. ★ 안전 로직 (평균 대신 '임계값 픽셀 카운트') ★
            # 'depth_collision_threshold'보다 가까운 픽셀의 개수를 셉니다.
            pixels_in_danger = valid_depths[valid_depths < self.depth_collision_threshold]
            
            collision = False
            if pixels_in_danger.size > self.min_pixels_for_collision:
                # 위험한 픽셀이 설정한 'min_pixels_for_collision' 개수보다 많으면 충돌로 간주
                collision = True
            
            # 5. 스레드 안전하게 충돌 상태 업데이트
            with self.plot_data_lock:
                self.imminent_collision_from_depth = collision

        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}\n{traceback.format_exc()}")
            with self.plot_data_lock:
                self.imminent_collision_from_depth = True # 에러 발생 시 안전을 위해 정지


    def world_to_grid_idx_numpy(self, x, y):
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    def stop_robot(self):
        """로봇을 정지시키고 컨트롤러 상태를 리셋합니다."""
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

    # --- (신규) 충돌 감지 함수 (Depth 기반) ---
    
    def check_for_imminent_collision(self) -> bool:
        """
        뎁스 콜백이 설정한 플래그(self.imminent_collision_from_depth)를 확인합니다.
        이 함수는 control_callback 스레드에서 호출됩니다.
        """
        collision_detected = False
        try:
            # depth_callback과 공유하는 변수를 스레드 안전하게 읽어옴
            with self.plot_data_lock:
                collision_detected = self.imminent_collision_from_depth
                
        except Exception as e:
            self.get_logger().error(f"Collision check error: {e}\n{traceback.format_exc()}")
            return True # 에러 발생 시 안전을 위해 멈춤
            
        return collision_detected
    
    # --- (신규) ★★★ 미션 요약 로깅 함수 ★★★ ---
    def log_mission_summary(self):
        """
        미션 종료 시(성공, 중단, 실패) 최종 통계를 로깅합니다.
        """
        # 중복 로깅 방지
        if self.summary_logged:
            return
        self.summary_logged = True
        
        completed_count = self.waypoint_index
        total_count = self.total_waypoints

        if total_count == 0:
            self.get_logger().info("Mission summary: No waypoints were loaded.")
            return

        # (e.g., 3 / 10) * 100.0 = 30.0
        percentage = (completed_count / total_count) * 100.0

        # --- (신규) 완료된 웨이포인트 목록 생성 ---
        completed_list_str = ""
        if completed_count > 0:
            # 1-based 인덱스(i+1)와 좌표를 함께 기록
            # 사용자가 요청한 대로 for-loop을 사용하여 목록 생성
            waypoint_lines = []
            for i in range(completed_count):
                # (수정) "waypoint success"와 1-based index 사용
                line = f"    > Waypoint {i + 1}번 성공 (좌표: {self.waypoints[i]})"
                waypoint_lines.append(line)
            
            completed_list_str = "\n".join(waypoint_lines)
        # ------------------------------------

        summary_msg = (
            f"\n--- 🏁 Mission Summary 🏁 ---\n"
            f"  Waypoints Completed: {completed_count} / {total_count}\n"
            f"  Completion Rate:     {percentage:.1f}%\n"
        )
        
        # (신규) 완료된 목록이 있으면 요약에 추가
        if completed_list_str:
            summary_msg += "  --- Completed List ---\n"
            summary_msg += f"{completed_list_str}\n"
            summary_msg += "  ----------------------\n"

        # 상태 확인
        if completed_count == total_count:
            # 100% 성공
            summary_msg += "  Status:              SUCCESS (All waypoints reached)"
            self.get_logger().info(summary_msg)
        else:
            # 100% 미만 (중단 또는 실패)
            with self.plot_data_lock:
                status = self.current_status
            
            # 상태가 '충돌'이나 '에러'가 아닌데 종료된 경우 (e.g. Ctrl+C)
            if "COLLISION" not in status and "ERROR" not in status:
                 status = "Interrupted (e.g., Ctrl+C or Viz close)"
            
            summary_msg += f"  Status:              STOPPED ({status})"
            # 실패/중단은 WARN 레벨로 로깅
            self.get_logger().warn(summary_msg)

    # --- 메인 제어 루프 ---

    def control_callback(self):
        """
        메인 제어 루프. 
        데이터를 준비하고, 컨트롤러를 호출하며, 결과를 발행하고, 시각화 데이터를 업데이트합니다.
        """
        control_start_time = time.perf_counter() # (신규) 전체 콜백 시간 측정 시작
        
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            with self.plot_data_lock:
                self.current_status = "Waiting for Odometry" # (신규) 상태 업데이트
            return

        try:
            # --- (신규) 0. 즉각적인 충돌 감지 ---
            # MPPI 계산 전에 뎁스 카메라 플래그를 기반으로 비상 정지 확인
            if self.check_for_imminent_collision():
                if not self.collision_detected_last_step:
                    self.get_logger().warn("🛑 IMMINENT COLLISION DETECTED! (Depth) Stopping robot.")
                
                self.stop_robot()
                with self.plot_data_lock:
                    self.current_status = "COLLISION STOP"
                self.collision_detected_last_step = True
                return # MPPI 계산 및 주행 중지
            
            # 충돌이 감지되었다가 해제된 경우
            if self.collision_detected_last_step:
                self.get_logger().info("✅ Collision clear. Resuming navigation.")
                self.collision_detected_last_step = False
            # ---------------------------------

            # 1. 웨이포인트 도달 확인
            if self.waypoint_index >= len(self.waypoints):
                # (신규) 상태를 'SUCCESS'로 설정
                with self.plot_data_lock:
                    self.current_status = "SUCCESS (All waypoints reached)"
                
                # (신규) 성공 로그 요약 함수 호출
                self.log_mission_summary() 
                
                self.stop_robot()
                self.control_timer.cancel()
                self.logging_timer.cancel() # (신규) 로깅 타이머도 중지
                return

            # (신규) 현재 상태 업데이트
            with self.plot_data_lock:
                # (수정) total_waypoints 사용
                self.current_status = f"Running to WP {self.waypoint_index+1}/{self.total_waypoints}"

            # 2. 현재 상태 및 목표 설정
            current_x, current_y, current_yaw = self.current_pose
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp[0], target_wp[1]

            # 3. 목표 도달 시 다음 웨이포인트로
            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                # (신규) ★★★ 웨이포인트 성공 로그 (요청사항 반영) ★★★
                # 0-based index(self.waypoint_index) 대신 1-based index(self.waypoint_index + 1) 사용
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index + 1}번 성공! (좌표: {target_wp})")
                self.waypoint_index += 1
                self.stop_robot() 
                return

            # 4. 글로벌 목표 -> 로컬 목표 변환
            dx_global = target_x - current_x
            dy_global = target_y - current_y
            local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            local_goal_tensor = torch.tensor(
                [local_target_x, local_target_y], device=self.device, dtype=torch.float32
            )
            
            # 5. ★ MPPI 컨트롤러 실행 ★
            # (신규) MPPI 연산 시간만 별도 측정
            mppi_start_time = time.perf_counter()
            
            control_tuple, opt_traj_gpu, sampled_trajs_gpu = self.controller.run_mppi(
                local_goal_tensor, 
                self.costmap_tensor # 최신 Costmap 텐서를 전달
            )
            
            mppi_end_time = time.perf_counter()
            mppi_run_time_ms = (mppi_end_time - mppi_start_time) * 1000.0
            
            # (신규) MPPI 연산 시간 저장
            with self.plot_data_lock:
                self.last_mppi_run_time_ms = mppi_run_time_ms
            
            # 6. 컨트롤러 실행 결과 처리
            if control_tuple is None: # e.g., Costmap이 준비되지 않음
                self.get_logger().warn("MPPI controller failed (Costmap not ready?). Stopping.")
                with self.plot_data_lock:
                    self.current_status = "Controller Failed (Costmap?)" 
                self.stop_robot()
                return
            
            # 7. ★ 시각화 데이터 업데이트 ★
            # (GPU 텐서를 CPU Numpy 배열로 변환하여 저장)
            with self.plot_data_lock:
                self.latest_local_goal = local_goal_tensor.cpu().numpy()
                self.latest_optimal_trajectory_local = opt_traj_gpu.cpu().numpy()
                self.latest_sampled_trajectories_local = sampled_trajs_gpu.cpu().numpy()
            
            # 8. 제어 명령 발행
            v, w = control_tuple
            twist_cmd = Twist()
            twist_cmd.linear.x = v
            twist_cmd.angular.z = w
            self.cmd_pub.publish(twist_cmd)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            with self.plot_data_lock:
                self.current_status = "ERROR in control loop" # (신규) 상태 업데이트
            self.stop_robot()
        finally:
            # (신규) 전체 콜백 시간 측정 및 저장
            control_end_time = time.perf_counter()
            with self.plot_data_lock:
                self.last_control_callback_time_ms = (control_end_time - control_start_time) * 1000.0

            
    def destroy_node(self):
        self.get_logger().info("Shutting down... Stopping robot.")
        if self.control_timer:
            self.control_timer.cancel()
        if self.logging_timer: # (신규) 로깅 타이머 취소
            self.logging_timer.cancel()
        self.stop_robot()
        
        # (신규) ★★★ 노드 종료 시 미션 요약 로그 호출 ★★★
        # (중단, 충돌 정지, 에러 등으로 100% 완료되지 못했을 경우)
        self.log_mission_summary()
        
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
