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
from bold_visualizer import setup_visualization
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
        self.declare_parameter('grid_size_x', 30.0)
        self.declare_parameter('grid_size_y', 20.0)
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('min_linear_velocity', 0.2)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_threshold', 0.6)
        self.declare_parameter('mppi_k', 2000)
        self.declare_parameter('mppi_t', 50)
        self.declare_parameter('mppi_dt', 0.1)
        self.declare_parameter('mppi_lambda', 1.0)
        self.declare_parameter('mppi_sigma_v', 0.1)
        self.declare_parameter('mppi_sigma_w', 0.2)
        self.declare_parameter('goal_cost_weight', 25.0)
        self.declare_parameter('obstacle_cost_weight', 40.0)
        self.declare_parameter('control_cost_weight', 0.1)
        self.declare_parameter('num_samples_to_plot', 50)

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

        # --- 3. Grid 및 BEV 설정 ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
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
        
        # --- 6. 웨이포인트 ---
        # --- 웨이포인트 ---
        # 6F 
        d1 = (-5.6,0.48)
        d2 = (-4.66,7.05)
        d3 = (2.844,6.9)
        d4 = (2.85,-0.68)
        d5 = (-5.0,0.132)


        d1 = (5.035,-5.204)
        d2 = (-3.25,-4.72) 
        d3 = (-4.32,-11.68)
        d4 = (4.52,-12.17)


        # 1029 6F
        d1 = (0.09,-0.08)
        d2 = (6.60,0.84)
        d3 = (7.92,-7.85)
        d4 = (0.74,-8.18)

        d5 = d1 

        self.waypoints = [d1, d2, d3, d4,d5, d1,d2,d3, d4,d5, d1,d2,d3, d4,d5, d1,d2]


        # 1F loop
        # d1 = (-0.3,1.88)
        # d2 = (5.58,19.915)
        # d3 = (2.606,36.25)
        # d4 = (-9.88,38.336)
        # d5 = (-21.88,29.57)
        
        # self.waypoints = [d1, d2, d3, d4, d5,d1]
        
        self.waypoint_index = 0
        
        # --- 7. Matplotlib 시각화 데이터 및 잠금 ---
        # (시각화 스레드와 ROS 스레드 간의 데이터 교환용)
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

        self.get_logger().info("✅ MPPI BEV Planner (Modularized) has started.")

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

    # --- 메인 제어 루프 ---

    def control_callback(self):
        """
        메인 제어 루프. 
        데이터를 준비하고, 컨트롤러를 호출하며, 결과를 발행하고, 시각화 데이터를 업데이트합니다.
        """
        
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            return

        try:
            # 1. 웨이포인트 도달 확인
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("🎉 All waypoints reached! Stopping.")
                self.stop_robot()
                self.control_timer.cancel()
                return

            # 2. 현재 상태 및 목표 설정
            current_x, current_y, current_yaw = self.current_pose
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp[0], target_wp[1]

            # 3. 목표 도달 시 다음 웨이포인트로
            distance_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
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
            # 컨트롤러는 (v, w), optimal_traj, sampled_trajs를 반환
            control_tuple, opt_traj_gpu, sampled_trajs_gpu = self.controller.run_mppi(
                local_goal_tensor, 
                self.costmap_tensor # 최신 Costmap 텐서를 전달
            )
            
            # 6. 컨트롤러 실행 결과 처리
            if control_tuple is None: # e.g., Costmap이 준비되지 않음
                self.get_logger().warn("MPPI controller failed. Stopping.")
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
            self.stop_robot()
            
    def destroy_node(self):
        self.get_logger().info("Shutting down... Stopping robot.")
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

