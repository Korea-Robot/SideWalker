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

# (신규) YAML 및 파일 경로 관련 임포트
import yaml
import os
# from ament_index_python.packages import get_package_share_directory # <-- 삭제

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- 모듈화된 코드 임포트 ---
from optimized_controller import MPPIController
from bold_visualizer import setup_visualization
from planner import SubgoalPlanner
# -----------------------------


"""
역할: 메인 ROS 2 노드입니다.

'config_file_path' ROS 파라미터에서 모든 설정을 로드합니다.
ROS 통신, 상태 관리, 그리고 모듈 오케스트레이션을 담당합니다.
"""
class MPPIBevPlanner(Node):
    """
    MPPI 플래너를 실행하는 메인 ROS 2 노드.
    ROS 통신, 상태 관리, 그리고 컨트롤러/시각화 모듈의 조율을 담당.
    """
    def __init__(self):
        super().__init__('mppi_bev_planner_viz_node')

        # --- 1. (수정) YAML 파일 경로를 ROS 파라미터로 읽어오기 ---
        # (수정) 기본 경로를 'config/config_params.yaml'로 변경
        self.declare_parameter('config_file_path', 'config/config_params.yaml')
        config_file_path = self.get_parameter('config_file_path').get_parameter_value().string_value

        try:
            if not os.path.exists(config_file_path):
                # (수정) rclpy.shutdown() 및 return 대신, main()이 잡을 수 있도록 예외 발생
                msg = f"Could not find config file: {config_file_path}. Looking in CWD. Use 'config_file_path' param for absolute path."
                self.get_logger().fatal(msg)
                self.get_logger().fatal(f"Please provide the correct path via the 'config_file_path' parameter.")
                raise FileNotFoundError(msg)

            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.get_logger().info(f"Loaded config from {config_file_path}")

        except Exception as e:
            # (수정) rclpy.shutdown() 및 return 대신, main()이 잡을 수 있도록 예외 발생
            self.get_logger().fatal(f"Failed to load config file '{config_file_path}'. Error: {e}")
            raise e # main()의 try/except 블록으로 예외를 전달
            
        # --- 2. (신규) config 딕셔너리에서 파라미터 추출 ---

        # ROS Topics
        ros_config = config.get('ros_topics', {})
        self.bev_map_topic = ros_config.get('bev_map_topic', '/bev_map')
        self.cmd_vel_topic = ros_config.get('cmd_vel_topic', '/mcu/command/manual_twist')
        self.odom_topic = ros_config.get('odom_topic', '/krm_auto_localization/odom')

        # Robot Limits
        robot_config = config.get('robot_limits', {})
        self.max_v = robot_config.get('max_linear_velocity', 1.0)
        self.min_v = robot_config.get('min_linear_velocity', 0.2)
        self.max_w = robot_config.get('max_angular_velocity', 1.0)

        # Grid Map
        grid_config = config.get('grid_map', {})
        self.grid_resolution = grid_config.get('resolution', 0.1)
        self.size_x = grid_config.get('size_x', 30.0)
        self.size_y = grid_config.get('size_y', 20.0)
        self.inflation_radius = grid_config.get('inflation_radius', 0.1)
        
        # Planner
        planner_config = config.get('planner', {})
        self.planner_lookahead_distance = planner_config.get('lookahead_distance', 4.0)
        self.goal_threshold = planner_config.get('goal_threshold', 0.6)
        self.subgoal_num_samples = planner_config.get('num_subgoal_samples', 50)
        self.subgoal_goal_cost_w = planner_config.get('goal_cost_w', 1.0)
        self.subgoal_obs_cost_w = planner_config.get('obs_cost_w', 5.0)

        # MPPI Controller
        mppi_config = config.get('controller_mppi', {})
        self.K = mppi_config.get('K', 1000)
        self.T = mppi_config.get('T', 100)
        self.dt = mppi_config.get('dt', 0.1) # ★ 제어 주기
        self.lambda_ = mppi_config.get('lambda', 1.0)
        sigma_v = mppi_config.get('sigma_v', 0.1)
        sigma_w = mppi_config.get('sigma_w', 0.2)
        self.goal_cost_w = mppi_config.get('goal_cost_w', 25.0)
        self.obstacle_cost_w = mppi_config.get('obstacle_cost_w', 100.0)
        self.control_cost_w = mppi_config.get('control_cost_w', 0.1)
        
        # Safety
        safety_config = config.get('safety_collision_checker', {})
        self.collision_check_distance = safety_config.get('check_distance', 0.5)
        self.collision_check_width = safety_config.get('check_width', 0.4)
        self.collision_cost_threshold = safety_config.get('cost_threshold', 250.0)

        # Visualization
        vis_config = config.get('visualization', {})
        self.num_samples_to_plot = vis_config.get('num_samples_to_plot', 50)

        # Waypoints (List of Lists -> List of Tuples)
        self.waypoints = [tuple(wp) for wp in config.get('waypoints', [])]
        if not self.waypoints:
            self.get_logger().warn("No waypoints loaded from config file!")
        self.waypoint_index = 0

        # --- 3. Grid 및 BEV 설정 (기존과 동일, 로드된 변수 사용) ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        self.robot_grid_c = int((0.0 - self.grid_origin_x) / self.grid_resolution)
        self.robot_grid_r = int((0.0 - self.grid_origin_y) / self.grid_resolution)
        check_dist_cells = int(self.collision_check_distance / self.grid_resolution)
        check_width_cells = int(self.collision_check_width / self.grid_resolution)
        self.roi_r_start = max(0, self.robot_grid_r - check_width_cells // 2)
        self.roi_r_end = min(self.cells_y, self.robot_grid_r + check_width_cells // 2)
        self.roi_c_start = max(0, self.robot_grid_c) 
        self.roi_c_end = min(self.cells_x, self.robot_grid_c + check_dist_cells) 

        self.get_logger().info(
            f"Collision checker ROI (grid indices):\n"
            f"  Rows (width): {self.roi_r_start} to {self.roi_r_end}\n"
            f"  Cols (dist):  {self.roi_c_start} to {self.roi_c_end}"
        )

        
        # --- 4. ROS2 Setup (로드된 토픽 이름 사용) ---
        self.bev_sub = self.create_subscription(
            PointCloud2, self.bev_map_topic, self.bev_map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)

        # --- 5. 상태 변수 ---
        self.current_pose = None
        self.costmap_tensor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        self.collision_detected_last_step = False
        
        # --- 6. 웨이포인트 (YAML에서 로드 완료) ---
        
        # --- 7. Matplotlib 시각화 데이터 및 잠금 ---
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.obstacle_points_local = np.array([])
        self.latest_local_goal = np.array([])
        self.latest_optimal_trajectory_local = np.array([])
        self.latest_sampled_trajectories_local = np.array([])

        # --- 8. ★ 모듈 생성 (로드된 변수 사용) ★ ---

        # 8-1. Cost-Aware 서브골 플래너 생성
        self.planner = SubgoalPlanner(
            logger=self.get_logger(),
            device=self.device,
            lookahead_distance=self.planner_lookahead_distance,
            goal_threshold=self.goal_threshold,
            num_subgoal_samples=self.subgoal_num_samples,
            subgoal_goal_cost_w=self.subgoal_goal_cost_w,
            subgoal_obs_cost_w=self.subgoal_obs_cost_w,
            grid_resolution=self.grid_resolution,
            grid_origin_x=self.grid_origin_x,
            grid_origin_y=self.grid_origin_y,
            cells_x=self.cells_x,
            cells_y=self.cells_y
        )
        
        # 8-2. MPPI 컨트롤러 모듈 생성
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

        # --- 9. 제어 루프 타이머 (dt 변수 사용) ---
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        # --- 10. 1초 로깅 타이머 및 상태 변수 ---
        self.last_control_callback_time_ms = 0.0
        self.last_mppi_run_time_ms = 0.0
        self.last_bev_map_callback_time_ms = 0.0
        self.current_status = "Initializing"
        self.logging_timer = self.create_timer(1.0, self.logging_callback)

        self.get_logger().info(f"✅ MPPI BEV Planner (Config from YAML) has started. Following {len(self.waypoints)} waypoints.")

    # --- 1초 로깅 콜백 ---
    
    def logging_callback(self):
        """1초마다 현재 상태와 성능을 로깅합니다."""
        
        with self.plot_data_lock:
            status = self.current_status
            mppi_time = self.last_mppi_run_time_ms
            control_time = self.last_control_callback_time_ms
            bev_time = self.last_bev_map_callback_time_ms
            other_control_time = control_time - mppi_time
        
        loop_slack_ms = (self.dt * 1000.0) - mppi_time 

        log_msg = (
            f"\n--- MPPI Status (1s Heartbeat) ---\n"
            f"  Status: {status}\n"
            f"  Planner: Aiming for *Cost-Aware* Subgoal ({self.planner.last_subgoal_x:.1f}, {self.planner.last_subgoal_y:.1f}) | IsFinal: {self.planner.is_final_subgoal}\n"
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
        
        with self.plot_data_lock:
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
        """미리 계산된 ROI를 사용해 costmap_tensor에서 즉각적인 충돌을 확인합니다."""
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
            return True
        return False

    # --- 메인 제어 루프 ---

    def control_callback(self):
        """
        메인 제어 루프.
        Planner / Controller를 호출하고 결과를 발행합니다.
        """
        control_start_time = time.perf_counter()
        
        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...")
            with self.plot_data_lock:
                self.current_status = "Waiting for Odometry"
            return

        try:
            # 0. 즉각적인 충돌 감지
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

            # 1. 웨이포인트 완료 확인
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("🎉 All waypoints reached! Stopping.")
                with self.plot_data_lock:
                    self.current_status = "All waypoints reached"
                self.stop_robot()
                self.control_timer.cancel()
                self.logging_timer.cancel()
                return

            with self.plot_data_lock:
                self.current_status = f"Running to WP {self.waypoint_index+1}/{len(self.waypoints)}"

            # 2. 현재 상태 및 *최종* 목표 설정
            current_x, current_y, current_yaw = self.current_pose
            target_wp = self.waypoints[self.waypoint_index]
            target_x, target_y = target_wp[0], target_wp[1]

            # 3. ★ Cost-Aware SubgoalPlanner 호출 ★
            (subgoal_x, subgoal_y), is_final_subgoal = self.planner.get_subgoal(
                current_x, current_y, target_x, target_y,
                self.costmap_tensor # (신규) Costmap을 플래너에 전달
            )

            # 4. 최종 목표 도달 시 다음 웨이포인트로
            if is_final_subgoal:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                self.stop_robot() 
                return

            # 5. 글로벌 *서브골* -> 로컬 *서브골* 변환
            dx_global = subgoal_x - current_x
            dy_global = subgoal_y - current_y
            local_target_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_target_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            local_goal_tensor = torch.tensor(
                [local_target_x, local_target_y], device=self.device, dtype=torch.float32
            )
            
            # 6. ★ MPPI 컨트롤러 실행 ★
            mppi_start_time = time.perf_counter()
            control_tuple, opt_traj_gpu, sampled_trajs_gpu = self.controller.run_mppi(
                local_goal_tensor,
                self.costmap_tensor
            )
            mppi_end_time = time.perf_counter()
            mppi_run_time_ms = (mppi_end_time - mppi_start_time) * 1000.0
            
            with self.plot_data_lock:
                self.last_mppi_run_time_ms = mppi_run_time_ms
            
            # 7. 컨트롤러 실행 결과 처리
            if control_tuple is None:
                self.get_logger().warn("MPPI controller failed (Costmap not ready?). Stopping.")
                with self.plot_data_lock:
                    self.current_status = "Controller Failed (Costmap?)" 
                self.stop_robot()
                return
            
            # 8. ★ 시각화 데이터 업데이트 ★
            with self.plot_data_lock:
                self.latest_local_goal = local_goal_tensor.cpu().numpy()
                self.latest_optimal_trajectory_local = opt_traj_gpu.cpu().numpy()
                self.latest_sampled_trajectories_local = sampled_trajs_gpu.cpu().numpy()
            
            # 9. 제어 명령 발행
            v, w = control_tuple
            twist_cmd = Twist()
            twist_cmd.linear.x = v
            twist_cmd.angular.z = w
            self.cmd_pub.publish(twist_cmd)

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
    # (수정) 노드 생성 실패 시를 대비한 예외 처리
    node = None
    try:
        node = MPPIBevPlanner()
    except Exception as e:
        # __init__에서 치명적 오류 발생 시 (e.g., config 파일 로드 실패)
        # node가 None일 수 있음.
        if node is None:
            print(f"Failed to initialize node: {e}")
        rclpy.shutdown()
        return

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


