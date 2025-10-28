#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import threading
import time
import math
import traceback
import sensor_msgs_py.point_cloud2 as pc2

# --- 모듈화된 클래스 임포트 ---
from planner import AStarPlanner
from controller import PurePursuitController
from visualizer import MatplotlibVisualizer

# ==============================================================================
# --- ROS2 Node ---
# ==============================================================================

class PlannerNode(Node):
    def __init__(self):
        super().__init__('astar_bev_planner_node')

        # --- ROS 2 파라미터 선언 ---
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size_x', 80.0)
        self.declare_parameter('grid_size_y', 80.0)
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('max_linear_velocity', 0.6)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('look_ahead_dist', 0.7)
        self.declare_parameter('turn_damping_factor', 1.5)

        # --- 파라미터 값 가져오기 ---
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        
        # 제어 파라미터
        max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        look_ahead_dist = self.get_parameter('look_ahead_dist').get_parameter_value().double_value
        turn_damping_factor = self.get_parameter('turn_damping_factor').get_parameter_value().double_value
        angular_gain = 2.0 # P-gain

        # --- Grid 설정 ---
        self.cells_x = int(self.size_x / self.grid_resolution)
        self.cells_y = int(self.size_y / self.grid_resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        
        # Inflation 커널 계산
        inflation_cells = int(self.inflation_radius / self.grid_resolution)
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        # --- 모듈 인스턴스화 ---
        self.planner = AStarPlanner()
        self.controller = PurePursuitController(
            max_linear=max_linear_velocity,
            max_angular=max_angular_velocity,
            look_ahead=look_ahead_dist,
            turn_damping=turn_damping_factor,
            angular_gain=angular_gain
        )
        
        # --- ROS2 Setup ---
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.bev_sub = self.create_subscription(
            PointCloud2, '/bev_map', self.bev_map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 ---
        self.current_pose = None # [x, y, yaw]
        self.occupancy_grid = None
        self.inflated_grid = None
        
        # --- 웨이포인트 ---
        # 6F 
        d1 = (-5.6,0.48)
        d2 = (-4.66,7.05)
        d3 = (2.844,6.9)
        d4 = (2.85,-0.68)
        d5 = (-5.0,0.132)
        self.waypoints = [d1, d2, d3, d4,d5, d1,d2,d3, d4,d5, d1,d2,d3, d4,d5, d1,d2]


        # 1F loop
        d1 = (-0.3,1.88)
        d2 = (5.58,19.915)
        d3 = (2.606,36.25)
        d4 = (-9.88,38.336)
        d5 = (-21.88,29.57)
        
        self.waypoints = [d1, d2, d3, d4, d5,d1]
         
        self.waypoint_index = 0
        self.goal_threshold = 0.7

        # --- 제어 루프 ---
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # --- CV2 시각화 (기존 로직 유지) ---
        self.visualization_image = None
        self.latest_bev_viz_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # --- Matplotlib 시각화 데이터 ---
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = [] # (N, 2) [x, y]
        self.latest_waypoints = np.array([]) # A*가 생성한 로컬 경로
        self.latest_lookahead_point = np.array([]) # 컨트롤러가 추종하는 지점
        self.obstacle_points = np.array([]) # (N, 2) [x, y] 로컬 장애물

        self.get_logger().info("✅ A* BEV Planner Node has started.")
        self.get_logger().info(f"  Grid: {self.cells_x}x{self.cells_y} @ {self.grid_resolution}m")
        self.get_logger().info(f"  Inflation Radius: {self.inflation_radius}m ({inflation_cells} cells)")

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

    def depth_callback(self, msg):
        """뎁스 이미지를 수신하여 CV2 시각화용으로만 처리"""
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 13.0
            depth_meters = (np.clip(depth_cv, 0, max_depth_value * 1000) / 1000.0).astype(np.float32)
            depth_display = cv2.applyColorMap((depth_meters / max_depth_value * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            with self.plot_data_lock:
                self.visualization_image = cv2.resize(depth_display, (640, 360))
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def bev_map_callback(self, msg: PointCloud2):
        """/bev_map 토픽을 구독하여 Costmap을 생성하고 팽창시킴."""
        try:
            # 1. 빈 그리드 초기화
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            obstacle_points_local = []
            
            # 2. PointCloud를 순회하며 그리드 채우기 (로봇 기준 좌표계)
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                
                # 로컬 월드 좌표 -> 그리드 인덱스
                grid_c, grid_r = self.world_to_grid_idx(x, y)
                
                if 0 <= grid_r < self.cells_y and 0 <= grid_c < self.cells_x:
                    grid[grid_r, grid_c] = 255 # 점유됨 (Occupied)
                    obstacle_points_local.append([x, y])
            
            # 3. Inflation Layer 적용
            inflated = cv2.dilate(grid, self.inflation_kernel)
            
            # 4. 상태 변수 업데이트
            self.occupancy_grid = grid
            self.inflated_grid = inflated
            
            # 5. 시각화용 데이터 업데이트
            with self.plot_data_lock:
                self.obstacle_points = np.array(obstacle_points_local)
                self.latest_bev_viz_image = cv2.applyColorMap(inflated, cv2.COLORMAP_BONE)

        except Exception as e:
            self.get_logger().error(f"BEV map processing error: {e}\n{traceback.format_exc()}")

    # --- 좌표 변환 헬퍼 ---
    
    def world_to_grid_idx(self, x, y):
        """(로컬) 월드 좌표(m)를 그리드 인덱스(c, r)로 변환"""
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    def grid_idx_to_world(self, grid_r, grid_c):
        """그리드 인덱스(r, c)를 셀의 중심 (로컬) 월드 좌표(m)로 변환"""
        x = self.grid_origin_x + (grid_c + 0.5) * self.grid_resolution
        y = self.grid_origin_y + (grid_r + 0.5) * self.grid_resolution
        return x, y

    def is_valid_grid_cell(self, r, c):
        """그리드 인덱스가 유효한 범위 내에 있는지 확인"""
        return 0 <= r < self.cells_y and 0 <= c < self.cells_x

    # --- 메인 제어 루프 ---

    def control_callback(self):
        if self.inflated_grid is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                self.stop_robot()
                return

            # 1. 현재 목표 웨이포인트 (글로벌)
            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

            # 2. 목표 도달 여부 확인 (글로벌)
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                return 

            # 3. 글로벌 목표 -> 로컬 목표 변환 (로봇 기준 좌표계)
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            # --- A* 플래닝 (로컬 좌표계) ---
            
            # 4. 시작/종료 지점을 그리드 인덱스로 변환
            # 시작점은 항상 로봇의 현재 위치 (0, 0) -> 맵의 원점
            start_c_idx, start_r_idx = self.world_to_grid_idx(0.0, 0.0)
            goal_c_idx, goal_r_idx = self.world_to_grid_idx(local_x, local_y)
            
            start_rc = (start_r_idx, start_c_idx)
            goal_rc = (goal_r_idx, goal_c_idx)
            
            # 5. 시작/종료 지점 유효성 검사
            if not (self.is_valid_grid_cell(start_r_idx, start_c_idx) and \
                    self.is_valid_grid_cell(goal_r_idx, goal_c_idx)):
                self.get_logger().warn("Start or Goal is outside the BEV map.")
                self.stop_robot()
                return

            if self.inflated_grid[start_r_idx, start_c_idx] > 0:
                self.get_logger().error("CRITICAL: Robot is inside an inflated obstacle! Stopping.")
                self.stop_robot()
                return

            if self.inflated_grid[goal_r_idx, goal_c_idx] > 0:
                self.get_logger().warn("Goal is inside an inflated obstacle! Stopping.")
                self.stop_robot()
                return
            
            # 6. A* 플래너 실행
            path_grid = self.planner.plan(
                start_rc, 
                goal_rc, 
                self.inflated_grid, 
                (self.cells_y, self.cells_x)
            )

            if path_grid is None or len(path_grid) < 2:
                self.get_logger().warn("No path found or path is too short. Stopping.")
                self.stop_robot()
                return

            # 7. 그리드 경로 -> 로컬 월드 경로 변환
            path_world = [self.grid_idx_to_world(r, c) for r, c in path_grid]
            path_world_np = np.array(path_world) # (N, 2) [x, y]

            # --- 제어 ---
            
            # 8. 컨트롤러 실행
            twist, lookahead_point = self.controller.calculate_command(path_world_np)
            
            # 9. 시각화 데이터 저장
            with self.plot_data_lock:
                self.latest_waypoints = path_world_np
                self.latest_lookahead_point = lookahead_point
            
            # 10. 명령 발행
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: v={twist.linear.x:.2f} w={twist.angular.z:.2f} | "
                f"PathLen: {len(path_grid)}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.stop_robot()
            
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        with self.plot_data_lock:
            # 시각화용 데이터 초기화
            self.latest_waypoints = np.array([])
            self.latest_lookahead_point = np.array([])

    def _visualization_thread(self):
        """CV2 시각화 스레드 (뎁스 + BEV Costmap)"""
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
                bev_display = self.latest_bev_viz_image.copy() if self.latest_bev_viz_image is not None else None
            
            if display_image is not None:
                if bev_display is not None:
                    bev_resized = cv2.resize(bev_display, (320, 320))
                    
                    # 로봇 위치 표시 (BEV 맵 중앙 하단)
                    robot_x = bev_resized.shape[1] // 2
                    robot_y = self.cells_y - int(self.grid_origin_y / self.grid_resolution) # Y=0 선
                    robot_y = int(robot_y * (320.0 / self.cells_y)) # 리사이즈 스케일링
                    cv2.circle(bev_resized, (robot_x, robot_y), 8, (0, 0, 255), -1) 
                    cv2.arrowedLine(bev_resized, (robot_x, robot_y), (robot_x, robot_y - 15), (0, 255, 0), 2)
                    
                    h, w = display_image.shape[:2]
                    combined = np.zeros((h, w + 320, 3), dtype=np.uint8)
                    combined[:h, :w] = display_image
                    combined[:320, w:] = bev_resized
                    
                    cv2.putText(combined, "Inflated BEV Map", (w + 10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("A* Planner Vision + BEV", combined)
                else:
                    cv2.imshow("A* Planner Vision + BEV", display_image)
                
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")
    
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        if self.vis_thread.is_alive():
            self.vis_thread.join()
        super().destroy_node()


# ==============================================================================
# --- Main 실행 ---
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    
    # 1. ROS 노드 생성
    node = PlannerNode()

    # 2. ROS 스핀을 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 3. Matplotlib 시각화기 생성 및 실행 (메인 스레드)
    #    plt.show()는 메인 스레드에서 실행되어야 함
    visualizer = MatplotlibVisualizer(node)
    
    try:
        visualizer.run() # plt.show()가 여기서 블로킹됨
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        # 종료 시
        node.destroy_node()
        rclpy.shutdown()
        if ros_thread.is_alive():
            ros_thread.join()
        node.get_logger().info("Shutdown complete.")


if __name__ == '__main__':
    main()
