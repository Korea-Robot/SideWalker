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
import heapq  # A*를 위한 Priority Queue

# BEV Map 관련
import sensor_msgs_py.point_cloud2 as pc2

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================================
# --- ROS2 Node ---
# ==============================================================================

class AStarBevPlanner(Node):
    def __init__(self):
        super().__init__('astar_bev_planner_viz')

        # --- ROS 2 파라미터 선언 ---
        # bev_map.py와 반드시 동일한 값을 사용해야 함
        self.declare_parameter('grid_resolution', 0.1)  # meters per cell
        # self.declare_parameter('grid_size_x', 15.0)     # total width in meters
        # self.declare_parameter('grid_size_y', 15.0)     # total height in meters
        self.declare_parameter('grid_size_x', 60.0)     # total width in meters
        self.declare_parameter('grid_size_y', 60.0)     # total height in meters        
        
        # A* 플래너 및 제어 파라미터
        self.declare_parameter('inflation_radius', 0.1) # meters
        self.declare_parameter('max_linear_velocity', 0.6)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('look_ahead_dist', 0.7) # meters
        self.declare_parameter('turn_damping_factor', 1.0)

        # --- 파라미터 값 가져오기 ---
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        
        # 제어 파라미터
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.look_ahead_dist = self.get_parameter('look_ahead_dist').get_parameter_value().double_value
        self.turn_damping_factor = self.get_parameter('turn_damping_factor').get_parameter_value().double_value
        self.angular_gain = 2.0 # 각도 오차에 대한 게인

        # --- Grid 설정 (bev_map.py와 동일하게) ---
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
        
        # --- ROS2 Setup ---
        self.bridge = CvBridge()
        # 뎁스 카메라는 이제 CV 시각화 용도로만 사용
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        # BEV 맵 구독 (핵심)
        self.bev_sub = self.create_subscription(
            PointCloud2, '/bev_map', self.bev_map_callback, 10)
            
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 ---
        self.current_pose = None
        self.occupancy_grid = None  # (cells_y, cells_x)
        self.inflated_grid = None   # (cells_y, cells_x)
        self.obstacle_points = np.array([]) # (N, 2) [x, y] 월드 좌표계
        
        # --- 웨이포인트 ---
        d1 = (0.0, 0.0)
        d2 = (2.7, 0)
        d3 = (2.433, 2.274)
        d4 = (-0.223, 2.4)
        d5 = (-2.55, 5.0)

        # 1F loop
        d1 = (-0.3,1.88)
        d2 = (5.58,19.915)
        d3 = (2.606,36.25)
        d4 = (-9.88,38.336)
        d5 = (-21.88,29.57)
        
        self.waypoints = [d1, d2, d3, d4, d5,d1]

        # # 6F 
        d1 = (-5.6,0.48)
        d2 = (-4.66,7.05)
        d3 = (2.844,6.9)
        d4 = (2.85,-0.68)
        d5 = (-5.0,0.132)

        self.waypoints = [d1, d2, d3, d4,d5, d1,d2,d3, d4,d5, d1,d2,d3, d4,d5, d1,d2]                 
        # self.waypoints = [d3, d4, d5, d1,d2]
        self.waypoint_index = 0
        self.goal_threshold = 0.7

        self.control_timer = self.create_timer(0.1, self.control_callback)

        # --- CV2 시각화 ---
        self.visualization_image = None
        self.latest_bev_viz_image = None # CV2 시각화용 BEV
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # --- Matplotlib 시각화 데이터 ---
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_waypoints = np.array([]) # A*가 생성한 경로
        self.latest_local_goal = np.array([])

        self.get_logger().info("✅ A* BEV Planner (Nav2-style) has started.")
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

        # if yaw < 0.0:
        #     yaw = yaw + math.pi
        # else:
        #     yaw = yaw - math.pi
        

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
        """
        /bev_map 토픽을 구독하여 Costmap을 생성하고 팽창시킴.
        이 BEV 맵은 bev_map.py에 의해 로봇('base_link') 중심
        """
        try:
            # 1. 빈 그리드 초기화
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            obstacle_points_local = []
            
            # 2. PointCloud를 순회하며 그리드 채우기
            # TODO : use vector instead of forloop
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                x, y = point[0], point[1]
                
                # 월드 좌표 -> 그리드 인덱스
                grid_c, grid_r = self.world_to_grid_idx(x, y)
                
                # 그리드 범위 체크
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
                # CV2 시각화용 이미지 생성 (팽창된 맵)
                self.latest_bev_viz_image = cv2.applyColorMap(inflated, cv2.COLORMAP_BONE)

        except Exception as e:
            self.get_logger().error(f"BEV map processing error: {e}\n{traceback.format_exc()}")

    # --- 좌표 변환 헬퍼 ---
    
    def world_to_grid_idx(self, x, y):
        """월드 좌표(m)를 그리드 인덱스(r, c)로 변환"""
        grid_c = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_r = int((y - self.grid_origin_y) / self.grid_resolution)
        return grid_c, grid_r

    def grid_idx_to_world(self, grid_r, grid_c):
        """그리드 인덱스(r, c)를 셀의 중심 월드 좌표(m)로 변환"""
        x = self.grid_origin_x + (grid_c + 0.5) * self.grid_resolution
        y = self.grid_origin_y + (grid_r + 0.5) * self.grid_resolution
        return x, y

    def is_valid_grid_cell(self, r, c):
        """그리드 인덱스가 유효한 범위 내에 있는지 확인"""
        return 0 <= r < self.cells_y and 0 <= c < self.cells_x

    # --- A* 플래너 ---

    def a_star_planner(self, start_rc, goal_rc, costmap):
        """A* 알고리즘으로 start (r, c)에서 goal (r, c)까지의 경로 탐색"""
        
        start_r, start_c = start_rc
        goal_r, goal_c = goal_rc

        # 휴리스틱 함수 (Euclidean distance)
        def heuristic(r, c):
            return math.sqrt((r - goal_r)**2 + (c - goal_c)**2)

        # 8-방향 이웃 (대각선 포함)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        move_costs = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

        open_list = []
        heapq.heappush(open_list, (0, start_rc)) # (f_cost, (r, c))
        
        came_from = {}
        g_cost = {start_rc: 0}
        f_cost = {start_rc: heuristic(start_r, start_c)}

        while open_list:
            _, current_rc = heapq.heappop(open_list)
            current_r, current_c = current_rc

            if current_rc == goal_rc:
                # 경로 재구성
                path = []
                while current_rc in came_from:
                    path.append(current_rc)
                    current_rc = came_from[current_rc]
                path.append(start_rc)
                return path[::-1] # start -> goal 순서로 반환

            for i, (dr, dc) in enumerate(neighbors):
                neighbor_r, neighbor_c = current_r + dr, current_c + dc
                neighbor_rc = (neighbor_r, neighbor_c)
                
                # 그리드 범위 체크
                if not self.is_valid_grid_cell(neighbor_r, neighbor_c):
                    continue
                    
                # 장애물 체크 (costmap 값이 0보다 크면 장애물)
                if costmap[neighbor_r, neighbor_c] > 0:
                    continue
                    
                new_g_cost = g_cost[current_rc] + move_costs[i]
                
                if neighbor_rc not in g_cost or new_g_cost < g_cost[neighbor_rc]:
                    g_cost[neighbor_rc] = new_g_cost
                    new_f_cost = new_g_cost + heuristic(neighbor_r, neighbor_c)
                    f_cost[neighbor_rc] = new_f_cost
                    heapq.heappush(open_list, (new_f_cost, neighbor_rc))
                    came_from[neighbor_rc] = current_rc
        
        # 경로를 찾지 못함
        self.get_logger().warn("A* Planner: No path found to goal.")
        return None

    # --- 메인 제어 루프 ---

    def control_callback(self):
        # BEV 맵(Costmap)이나 현재 위치 정보가 없으면 대기
        if self.inflated_grid is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                # 모든 웨이포인트 도달
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

            # 1. 현재 목표 웨이포인트 가져오기
            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

            # 2. 목표 도달 여부 확인
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                return # 다음 콜백에서 새 웨이포인트로 플래닝

            # 3. 글로벌 목표 -> 로컬 목표 변환 (로봇 기준 좌표계)
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            # --- A* 플래닝 ---
            
            # 4. 시작/종료 지점을 그리드 인덱스로 변환
            # 시작점은 항상 로봇의 현재 위치 (0, 0)
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
                # TODO: 더 나은 처리 (가장 가까운 유효 셀 찾기 등)
                self.stop_robot()
                return
            
            # 6. A* 플래너 실행
            path_grid = self.a_star_planner(start_rc, goal_rc, self.inflated_grid)

            if path_grid is None or len(path_grid) < 2:
                self.get_logger().warn("No path found or path is too short. Stopping.")
                self.stop_robot()
                return

            # 7. 그리드 경로 -> 월드 경로 변환
            path_world = [self.grid_idx_to_world(r, c) for r, c in path_grid]
            path_world_np = np.array(path_world) # (N, 2) [x, y]

            # --- Pure Pursuit (유사) 제어 ---
            
            # 8. Lookahead 지점 찾기
            # 로봇(0,0)으로부터 path_world_np 상의 점들까지의 거리 계산
            distances = np.linalg.norm(path_world_np, axis=1)
            
            # look_ahead_dist보다 멀리 있는 첫 번째 점 찾기
            lookahead_idx = np.argmax(distances >= self.look_ahead_dist)
            
            if lookahead_idx == 0: # 경로가 lookahead 거리보다 짧은 경우
                # 그냥 마지막 점을 사용
                lookahead_idx = len(path_world_np) - 1
            
            la_x, la_y = path_world_np[lookahead_idx]

            # 9. 제어 명령 계산
            # 로봇 전방 기준 lookahead 지점까지의 각도
            target_angle = math.atan2(la_y, la_x)
            
            # P 제어기 (각도 오차에 비례)
            angular_z = self.angular_gain * target_angle
            angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)
            
            if angular_z < 0.1 and angular_z > -0.1:
                angular_z = 0.0

            # 각도가 클수록 속도 감속 (Nav2의 DWB 컨트롤러와 유사)
            linear_x = self.max_linear_velocity / (1.0 + self.turn_damping_factor * abs(angular_z))
            
            # 10. 시각화 데이터 저장
            with self.plot_data_lock:
                self.latest_waypoints = path_world_np # Matplotlib용
                self.latest_local_goal = np.array([local_x, local_y])
            
            # 11. 명령 발행
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: v={linear_x:.2f} w={angular_z:.2f} | "
                f"PathLen: {len(path_grid)} | Obstacles:{len(self.obstacle_points)}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.stop_robot()
            
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _visualization_thread(self):
        """CV2 시각화 스레드 (뎁스 + BEV Costmap)"""
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
                # Inflated BEV Map
                bev_display = self.latest_bev_viz_image.copy() if self.latest_bev_viz_image is not None else None
            
            if display_image is not None:
                if bev_display is not None:
                    bev_resized = cv2.resize(bev_display, (320, 320))
                    
                    # 로봇 위치 표시 (BEV 맵 중앙 하단)
                    robot_x = bev_resized.shape[1] // 2
                    robot_y = self.cells_y - int(self.grid_origin_y / self.grid_resolution) # Y=0 선
                    # 리사이즈에 맞게 스케일링
                    robot_y = int(robot_y * (320.0 / self.cells_y)) 
                    cv2.circle(bev_resized, (robot_x, robot_y), 8, (0, 0, 255), -1) # 빨간색
                    cv2.arrowedLine(bev_resized, (robot_x, robot_y), (robot_x, robot_y - 15), (0, 255, 0), 2) # 초록색
                    
                    # 뎁스 이미지 옆에 BEV 배치
                    h, w = display_image.shape[:2] # 360, 640
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
        self.vis_thread.join()
        super().destroy_node()


# ==============================================================================
# --- Matplotlib 시각화 함수 ---
# ==============================================================================

def update_plot(frame, node: AStarBevPlanner, ax, traj_line, 
                waypoints_line, current_point, heading_line, goal_point, 
                reached_wps_plot, pending_wps_plot, obstacle_scatter):
    
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        # A*가 생성한 로컬 경로 (로봇 기준)
        waypoints_local = node.latest_waypoints.copy()
        # 로컬 목표 (로봇 기준)
        goal_local = node.latest_local_goal.copy()
        # BEV 장애물 (로봇 기준)
        obstacles_local = node.obstacle_points.copy()
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
    if waypoints_local.size > 0 and goal_local.size > 0:
        rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                              [math.sin(current_yaw),  math.cos(current_yaw)]])
        
        # A* 경로 (로컬 -> 글로벌)
        waypoints_global = (rot_matrix @ waypoints_local[:, :2].T).T + np.array([current_x, current_y])
        # 로컬 골 (로컬 -> 글로벌)
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        
        # 장애물 포인트 (로컬 -> 글로벌)
        if obstacles_local.size > 0:
            obstacles_global = (rot_matrix @ obstacles_local.T).T + np.array([current_x, current_y])
            obstacle_scatter.set_offsets(np.c_[-obstacles_global[:, 1], obstacles_global[:, 0]])
        else:
            obstacle_scatter.set_offsets(np.empty((0, 2)))
        
        waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])

    return [traj_line, waypoints_line, current_point, 
            heading_line, goal_point, reached_wps_plot, pending_wps_plot, obstacle_scatter]


def main(args=None):
    rclpy.init(args=args)
    node = AStarBevPlanner()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_title('Real-time A* BEV Planner', fontsize=14)
    ax.set_xlabel('-Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.5
    x_max, y_max = wps_array.max(axis=0) + 1.5
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(-y_max, -y_min)
    
    # 플롯 요소들
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    
    # PlannerNet 예측 대신 A* 경로
    waypoints_line, = ax.plot([], [], 'c.-', lw=2, label='Final Path (A*)') 
    
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    
    # 장애물 포인트 표시 (산점도)
    obstacle_scatter = ax.scatter([], [], c='red', s=2, alpha=0.4, label='BEV Obstacles')
    
    ax.legend(loc='upper right', fontsize=9)
    
    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax, traj_line, 
               waypoints_line, current_point, heading_line, goal_point, 
               reached_wps_plot, pending_wps_plot, obstacle_scatter),
        interval=100, blit=True
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == '__main__':
    main()
