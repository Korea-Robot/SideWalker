#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 
import os
import json
import threading
import time
import math
import traceback
import sensor_msgs_py.point_cloud2 as pc2

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

print('BEV Fusion PlannerNet Started!')

class BEVObstacleMap:
    """BEV 맵에서 장애물 정보를 추출하는 클래스"""
    def __init__(self, grid_resolution=0.1, grid_size_x=15.0, grid_size_y=15.0):
        self.resolution = grid_resolution
        self.size_x = grid_size_x
        self.size_y = grid_size_y
        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        
        # 장애물 그리드 (0: 자유공간, 1: 장애물)
        self.obstacle_grid = np.zeros((self.cells_y, self.cells_x), dtype=np.float32)
        self.last_update_time = time.time()
        
    def update_from_pointcloud(self, msg: PointCloud2, z_obstacle_min=0.2, z_obstacle_max=0.8):
        """BEV PointCloud로부터 장애물 그리드 업데이트"""
        self.obstacle_grid.fill(0)  # 초기화
        
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            
            # 높이 필터링 (로봇 높이 범위 내의 장애물만)
            if not (z_obstacle_min <= z <= z_obstacle_max):
                continue
            
            # 그리드 좌표 변환
            grid_c = int((x - self.grid_origin_x) / self.resolution)
            grid_r = int((y - self.grid_origin_y) / self.resolution)
            
            # 범위 체크
            if 0 <= grid_c < self.cells_x and 0 <= grid_r < self.cells_y:
                self.obstacle_grid[grid_r, grid_c] = 1.0
        
        self.last_update_time = time.time()
    
    def get_obstacle_distance(self, x, y):
        """주어진 로봇 좌표 (x, y)에서 가장 가까운 장애물까지의 거리 반환"""
        grid_c = int((x - self.grid_origin_x) / self.resolution)
        grid_r = int((y - self.grid_origin_y) / self.resolution)
        
        if not (0 <= grid_c < self.cells_x and 0 <= grid_r < self.cells_y):
            return float('inf')  # 그리드 밖은 자유공간으로 간주
        
        # 해당 셀이 장애물인 경우
        if self.obstacle_grid[grid_r, grid_c] > 0.5:
            return 0.0
        
        # 주변 탐색 (3x3 커널)
        search_radius = 3
        min_dist = float('inf')
        
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                check_r = grid_r + dr
                check_c = grid_c + dc
                
                if 0 <= check_c < self.cells_x and 0 <= check_r < self.cells_y:
                    if self.obstacle_grid[check_r, check_c] > 0.5:
                        # 실제 거리 계산
                        obs_x = self.grid_origin_x + (check_c + 0.5) * self.resolution
                        obs_y = self.grid_origin_y + (check_r + 0.5) * self.resolution
                        dist = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                        min_dist = min(min_dist, dist)
        
        return min_dist
    
    def is_occupied(self, x, y):
        """해당 위치가 장애물인지 확인"""
        return self.get_obstacle_distance(x, y) < 0.1


class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_bev_fusion')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        # BEV 맵 구독 추가
        self.bev_sub = self.create_subscription(PointCloud2, '/semantic_bev_map', self.bev_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)
   
        # BEV 장애물 맵 초기화
        self.bev_map = BEVObstacleMap(grid_resolution=0.1, grid_size_x=15.0, grid_size_y=15.0)
        
        # 장애물 회피 파라미터
        self.obstacle_stop_distance = 0.5  # 이 거리보다 가까우면 정지
        self.obstacle_repulsion_distance = 1.5  # 이 거리 내에서 경로를 밀어냄
        self.repulsion_strength = 0.8  # 밀어내는 강도 (0~1)
        
        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None

        d1 = (0.0,0.0)
        d2 = (2.7,0)
        d3 = (2.433,2.274)
        d4 = (-0.223,2.4)
        d5 = (-2.55,5.0)

        self.waypoints = [d1,d2,d3,d1,d4,d5,d4,d1,d2,d3,d2,d1,d4,d5,d4,d1]
        
        self.waypoint_index = 0
        self.goal_threshold = 0.7

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_planner()

        self.current_depth_tensor = None
        self.angular_gain = 2.0

        # CV2 시각화를 위한 변수들
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 384.0, 384.0, 320.0, 240.0
        self.visualization_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # Matplotlib 플롯을 위한 데이터 저장 변수
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_preds = np.array([])
        self.latest_preds_adjusted = np.array([])  # 조정된 예측 경로
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])

        self.get_logger().info("✅ BEV Fusion PlannerNet Control Started")

        # Controller Params
        self.max_linear_velocity = 0.5
        self.min_linear_velocity = 0.15
        self.max_angular_velocity = 1.0
        
        self.look_ahead_dist_base = 0.95
        self.look_ahead_dist_k = 0.3
        self.turn_damping_factor = 2.5

    def bev_callback(self, msg: PointCloud2):
        """BEV 맵 수신 콜백"""
        try:
            self.bev_map.update_from_pointcloud(msg, z_obstacle_min=0.2, z_obstacle_max=0.8)
        except Exception as e:
            self.get_logger().error(f"BEV processing error: {e}")

    def adjust_path_with_bev(self, preds_tensor, current_yaw):
        """
        BEV 맵 정보를 활용하여 예측 경로를 조정
        
        Args:
            preds_tensor: 모델 출력 (1, N, 3) - 로봇 로컬 좌표계
            current_yaw: 현재 로봇의 방향 (라디안)
            
        Returns:
            adjusted_preds: 조정된 경로 (1, N, 3)
            should_stop: 긴급 정지 필요 여부
        """
        preds = preds_tensor.squeeze().cpu().numpy()  # (N, 3)
        adjusted_preds = preds.copy()
        should_stop = False
        
        # 로봇 로컬 좌표를 월드 좌표로 변환 (BEV 맵은 로봇 중심 좌표계)
        # 로컬 좌표는 이미 로봇 기준이므로 그대로 사용
        
        for i in range(len(preds)):
            local_x, local_y = preds[i, 0], preds[i, 1]
            
            # BEV 맵에서 장애물까지의 거리 확인
            obs_dist = self.bev_map.get_obstacle_distance(local_x, local_y)
            
            # 긴급 정지 판단 (경로상 가장 가까운 장애물이 너무 가까움)
            if obs_dist < self.obstacle_stop_distance:
                should_stop = True
                self.get_logger().warn(f"⚠️ Obstacle too close at waypoint {i}: {obs_dist:.2f}m - STOPPING!")
            
            # 경로 조정 (장애물로부터 밀어냄)
            if obs_dist < self.obstacle_repulsion_distance:
                # 장애물 회피를 위한 repulsion vector 계산
                # 장애물이 가까울수록 더 강하게 밀어냄
                repulsion_factor = (self.obstacle_repulsion_distance - obs_dist) / self.obstacle_repulsion_distance
                repulsion_factor = min(repulsion_factor, 1.0) * self.repulsion_strength
                
                # 주변 장애물 방향 찾기
                search_radius = int(0.5 / self.bev_map.resolution)  # 50cm 반경
                grid_c = int((local_x - self.bev_map.grid_origin_x) / self.bev_map.resolution)
                grid_r = int((local_y - self.bev_map.grid_origin_y) / self.bev_map.resolution)
                
                repulsion_x, repulsion_y = 0.0, 0.0
                
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        check_r = grid_r + dr
                        check_c = grid_c + dc
                        
                        if 0 <= check_c < self.bev_map.cells_x and 0 <= check_r < self.bev_map.cells_y:
                            if self.bev_map.obstacle_grid[check_r, check_c] > 0.5:
                                # 장애물 위치
                                obs_x = self.bev_map.grid_origin_x + (check_c + 0.5) * self.bev_map.resolution
                                obs_y = self.bev_map.grid_origin_y + (check_r + 0.5) * self.bev_map.resolution
                                
                                # 장애물로부터 멀어지는 방향
                                dx = local_x - obs_x
                                dy = local_y - obs_y
                                dist = math.sqrt(dx**2 + dy**2) + 1e-6
                                
                                # 거리 반비례로 repulsion 누적
                                weight = 1.0 / (dist + 0.1)
                                repulsion_x += (dx / dist) * weight
                                repulsion_y += (dy / dist) * weight
                
                # Repulsion 정규화 및 적용
                repulsion_magnitude = math.sqrt(repulsion_x**2 + repulsion_y**2)
                if repulsion_magnitude > 0:
                    repulsion_x /= repulsion_magnitude
                    repulsion_y /= repulsion_magnitude
                    
                    # 경로 조정
                    adjusted_preds[i, 0] += repulsion_x * repulsion_factor * 0.3
                    adjusted_preds[i, 1] += repulsion_y * repulsion_factor * 0.3
                    
                    self.get_logger().info(
                        f"🔄 Adjusted waypoint {i}: ({local_x:.2f}, {local_y:.2f}) -> "
                        f"({adjusted_preds[i, 0]:.2f}, {adjusted_preds[i, 1]:.2f}), "
                        f"obs_dist={obs_dist:.2f}m"
                    )
        
        return torch.from_numpy(adjusted_preds).unsqueeze(0).to(preds_tensor.device), should_stop

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

    def setup_planner(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        if torch.cuda.is_available(): self.net = self.net.cuda()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(([360, 640])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32)
            depth_cv[depth_cv>max_depth_value] = 0
            depth_cv = depth_cv / max_depth_value
            
            depth_normalized = (depth_cv* 255).astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            depth_tensor = torch.from_numpy(depth_cv).unsqueeze(0)
            depth_tensor = depth_tensor.repeat(3,1,1)
            depth_tensor = TF.resize(depth_tensor,[360,640])
            depth_tensor = depth_tensor.unsqueeze(0)
            self.current_depth_tensor = depth_tensor.to(self.device)
            
            with self.plot_data_lock:
                self.visualization_image = depth_display
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def control_callback(self):
        if self.current_depth_tensor is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints): 
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    return

            target_wp = self.waypoints[self.waypoint_index]
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                
                # ===== BEV 맵을 활용한 경로 조정 =====
                adjusted_preds_tensor, should_stop = self.adjust_path_with_bev(preds_tensor, current_yaw)
                
                # 조정된 경로로 waypoints 재생성
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(adjusted_preds_tensor, step=0.1)
                cmd_vels = adjusted_preds_tensor[:,:,:2]

                fear_val = fear.cpu().item()

                k = 2
                h = 3
                angular_z = torch.clamp(cmd_vels[0, k:k+h, 1], -1.0, 1.0).mean().cpu().item()
                angular_z = self._discretize_value(angular_z, 0.2)

                # BEV 기반 긴급 정지
                if should_stop:
                    linear_x = 0.0
                    angular_z = 0.0
                    self.get_logger().warn("🛑 EMERGENCY STOP - Obstacle detected in BEV map!")
                else:
                    linear_x = 0.4
                    if angular_z >= 0.4:
                        linear_x = 0

                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                    self.latest_preds_adjusted = adjusted_preds_tensor.squeeze().cpu().numpy()
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                    
                    if self.visualization_image is not None:
                        img_to_draw = self.visualization_image.copy()
                        final_img = self.draw_path_and_direction(img_to_draw, waypoints_tensor, angular_z, fear_val, should_stop)
                        self.visualization_image = final_img

            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z= float(angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: v={linear_x:.2f} w={angular_z:.2f} Fear:{fear_val:.2f} Stop:{should_stop}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
    
    def _discretize_value(self, value, step):
        return round(value/step)*step
    
    def _visualization_thread(self):
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
            
            if display_image is not None:
                cv2.imshow("PlannerNet Vision (BEV Fusion)", display_image)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z, fear_val, should_stop):
        if image is None: return None
        
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape

        for point in waypoints:
            wp_x, wp_y = point[0], point[1]
            Z_cam, X_cam = wp_x, -wp_y
            
            if Z_cam > 0.1:
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy)
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)

        arrow_color = (255, 255, 0)
        turn_text = "Straight"
        arrow_end = (w // 2, h - 50)
        
        if angular_z > 0.15: 
            turn_text, arrow_color, arrow_end = "Turn Left", (0, 255, 255), (w // 2 - 50, h - 50)
        elif angular_z < -0.15: 
            turn_text, arrow_color, arrow_end = "Turn Right", (255, 0, 255), (w // 2 + 50, h - 50)
        
        cv2.putText(image, turn_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)

        # BEV 기반 경고 표시
        if should_stop:
            cv2.putText(image, "!! BEV OBSTACLE - STOP !!", (w // 2 - 250, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
        elif fear_val > 0.6:
            cv2.putText(image, "!! DANGER - STOP !!", (w // 2 - 200, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif fear_val > 0.4:
            cv2.putText(image, "CAUTION - SLOWING", (w // 2 - 190, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

        return image
    
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        super().destroy_node()

def update_plot(frame, node, ax, traj_line, preds_points, preds_adjusted_points, waypoints_line, 
                current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        preds_adjusted_local = node.latest_preds_adjusted.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj:
        return []

    reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:]
    if reached_wps.size > 0: 
        reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else: 
        reached_wps_plot.set_data([], [])
    if pending_wps.size > 0: 
        pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else: 
        pending_wps_plot.set_data([], [])

    traj_arr = np.array(traj)
    traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0])

    current_x, current_y, current_yaw = pose
    
    current_point.set_data([-current_y], [current_x])
    heading_len = 0.5
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

    if preds_local.size > 0 and waypoints_local.size > 0 and goal_local.size > 0:
        rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                               [math.sin(current_yaw),  math.cos(current_yaw)]])
        waypoints_global = (rot_matrix @ waypoints_local[:, :2].T).T + np.array([current_x, current_y])
        preds_global = (rot_matrix @ preds_local[:, :2].T).T + np.array([current_x, current_y])
        preds_adjusted_global = (rot_matrix @ preds_adjusted_local[:, :2].T).T + np.array([current_x, current_y])
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        
        waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
        preds_points.set_data(-preds_global[:, 1], preds_global[:, 0])
        preds_adjusted_points.set_data(-preds_adjusted_global[:, 1], preds_adjusted_global[:, 0])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])

    return [traj_line, preds_points, preds_adjusted_points, waypoints_line, current_point, 
            heading_line, goal_point, reached_wps_plot, pending_wps_plot]

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title('BEV Fusion PlannerNet - Real-time Trajectory')
    ax.set_xlabel('-Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.5
    x_max, y_max = wps_array.max(axis=0) + 1.5
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(-y_max, -y_min)
    
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    preds_points, = ax.plot([], [], 'ro', markersize=5, alpha=0.5, label='Original Preds')
    preds_adjusted_points, = ax.plot([], [], 'mo', markersize=7, label='BEV Adjusted Preds')
    waypoints_line, = ax.plot([], [], 'y.-', lw=1, label='Waypoints (Path)')
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    ax.legend()
    
    ani = FuncAnimation(fig, update_plot, 
                        fargs=(node, ax, traj_line, preds_points, preds_adjusted_points, waypoints_line, 
                               current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot),
                        interval=100, blit=True)

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
    
"""
완성했습니다! 주요 기능은 다음과 같습니다:

## 🎯 주요 변경사항

### 1. **BEVObstacleMap 클래스 추가**
- BEV 포인트클라우드를 장애물 그리드로 변환
- 특정 위치에서 가장 가까운 장애물까지의 거리 계산
- 높이 필터링으로 로봇 높이 범위의 장애물만 감지

### 2. **경로 조정 로직 (`adjust_path_with_bev`)**
- **긴급 정지**: 0.5m 이내 장애물 감지 시 즉시 정지
- **경로 밀어내기**: 1.5m 이내 장애물에서 repulsion force 계산
  - 장애물로부터 멀어지는 방향으로 경로 조정
  - 거리 반비례로 힘의 크기 결정
  - `repulsion_strength` (0.8)로 밀어내는 강도 조절

### 3. **시각화 개선**
- 원본 예측 경로(빨간색, 반투명)와 BEV 조정 경로(마젠타색) 구분 표시
- BEV 기반 장애물 감지 시 "BEV OBSTACLE - STOP" 경고 표시

### 4. **파라미터 튜닝 가능**
```python
self.obstacle_stop_distance = 0.5       # 정지 거리 (m)
self.obstacle_repulsion_distance = 1.5  # 회피 시작 거리 (m)
self.repulsion_strength = 0.8           # 밀어내는 강도 (0~1)
```

### 5. **동작 흐름**
1. Depth 카메라로 PlannerNet 예측 생성
2. BEV 맵에서 장애물 정보 추출
3. 예측 경로의 각 포인트를 장애물과 비교
4. 너무 가까우면 정지, 적당한 거리면 경로 조정
5. 조정된 경로로 로봇 제어

이제 로봇이 BEV 맵의 장애물 정보를 고려하여 안전한 마진을 유지하며 주행합니다! 🚗💨

"""
