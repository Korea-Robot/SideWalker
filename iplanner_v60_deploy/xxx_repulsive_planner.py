#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

# BEV 관련 임포트
from transforms3d.quaternions import quat2mat


# ==============================================================================
# --- BEV 생성 함수 ---
# ==============================================================================

def intrinsics_from_fov(width: int, height: int, fov_h_deg: float, fov_v_deg: float,
                        cx: float | None = None, cy: float | None = None) -> np.ndarray:
    fov_h = math.radians(fov_h_deg)
    fov_v = math.radians(fov_v_deg)
    fx = width  / (2.0 * math.tan(fov_h / 2.0))
    fy = height / (2.0 * math.tan(fov_v / 2.0))
    if cx is None: cx = width  / 2.0
    if cy is None: cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

DEPTH_INTRINSICS = np.array([[336.1, 0.0, 320.0], [0.0, 433.1, 240.0], [0.0, 0.0, 1.0]])

TRANS = [-0.015, 0.22, 0.05]
QUAT = [0.49, -0.51, 0.5, -0.5]

def create_hmt(translation, quaternion):
    rot_matrix = quat2mat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    hmt = np.eye(4)
    hmt[:3, :3] = rot_matrix
    hmt[:3, 3] = translation
    return hmt

EXTRINSIC_HMT = create_hmt(TRANS, QUAT)

# simplify  2D Depth => 3D pointcloud # 480,640
def unproject_depth_to_pointcloud(depth_map, camera_k):
    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_mask = (depth_map > 0) & np.isfinite(depth_map)
    z = np.where(valid_mask, depth_map, 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

# camera frame => robot frame # 3D 3D 
def apply_transform_to_pointcloud(points, hmt):
    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack((points, ones))
    transformed_points = homo_points @ hmt.T
    return transformed_points[:, :3]

# 3D point => bev map 
def create_bev_from_pointcloud(depth_image, intrinsics, extrinsics_hmt,
                               bev_resolution=0.05, bev_size_m=10.0, z_min=0.1, z_max=1.0):
    points_camera_frame = unproject_depth_to_pointcloud(depth_image, intrinsics)
    points_robot_frame = apply_transform_to_pointcloud(points_camera_frame, extrinsics_hmt)
    height_filter = (points_robot_frame[:, 2] > z_min) & (points_robot_frame[:, 2] < z_max)
    points_filtered = points_robot_frame[height_filter]
    
    bev_pixel_size = int(bev_size_m / bev_resolution)
    bev_image = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)
    
    x_robot = points_filtered[:, 0]
    y_robot = points_filtered[:, 1]
    
    u_bev = (bev_pixel_size // 2 - y_robot / bev_resolution).astype(int)
    v_bev = (bev_pixel_size - 1 - x_robot / bev_resolution).astype(int)
    
    valid_bev_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & \
                        (v_bev >= 0) & (v_bev < bev_pixel_size)
    
    u_bev_valid = u_bev[valid_bev_indices]
    v_bev_valid = v_bev[valid_bev_indices]
    
    bev_image[v_bev_valid, u_bev_valid] = 255
    
    # 장애물 포인트들의 실제 좌표 반환 (x, y)
    obstacle_points = np.column_stack((x_robot[valid_bev_indices], y_robot[valid_bev_indices]))
    
    return bev_image, obstacle_points


# ==============================================================================
# --- Repulsive Force 함수 ---
# ==============================================================================

# repulsive force 
def apply_repulsive_force(predicted_points, obstacle_points, 
                         force_strength=0.5, influence_radius=1.0, min_distance=0.1):
    """
    장애물 포인트들로부터 repulsive force를 계산하여 예측 경로 포인트를 수정
    
    Args:
        predicted_points: (N, 2) numpy array - 플래너가 예측한 경로 포인트 (x, y)
        obstacle_points: (M, 2) numpy array - BEV에서 추출한 장애물 포인트 (x, y)
        force_strength: repulsive force의 강도
        influence_radius: 장애물의 영향 범위 (미터)
        min_distance: 최소 거리 (division by zero 방지)
    
    Returns:
        adjusted_points: (N, 2) numpy array - 수정된 경로 포인트
    """
    if len(obstacle_points) == 0 or len(predicted_points) == 0:
        return predicted_points
    
    adjusted_points = predicted_points.copy()
    
    for i, point in enumerate(predicted_points):
        # 각 예측 포인트에 대해 모든 장애물과의 거리 계산
        distances = np.linalg.norm(obstacle_points - point, axis=1)
        
        # 영향 범위 내의 장애물만 고려
        mask = distances < influence_radius
        nearby_obstacles = obstacle_points[mask]
        nearby_distances = distances[mask]
        
        if len(nearby_obstacles) == 0:
            continue
        
        # Repulsive force 계산
        total_force = np.array([0.0, 0.0])
        for obs, dist in zip(nearby_obstacles, nearby_distances):
            # 거리가 너무 가까우면 최소 거리로 클리핑
            dist = max(dist, min_distance)
            
            # 장애물에서 멀어지는 방향 벡터
            direction = (point - obs) / dist
            
            # 거리의 제곱에 반비례하는 힘 (가까울수록 강한 힘)
            force_magnitude = force_strength * (1.0 / dist**2)
            
            # 영향 범위에 따른 감쇠
            decay_factor = 1.0 - (dist / influence_radius)
            
            total_force += direction * force_magnitude * decay_factor
        
        # 힘을 적용하여 포인트 수정
        adjusted_points[i] += total_force
    
    return adjusted_points


def apply_repulsive_force_tensor(predicted_points_tensor, obstacle_points,
                                 force_strength=0.5, influence_radius=1.0, min_distance=0.1):
    """
    Tensor 버전의 repulsive force 적용
    
    Args:
        predicted_points_tensor: (1, N, 3) torch tensor - 플래너 예측 포인트
        obstacle_points: (M, 2) numpy array - 장애물 포인트
        
    Returns:
        adjusted_tensor: (1, N, 3) torch tensor - 수정된 예측 포인트
    """
    device = predicted_points_tensor.device
    batch_size = predicted_points_tensor.shape[0]
    
    adjusted_tensor = predicted_points_tensor.clone()
    
    for b in range(batch_size):
        # (N, 3) -> (N, 2) xy만 추출
        pred_points_np = predicted_points_tensor[b, :, :2].cpu().numpy()
        
        # Repulsive force 적용
        adjusted_points = apply_repulsive_force(
            pred_points_np, obstacle_points,
            force_strength=force_strength,
            influence_radius=influence_radius,
            min_distance=min_distance
        )
        
        # 다시 텐서로 변환하여 업데이트
        adjusted_tensor[b, :, :2] = torch.from_numpy(adjusted_points).to(device)
    
    return adjusted_tensor


# ==============================================================================
# --- ROS2 Node ---
# ==============================================================================

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)
        
        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None

        d1 = (0.0, 0.0)
        d2 = (2.7, 0)
        d3 = (2.433, 2.274)
        d4 = (-0.223, 2.4)
        d5 = (-2.55, 5.0)
        self.waypoints = [d1, d2, d3, d1, d4, d5]
        
        self.waypoint_index = 0
        self.goal_threshold = 0.6

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_planner()

        self.current_depth_tensor = None
        self.current_depth_image = None  # BEV 생성용 원본 depth
        self.obstacle_points = np.array([])  # BEV에서 추출한 장애물 포인트
        self.latest_bev_image = None  # 시각화용 BEV 이미지
        
        self.angular_gain = 2.0
        self.depth_cv = None

        # BEV 파라미터
        self.bev_resolution = 0.05
        self.bev_size_m = 10.0
        self.z_min = 0.2
        self.z_max = 1.0
        
        # Repulsive force 파라미터
        self.repulsive_force_strength = 0.3
        self.repulsive_influence_radius = 1.2
        self.repulsive_min_distance = 0.15

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
        self.latest_preds_adjusted = np.array([])  # Repulsive force 적용 후
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])

        self.get_logger().info("✅ RealSense PlannerNet with BEV Repulsive Force has started.")

        # Controller Params
        self.max_linear_velocity = 0.5
        self.min_linear_velocity = 0.15
        self.max_angular_velocity = 1.0
        self.look_ahead_dist_base = 0.95
        self.look_ahead_dist_k = 0.3
        self.turn_damping_factor = 2.5

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
        if torch.cuda.is_available(): 
            self.net = self.net.cuda()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(([360, 640])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded successfully on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            
            # BEV 생성용 원본 depth (미터 단위)
            depth_meters = (np.clip(depth_cv, 0, max_depth_value * 1000) / 1000.0).astype(np.float32)
            depth_meters[depth_meters > max_depth_value] = 0
            self.current_depth_image = depth_meters
            
            # BEV 생성 및 장애물 포인트 추출
            bev_image, obstacle_points = create_bev_from_pointcloud(
                depth_image=depth_meters,
                intrinsics=DEPTH_INTRINSICS,
                extrinsics_hmt=EXTRINSIC_HMT,
                bev_resolution=self.bev_resolution,
                bev_size_m=self.bev_size_m,
                z_min=self.z_min,
                z_max=self.z_max
            )
            
            self.obstacle_points = obstacle_points
            self.latest_bev_image = bev_image
            
            # AI 모델 입력용 정규화
            depth_normalized = depth_meters / max_depth_value
            depth_display = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0)
            depth_tensor = depth_tensor.repeat(3, 1, 1)
            depth_tensor = TF.resize(depth_tensor, [360, 640])
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
                # 원본 예측
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                
                # Repulsive force 적용
                preds_adjusted = apply_repulsive_force_tensor(
                    preds_tensor,
                    self.obstacle_points,
                    force_strength=self.repulsive_force_strength,
                    influence_radius=self.repulsive_influence_radius,
                    min_distance=self.repulsive_min_distance
                )
                
                # 수정된 예측으로 경로 생성
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_adjusted, step=0.1)
                cmd_vels = preds_adjusted[:, :, :2]
                fear_val = fear.cpu().item()

                k = 2
                h = 3
                angular_z = torch.clamp(cmd_vels[0, k:k+h, 1], -1.0, 1.0).mean().cpu().item()
                angular_z = self._discretize_value(angular_z, 0.2)

                linear_x = 0.4
                if angular_z >= 0.4:
                    linear_x = 0

                stop_distance = 0.01
                width, height = 640, 360
                roi_x_start = int(width * 0.4)
                roi_x_end = int(width * 0.6)
                roi_y_start = int(height * 0.4)
                roi_y_end = int(height * 0.6)

                front_roi = self.current_depth_tensor[:, 0, roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                if torch.mean(front_roi).item() < stop_distance:
                    linear_x = 0.0
                    angular_z = 0.0

                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                    self.latest_preds_adjusted = preds_adjusted.squeeze().cpu().numpy()
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                    
                    if self.visualization_image is not None:
                        img_to_draw = self.visualization_image.copy()
                        final_img = self.draw_path_and_direction(img_to_draw, waypoints_tensor, angular_z, fear_val)
                        self.visualization_image = final_img

            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: v={linear_x:.2f} w={angular_z:.2f} Fear:{fear_val:.2f} | "
                f"Obstacles:{len(self.obstacle_points)}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
    
    def _discretize_value(self, value, step):
        return round(value / step) * step

    def _visualization_thread(self):
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
                bev_display = self.latest_bev_image.copy() if self.latest_bev_image is not None else None
            
            if display_image is not None:
                # BEV 이미지도 함께 표시
                if bev_display is not None:
                    bev_colored = cv2.applyColorMap(bev_display, cv2.COLORMAP_BONE)
                    bev_resized = cv2.resize(bev_colored, (320, 320))
                    
                    # 로봇 위치 표시
                    robot_x = bev_resized.shape[1] // 2
                    robot_y = bev_resized.shape[0] - 10
                    cv2.circle(bev_resized, (robot_x, robot_y), 8, (0, 0, 255), -1)
                    
                    # Depth 이미지 옆에 BEV 배치
                    h, w = display_image.shape[:2]
                    combined = np.zeros((h, w + 320, 3), dtype=np.uint8)
                    combined[:h, :w] = display_image
                    combined[:320, w:] = bev_resized
                    
                    cv2.putText(combined, "BEV Map", (w + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("PlannerNet Vision + BEV", combined)
                else:
                    cv2.imshow("PlannerNet Vision + BEV", display_image)
                    
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z, fear_val):
        if image is None: 
            return None
        
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

        if fear_val > 0.6:
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


def update_plot(frame, node, ax, traj_line, preds_points, preds_adjusted_points, 
                waypoints_line, current_point, heading_line, goal_point, 
                reached_wps_plot, pending_wps_plot, obstacle_scatter):
    
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        preds_adjusted_local = node.latest_preds_adjusted.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        obstacles_local = node.obstacle_points.copy()
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
        
        # 장애물 포인트를 글로벌 좌표로 변환
        if obstacles_local.size > 0:
            obstacles_global = (rot_matrix @ obstacles_local.T).T + np.array([current_x, current_y])
            obstacle_scatter.set_offsets(np.c_[-obstacles_global[:, 1], obstacles_global[:, 0]])
        else:
            obstacle_scatter.set_offsets(np.empty((0, 2)))
        
        waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
        preds_points.set_data(-preds_global[:, 1], preds_global[:, 0])
        preds_adjusted_points.set_data(-preds_adjusted_global[:, 1], preds_adjusted_global[:, 0])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])

    return [traj_line, preds_points, preds_adjusted_points, waypoints_line, current_point, 
            heading_line, goal_point, reached_wps_plot, pending_wps_plot, obstacle_scatter]


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_title('Real-time Trajectory with BEV Repulsive Force', fontsize=14)
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
    preds_points, = ax.plot([], [], 'ro', markersize=6, alpha=0.5, label='Original Preds')
    preds_adjusted_points, = ax.plot([], [], 'mo', markersize=6, label='Adjusted Preds (w/ Repulsive)')
    waypoints_line, = ax.plot([], [], 'y.-', lw=2, label='Final Path')
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    
    # 장애물 포인트 표시 (산점도)
    obstacle_scatter = ax.scatter([], [], c='red', s=1, alpha=0.3, label='BEV Obstacles')
    
    ax.legend(loc='upper right', fontsize=9)
    
    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax, traj_line, preds_points, preds_adjusted_points, 
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
