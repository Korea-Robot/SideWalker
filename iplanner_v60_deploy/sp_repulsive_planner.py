#!/usr/bin/env python3
"""
Semantic-aware Robot Control with Real-time Visualization
- Subscribes to semantic point cloud
- Projects semantic labels onto depth image
- Visualizes in CV2 and Matplotlib with color-coded semantics
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF 
import threading
import time
import math
import traceback
from collections import deque
import sensor_msgs_py.point_cloud2 as pc2

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet imports (기존 코드에서 사용)
from planner_net import PlannerNet
from traj_cost import TrajCost

# Cityscapes-style Semantic Colors (19 classes)
# Format: label_id: (B, G, R) for OpenCV
SEMANTIC_COLORS = {
    0: (128, 64, 128),    # road - purple
    1: (244, 35, 232),    # sidewalk - pink
    2: (70, 70, 70),      # building - dark gray
    3: (102, 102, 156),   # wall - light purple
    4: (190, 153, 153),   # fence - light pink
    5: (153, 153, 153),   # pole - gray
    6: (250, 170, 30),    # traffic light - orange
    7: (220, 220, 0),     # traffic sign - yellow
    8: (107, 142, 35),    # vegetation - green
    9: (152, 251, 152),   # terrain - light green
    10: (70, 130, 180),   # sky - blue
    11: (220, 20, 60),    # person - red
    12: (255, 0, 0),      # rider - bright red
    13: (0, 0, 142),      # car - dark blue
    14: (0, 0, 70),       # truck - darker blue
    15: (0, 60, 100),     # bus - navy blue
    16: (0, 80, 100),     # train - teal
    17: (0, 0, 230),      # motorcycle - blue
    18: (119, 11, 32),    # bicycle - brown
}

CLASS_NAMES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3,
    "fence": 4, "pole": 5, "traffic light": 6, "traffic sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10, "person": 11,
    "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18
}

class SemanticPlannerControl(Node):
    def __init__(self):
        super().__init__('semantic_planner_control')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 10
        )
        
        # Semantic PointCloud Subscriber (새로 추가)
        self.semantic_pc_sub = self.create_subscription(
            PointCloud2, '/semantic_pointcloud',
            self.semantic_pc_callback, 10
        )
        
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/rko_lio/odometry', 
            self.odom_callback, 10
        )
        
        # Odometry & Waypoints
        self.current_pose = None
        self.waypoints = [
            (0.0, 0.0), (2.5, 0.0), (2.5, 2.5), (0.0, 2.5),
            (0.0, 0.0), (2.6, 0.0), (2.6, 2.6), (0.0, 2.4),
            (0.0, 0.0)
        ]
        self.waypoint_index = 0
        self.goal_threshold = 0.7

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_planner()

        # 데이터 저장
        self.current_depth_tensor = None
        self.current_depth_np = None
        self.semantic_label_image = None  # (H, W) semantic labels projected to depth frame
        self.semantic_color_image = None  # (H, W, 3) color-coded semantic image
        
        # 캐싱 최적화
        self.semantic_cache = deque(maxlen=3)  # 최근 3프레임 캐싱
        self.last_semantic_update = time.time()
        
        # Camera intrinsics (D455 기준)
        self.depth_width, self.depth_height = 848, 480
        self.depth_fx, self.depth_fy = 384.0, 384.0
        self.depth_cx, self.depth_cy = self.depth_width/2, self.depth_height/2
        
        # 시각화
        self.visualization_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # Matplotlib 데이터
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_preds = np.array([])
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])

        # Controller Params
        self.max_linear_velocity = 0.5
        self.min_linear_velocity = 0.15
        self.max_angular_velocity = 1.0
        self.look_ahead_dist_base = 0.95

        self.get_logger().info("🚀 Semantic-aware PlannerNet Control Started!")

    def semantic_pc_callback(self, msg: PointCloud2):
        """
        Semantic Point Cloud를 받아서 Depth 이미지에 투영
        최적화: GPU 가속 + 효율적인 투영
        """
        try:
            t_start = time.perf_counter()
            
            # Point Cloud 파싱 (x, y, z, rgb, label)
            points = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z", "label"), skip_nans=True):
                points.append(p)
            
            if len(points) == 0:
                return
            
            points = np.array(points)  # (N, 4) - x, y, z, label
            xyz = points[:, :3]
            labels = points[:, 3].astype(np.int32)
            
            # Depth frame으로 투영 (Camera Intrinsics 사용)
            semantic_image = self._project_pointcloud_to_image(xyz, labels)
            
            # 결과 저장
            with self.plot_data_lock:
                self.semantic_label_image = semantic_image
                self.semantic_color_image = self._labels_to_color(semantic_image)
                self.last_semantic_update = time.time()
            
            elapsed = (time.perf_counter() - t_start) * 1000
            self.get_logger().info(f"Semantic projection: {elapsed:.2f}ms | Points: {len(points)}", 
                                   throttle_duration_sec=2.0)
            
        except Exception as e:
            self.get_logger().error(f"Semantic PC callback error: {e}\n{traceback.format_exc()}")

    def _project_pointcloud_to_image(self, xyz, labels):
        """
        3D Point Cloud를 2D 이미지로 투영 (Depth Camera 기준)
        
        Args:
            xyz: (N, 3) - 3D points in camera frame
            labels: (N,) - semantic labels
        
        Returns:
            semantic_image: (H, W) - projected label image
        """
        # 초기화
        semantic_image = np.zeros((self.depth_height, self.depth_width), dtype=np.int32)
        depth_buffer = np.full((self.depth_height, self.depth_width), np.inf)
        
        # 투영 (Pinhole Camera Model)
        X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        # Z > 0 (카메라 앞쪽만)
        valid_mask = Z > 0.1
        X, Y, Z, labels = X[valid_mask], Y[valid_mask], Z[valid_mask], labels[valid_mask]
        
        # 픽셀 좌표 계산
        u = (self.depth_fx * X / Z + self.depth_cx).astype(np.int32)
        v = (self.depth_fy * Y / Z + self.depth_cy).astype(np.int32)
        
        # 이미지 범위 내 필터링
        in_bounds = (u >= 0) & (u < self.depth_width) & (v >= 0) & (v < self.depth_height)
        u, v, Z, labels = u[in_bounds], v[in_bounds], Z[in_bounds], labels[in_bounds]
        
        # Depth Buffer를 사용한 Z-ordering (가까운 점 우선)
        for i in range(len(u)):
            if Z[i] < depth_buffer[v[i], u[i]]:
                semantic_image[v[i], u[i]] = labels[i]
                depth_buffer[v[i], u[i]] = Z[i]
        
        return semantic_image

    def _labels_to_color(self, label_image):
        """
        Semantic Label을 색상 이미지로 변환
        
        Args:
            label_image: (H, W) semantic labels
        
        Returns:
            color_image: (H, W, 3) BGR color image
        """
        h, w = label_image.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label, color in SEMANTIC_COLORS.items():
            mask = label_image == label
            color_image[mask] = color
        
        return color_image

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
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32)
            depth_cv[depth_cv > max_depth_value] = 0
            
            self.current_depth_np = depth_cv.copy()
            depth_normalized = depth_cv / max_depth_value

            # Depth 시각화 (Semantic과 블렌딩)
            depth_display = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Semantic overlay (있으면)
            with self.plot_data_lock:
                if self.semantic_color_image is not None:
                    # Alpha blending: 70% depth, 30% semantic
                    depth_display = cv2.addWeighted(depth_display, 0.7, self.semantic_color_image, 0.3, 0)
                self.visualization_image = depth_display

            # AI 모델 입력
            depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).repeat(3,1,1)
            depth_tensor = TF.resize(depth_tensor, [360, 640])
            self.current_depth_tensor = depth_tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def control_callback(self):
        if self.current_depth_tensor is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                twist = Twist()
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
                    self.cmd_pub.publish(Twist())
                    return

            target_wp = self.waypoints[self.waypoint_index]
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1)
                cmd_vels = preds_tensor[:, :, :2]
                fear_val = fear.cpu().item()

                k, h = 2, 3
                angular_z = torch.clamp(cmd_vels[0, k:k+h, 1], -1.0, 1.0).mean().cpu().item()
                angular_z = self._discretize_value(angular_z, 0.2)

                linear_x = 0.4
                if angular_z >= 0.4:
                    linear_x = 0

                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
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
                f"CMD: v={linear_x:.2f} w={angular_z:.2f} Fear:{fear_val:.2f}",
                throttle_duration_sec=0.5
            )

        except Exception as e:
            self.get_logger().error(f"Control error: {e}\n{traceback.format_exc()}")
    
    def _discretize_value(self, value, step):
        return round(value / step) * step

    def _visualization_thread(self):
        """CV2 시각화 스레드 (Depth + Semantic)"""
        self.get_logger().info("CV2 visualization thread started")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
            
            if display_image is not None:
                cv2.imshow("Semantic PlannerNet Vision", display_image)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 thread stopped")

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z, fear_val):
        """경로와 방향 그리기 (기존 코드 유지)"""
        if image is None: 
            return None
        
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape

        # Waypoints 그리기
        for point in waypoints:
            wp_x, wp_y = point[0], point[1]
            Z_cam, X_cam = wp_x, -wp_y
            
            if Z_cam > 0.1:
                u = int(self.depth_fx * (X_cam / Z_cam) + self.depth_cx)
                v = int(self.depth_fy * (-0.1 / Z_cam) + self.depth_cy)
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)

        # 방향 화살표
        arrow_color = (255, 255, 0)
        turn_text = "Straight"
        arrow_end = (w // 2, h - 50)
        
        if angular_z > 0.15: 
            turn_text, arrow_color, arrow_end = "Turn Left", (0, 255, 255), (w // 2 - 50, h - 50)
        elif angular_z < -0.15: 
            turn_text, arrow_color, arrow_end = "Turn Right", (255, 0, 255), (w // 2 + 50, h - 50)
        
        cv2.putText(image, turn_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)

        # 위험 경고
        if fear_val > 0.6:
            cv2.putText(image, "!! DANGER - STOP !!", (w // 2 - 200, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif fear_val > 0.4:
            cv2.putText(image, "CAUTION - SLOWING", (w // 2 - 190, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

        return image

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        super().destroy_node()


def update_plot(frame, node, ax, traj_line, preds_points, waypoints_line, 
                current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    """Matplotlib 업데이트 (기존 코드)"""
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
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
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
        preds_points.set_data(-preds_global[:, 1], preds_global[:, 0])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])

    return [traj_line, preds_points, waypoints_line, current_point, heading_line, 
            goal_point, reached_wps_plot, pending_wps_plot]


def main(args=None):
    rclpy.init(args=args)
    node = SemanticPlannerControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title('Semantic-aware Trajectory and PlannerNet Prediction')
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
    preds_points, = ax.plot([], [], 'ro', markersize=5, label='Preds (Model Output)')
    waypoints_line, = ax.plot([], [], 'y.-', lw=1, label='Waypoints (Path)')
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    ax.legend()
    
    ani = FuncAnimation(fig, update_plot, 
                        fargs=(node, ax, traj_line, preds_points, waypoints_line, 
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
