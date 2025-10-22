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
import os
import json
import threading
import time
import math
import traceback

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Enhanced PlannerNet imports - MODIFIED
from enhanced_planner_net import EnhancedPlannerNet, load_and_enhance_model
from traj_cost import TrajCost

class EnhancedRealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('enhanced_realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/command_odom', self.odom_callback, 10)

        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None
        self.waypoints = [(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0)]
        
        self.waypoint_index = 0
        self.goal_threshold = 0.4

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_enhanced_planner()  # MODIFIED

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
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])
        self.latest_actions = np.array([])  # ADDED: Action predictions storage

        self.get_logger().info("✅ Enhanced RealSense PlannerNet Control with Action Prediction has started.")

        # Controller Params
        self.max_linear_velocity = 0.3
        self.min_linear_velocity = 0.15
        self.max_angular_velocity = 1.0
        
        self.look_ahead_dist_base = 0.5
        self.look_ahead_dist_k = 0.3
        self.turn_damping_factor = 2.5

        # ADDED: Action class definitions
        self.action_names = {
            0: "STRAIGHT",
            1: "LEFT_SLIGHT",
            2: "LEFT_SHARP", 
            3: "RIGHT_SLIGHT",
            4: "RIGHT_SHARP",
            5: "STOP",
            6: "REVERSE"
        }

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

    def setup_enhanced_planner(self):  # MODIFIED
        """Setup enhanced planner with action prediction capabilities"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MODIFIED: Load enhanced model instead of basic PlannerNet
        model_path = "./models/enhanced_plannernet_multidata.pt"  # Change to your enhanced model path
        
        if os.path.exists(model_path):
            self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
            self.get_logger().info(f"Enhanced PlannerNet model loaded from {model_path}")
        else:
            # Fallback: load and enhance existing model
            basic_model_path = "./models/plannernet.pt"
            if os.path.exists(basic_model_path):
                self.net, _ = load_and_enhance_model(basic_model_path, self.device, num_action_classes=7)
                self.get_logger().info(f"Basic model enhanced and loaded from {basic_model_path}")
            else:
                raise FileNotFoundError(f"No model found at {model_path} or {basic_model_path}")
        
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
        self.get_logger().info(f"Enhanced PlannerNet model loaded successfully on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32)
            depth_cv[depth_cv>max_depth_value] = 0

            depth_normalized = (depth_cv / max_depth_value * 255).astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            self.current_depth_tensor = self.transform(depth_cv).unsqueeze(0).to(self.device)
            
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
                # MODIFIED: Enhanced model returns waypoints, collision_prob, and actions
                preds_tensor, collision_prob, actions_pred = self.net(self.current_depth_tensor, local_goal_tensor)

                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1)

                collision_val = collision_prob.cpu().item()  # MODIFIED: Use collision_prob instead of fear
                
                # ADDED: Process action predictions
                actions_probs = torch.nn.functional.softmax(actions_pred, dim=-1)  # (batch, k, num_classes)
                predicted_actions = torch.argmax(actions_probs, dim=-1)  # (batch, k)
                
                # Use first waypoint's action for immediate control
                immediate_action = predicted_actions[0, 0].cpu().item()
                immediate_action_prob = actions_probs[0, 0, immediate_action].cpu().item()

                linear_x, angular_z = self.enhanced_waypoints_to_cmd_vel(
                    waypoints_tensor, immediate_action, collision_val
                )  # MODIFIED: Use enhanced control function

                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                    self.latest_actions = predicted_actions.squeeze().cpu().numpy()  # ADDED
                    
                    if self.visualization_image is not None:
                        img_to_draw = self.visualization_image.copy()
                        # MODIFIED: Enhanced drawing function with action info
                        final_img = self.draw_enhanced_path_and_direction(
                            img_to_draw, waypoints_tensor, angular_z, collision_val, 
                            immediate_action, immediate_action_prob
                        )
                        self.visualization_image = final_img

            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)

            # MODIFIED: Enhanced logging with action info
            action_name = self.action_names.get(immediate_action, f"UNKNOWN_{immediate_action}")
            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: linear_x={linear_x:.2f} angular_z={angular_z:.2f} | "
                f"Collision:{collision_val:.2f} Action:{action_name}({immediate_action_prob:.2f})"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")

    def enhanced_waypoints_to_cmd_vel(self, waypoints_tensor, predicted_action, collision_prob):
        """
        MODIFIED: Enhanced control using both waypoints and predicted actions
        """
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        
        # Base Pure Pursuit control
        base_linear, base_angular = self.pure_pursuit_control(waypoints)
        
        # ADDED: Action-based modifications
        action_linear_modifier = 1.0
        action_angular_modifier = 1.0
        
        if predicted_action == 5:  # STOP
            return 0.0, 0.0
        elif predicted_action == 6:  # REVERSE
            action_linear_modifier = -0.5
        elif predicted_action in [1, 2]:  # LEFT turns
            action_angular_modifier = 1.2 if predicted_action == 2 else 1.1
        elif predicted_action in [3, 4]:  # RIGHT turns  
            action_angular_modifier = 1.2 if predicted_action == 4 else 1.1
        
        # MODIFIED: Collision-aware control
        if collision_prob > 0.8:
            return 0.0, 0.0  # Emergency stop
        elif collision_prob > 0.5:
            action_linear_modifier *= 0.5  # Slow down
        
        # Apply modifications
        final_linear = base_linear * action_linear_modifier
        final_angular = base_angular * action_angular_modifier
        
        # Clamp to limits
        final_linear = np.clip(final_linear, -self.max_linear_velocity, self.max_linear_velocity)
        final_angular = np.clip(final_angular, -self.max_angular_velocity, self.max_angular_velocity)
        
        return final_linear, final_angular

    def pure_pursuit_control(self, waypoints):
        """Original Pure Pursuit control logic (unchanged)"""
        look_ahead_dist = 1
        target_point = None
        
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue

            if p2[0] > 0 and np.linalg.norm(p2) > look_ahead_dist:
                 target_point = p2
                 break
        
        if target_point is None:
            target_point = waypoints[-1]
            if np.linalg.norm(target_point) < 0.1:
                 return 0.0, 0.0

        alpha = math.atan2(target_point[1], target_point[0])
        distance_to_target = np.linalg.norm(target_point)
        curvature = (2.0 * math.sin(alpha)) / distance_to_target
        angular_z = curvature * self.max_linear_velocity
        
        linear_x = self.max_linear_velocity / (1 + self.turn_damping_factor * abs(curvature))
        linear_x = np.clip(linear_x, self.min_linear_velocity, self.max_linear_velocity)

        angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)
        angular_z = curvature * linear_x
        angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)

        return linear_x, angular_z

    def draw_enhanced_path_and_direction(self, image, waypoints_tensor, angular_z, collision_val, 
                                       predicted_action, action_confidence):
        """MODIFIED: Enhanced drawing with action prediction info"""
        if image is None: 
            return None
        
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape

        # Draw waypoints (unchanged)
        for point in waypoints:
            wp_x, wp_y = point[0], point[1]
            Z_cam, X_cam = wp_x, -wp_y
            
            if Z_cam > 0.1:
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy)
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)

        # MODIFIED: Action-based direction display
        action_name = self.action_names.get(predicted_action, f"ACTION_{predicted_action}")
        arrow_color = (255, 255, 0)  # Default yellow
        
        # Color code by action type
        if predicted_action == 5:  # STOP
            arrow_color = (0, 0, 255)  # Red
        elif predicted_action == 6:  # REVERSE
            arrow_color = (128, 0, 128)  # Purple
        elif predicted_action in [1, 2]:  # LEFT
            arrow_color = (0, 255, 255)  # Cyan
        elif predicted_action in [3, 4]:  # RIGHT
            arrow_color = (255, 0, 255)  # Magenta

        # Draw action info
        cv2.putText(image, f"{action_name} ({action_confidence:.2f})", 
                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        
        # Draw direction arrow based on action
        arrow_end = (w // 2, h - 50)
        if predicted_action in [1, 2]:  # LEFT
            arrow_end = (w // 2 - 50, h - 50)
        elif predicted_action in [3, 4]:  # RIGHT
            arrow_end = (w // 2 + 50, h - 50)
        elif predicted_action == 6:  # REVERSE
            arrow_end = (w // 2, h + 20)
        
        if predicted_action != 5:  # Don't draw arrow for STOP
            cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)

        # MODIFIED: Collision warning (use collision_prob instead of fear_val)
        if collision_val > 0.6:
            cv2.putText(image, "!! COLLISION RISK !!", (w // 2 - 200, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif collision_val > 0.4:
            cv2.putText(image, "CAUTION - HIGH RISK", (w // 2 - 190, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

        return image

    def _visualization_thread(self):
        """CV2 뎁스 영상과 AI 판단 정보를 보여주는 스레드 (unchanged)"""
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
            
            if display_image is not None:
                cv2.imshow("Enhanced PlannerNet Vision", display_image)
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

# Matplotlib animation function - MODIFIED to include actions
def update_enhanced_plot(frame, node, ax, traj_line, preds_points, waypoints_line, current_point, 
                        heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        actions_local = node.latest_actions.copy()  # ADDED
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

    return [traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot]

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedRealSensePlannerControl()  # MODIFIED: Use enhanced node

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib setup (mostly unchanged, just function name change)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Real-time Trajectory and Enhanced PlannerNet Prediction')  # MODIFIED title
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
    
    ani = FuncAnimation(fig, update_enhanced_plot,   # MODIFIED: Use enhanced plot function
                        fargs=(node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot),
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
