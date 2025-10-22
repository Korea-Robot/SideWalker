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
import torchvision.transforms.functional as TF
import threading
import time
import math
import traceback

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

# ✅ --- NEW IMPORTS for BEV and Repulsion ---
from bev_generator import SemanticBEVGenerator
from bev_repulsion_planner import apply_repulsive_force

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_node')
        self.bridge = CvBridge()
        
        # --- ROS2 Setup ---
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10) # ✅ Subscribe to RGB
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)

        # --- Member Variables ---
        self.current_pose = None
        self.current_depth_tensor = None
        self.current_rgb_image_np = None # ✅ Store RGB image
        self.current_depth_image_np = None # ✅ Store depth in meters
        self.waypoints = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)] # Example waypoints
        self.waypoint_index = 0
        self.goal_threshold = 0.6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Setup AI Models ---
        self.setup_path_planner()
        # ✅ Instantiate the BEV Generator
        self.bev_generator = SemanticBEVGenerator(model_path='best_model2.pth', device=self.device)
        self.get_logger().info("✅ Semantic BEV Generator initialized.")

        # ✅ --- Repulsion Parameters ---
        self.bev_resolution = 0.05  # 5 cm per pixel
        self.bev_size_m = 10.0      # 10x10 meter map
        self.repulsion_strength = 0.4 # How strongly obstacles push the path
        self.repulsion_radius = 1.0   # How close a waypoint must be to be pushed (meters)

        # --- Control and Visualization ---
        self.control_timer = self.create_timer(0.1, self.control_callback)
        # (Other visualization setup remains the same)
        # ...

    def setup_path_planner(self):
        # (This function is the same as your `setup_planner` function)
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"✅ PlannerNet model loaded successfully on {self.device}")

    def rgb_callback(self, msg):
        # ✅ New callback to handle RGB images
        try:
            self.current_rgb_image_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB processing error: {e}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            
            # ✅ Store depth map in meters for BEV generation
            self.current_depth_image_np = (np.clip(depth_cv, 0, max_depth_value * 1000) / 1000.0).astype(np.float32)
            
            # Create tensor for PlannerNet (normalized)
            depth_normalized_for_nn = self.current_depth_image_np / max_depth_value
            depth_tensor = torch.from_numpy(depth_normalized_for_nn).unsqueeze(0).repeat(3, 1, 1)
            depth_tensor = TF.resize(depth_tensor, [360, 640])
            self.current_depth_tensor = depth_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def control_callback(self):
        # --- Pre-condition checks ---
        if self.current_depth_tensor is None or self.current_rgb_image_np is None or self.current_pose is None:
            return
        if self.waypoint_index >= len(self.waypoints):
            # (Stop the robot logic)
            return
        
        try:
            # --- 1. Calculate Local Goal (same as before) ---
            target_wp = self.waypoints[self.waypoint_index]
            # ... (code to calculate local_x, local_y, and local_goal_tensor)
            current_x, current_y, current_yaw = self.current_pose
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([[local_x, local_y, 0.0]], dtype=torch.float32, device=self.device)

            # --- 2. Generate Initial Path with PlannerNet ---
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                initial_waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1)
                initial_waypoints_np = initial_waypoints_tensor.squeeze().cpu().numpy()

            # ✅ --- 3. Generate Semantic BEV and Extract Obstacles ---
            semantic_bev = self.bev_generator.generate_bev(
                self.current_rgb_image_np, self.current_depth_image_np,
                self.bev_resolution, self.bev_size_m
            )
            
            # Find pixel coordinates of all obstacles (anything not background)
            obstacle_indices = np.argwhere(semantic_bev > 0)
            if obstacle_indices.size > 0:
                # Convert BEV pixel coordinates (v, u) to robot coordinates (x, y)
                v_bev, u_bev = obstacle_indices[:, 0], obstacle_indices[:, 1]
                bev_pixel_size = semantic_bev.shape[0]
                x_robot = (bev_pixel_size - 1 - v_bev) * self.bev_resolution
                y_robot = (bev_pixel_size / 2 - u_bev) * self.bev_resolution
                obstacle_points_np = np.stack((x_robot, y_robot), axis=1)
            else:
                obstacle_points_np = np.empty((0, 2))

            # ✅ --- 4. Apply Repulsive Force to Modify Path ---
            modified_waypoints_np = apply_repulsive_force(
                initial_waypoints=initial_waypoints_np,
                obstacle_points=obstacle_points_np,
                strength=self.repulsion_strength,
                radius=self.repulsion_radius
            )
            
            # --- 5. Calculate Final Control Command from the MODIFIED path ---
            # Using your 'Pure Pursuit' style controller
            linear_x, angular_z = self.waypoints_to_cmd_vel(modified_waypoints_np)

            # --- 6. Publish Command ---
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)
            
            self.get_logger().info(f"Obstacles: {obstacle_points_np.shape[0]} | CMD: L={linear_x:.2f}, A={angular_z:.2f}")

            # (Update visualization data here, including original path, modified path, and obstacle points)
        
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            
    def waypoints_to_cmd_vel(self, waypoints):
        # ✅ Make sure this function accepts a numpy array
        if waypoints.shape[0] < 2: return 0.0, 0.0
        
        # (Your Pure Pursuit or other controller logic here)
        # Using a simplified version for demonstration:
        # Target the second point in the path to get a direction
        target_point = waypoints[1] if len(waypoints) > 1 else waypoints[-1]
        
        alpha = math.atan2(target_point[1], target_point[0])
        
        linear_x = 0.5 # Constant for simplicity
        angular_z = 2.0 * alpha # Proportional controller for angle
        
        # Clamp values
        angular_z = np.clip(angular_z, -1.0, 1.0)
        
        return linear_x, angular_z

    # ... (rest of your class methods like odom_callback, quaternion_to_yaw, etc.)
    def odom_callback(self, msg: Odometry):
        # (Your odom_callback logic)
        self.current_pose = [
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            self.quaternion_to_yaw(msg.pose.pose.orientation)
        ]

    def quaternion_to_yaw(self, q):
        # (Your quaternion_to_yaw logic)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
        
def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
