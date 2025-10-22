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

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        

        self.current_pose = None # [x,y,yaw]
        self.waypoints = [(3.0,0.0),(3.0,3.0),(0.0,3.0),(0.0,0.0)] # ego odom coordinate 

        self.waypoint_index = 0 # current waypoint index
        self.goal_threshodl = 0.4 # distance to change 

        # odom subscriber 
        self.odom_sub = self.create_subscription(
            Odometry,
            '/command_odom',  # from user's CmdVelToOdom node
            self.odom_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)

        # Control Timer (10Hz) 10hz decision making.
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # PlannerNet Initialization
        self.setup_planner()

        # State Variables
        self.current_depth_tensor = None
        self.goal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32) # 1m forward goal
        self.angular_gain = 2.0

        # Camera Intrinsics for Visualization
        # (Assuming standard RealSense D435 parameters, adjust if needed)
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 384.0, 384.0, 320.0, 240.0


        # another thread for visualize image
        # Visualization Thread Setup
        self.visualization_image = None
        self.running = True
        self.data_lock = threading.Lock()
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()

        self.get_logger().info("✅ RealSense PlannerNet Control with Visualization has started.")

    def quaternion_to_yaw(self, q):
        """쿼터니언을 Yaw 각도로 변환하는 헬퍼 함수"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        """Odometry 메시지를 받아 현재 위치와 방향(yaw)을 업데이트합니다."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation_q)
        
        self.current_pose = [x, y, yaw]
        # self.get_logger().info(f"Odom updated: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.2f}")


    # visualize thread 33fps 
    def _visualization_thread(self):
        """Dedicated thread for handling cv2.imshow to avoid blocking the main ROS loop."""
        self.get_logger().info("Starting visualization thread.")
        while self.running and rclpy.ok():
            with self.data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None

            if display_image is not None:
                cv2.imshow("PlannerNet Navigation", display_image)
                cv2.waitKey(100) # ~33 FPS
            else:
                time.sleep(0.1) # Wait for an image to become available

        cv2.destroyAllWindows()
        self.get_logger().info("Visualization thread stopped.")

    def setup_planner(self):
        """Sets up the PlannerNet model."""
        config_path = os.path.join(os.path.dirname(os.getcwd()), 'config', 'training_config.json')
        with open(config_path) as f:
            config = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['dataConfig']['crop-size'])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded successfully on {self.device}")

    def depth_callback(self, msg):
        """Processes depth image and prepares it for both the model and visualization."""
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_cv = np.clip(depth_cv, 0, 5000) # Clip depth to 5 meters
            depth_normalized = (depth_cv / 5000.0 * 255).astype(np.uint8)

            # Prepare tensor for the neural network
            self.current_depth_tensor = self.transform(depth_normalized).unsqueeze(0).to(self.device)

            # Create a color image for visualization and share it with the viz thread
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            with self.data_lock:
                self.visualization_image = depth_display

        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def waypoints_to_cmd_vel(self, waypoints, dt=0.1):
        """Converts trajectory waypoints to linear and angular velocity commands."""
        if waypoints.shape[1] < 2:
            return torch.zeros(1, 1, 2, device=waypoints.device)

        dx = waypoints[:, 1:, 0] - waypoints[:, :-1, 0]
        dy = waypoints[:, 1:, 1] - waypoints[:, :-1, 1]
        
        linear_x = torch.sqrt(dx**2 + dy**2) / dt
        heading_angles = torch.atan2(dy, dx)
        angular_z = torch.zeros_like(linear_x)
        
        if waypoints.shape[1] > 2:
            angle_diff = heading_angles[:, 1:] - heading_angles[:, :-1]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angular_z[:, 1:] = angle_diff / dt
        
        # Calculate the initial angular velocity based on the first heading angle
        angular_z[:, 0] = heading_angles[:, 0] / (dt * self.angular_gain)
        
        return torch.stack([linear_x, angular_z], dim=-1)

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z):
        """Draws the planned path and turning direction on the image."""
        if image is None:
            return None
            
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape

        # Draw waypoints
        for i, point in enumerate(waypoints):
            wp_x ,wp_y = point[0],point[1] # select only x and y
            # Coordinate transformation: robot frame (x-fwd, y-left) to camera frame (z-fwd, x-right)
            Z_cam, X_cam = wp_x, -wp_y
            if Z_cam > 0.1: # Only draw points in front of the camera
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy) # Project to a plane slightly below camera center
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)

        # Determine and draw turning direction
        turn_text = "Straight"
        arrow_color = (255, 255, 0) # Cyan for Straight
        arrow_end = (w // 2, h - 50)
        
        if angular_z > 0.15: # Threshold for turning
            turn_text = "Turn Left"
            arrow_color = (0, 255, 255) # Yellow for Left
            arrow_end = (w // 2 - 50, h - 50)
        elif angular_z < -0.15:
            turn_text = "Turn Right"
            arrow_color = (255, 0, 255) # Magenta for Right
            arrow_end = (w // 2 + 50, h - 50)

        cv2.putText(image, turn_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)
        return image


    # main control loop 
    def control_callback(self):
        """Main control loop for inference and command publishing."""
        if self.current_depth_tensor is None:
            return

        try:
            with torch.no_grad():
                goal_tensor = self.goal.unsqueeze(0).to(self.device)
                
                # PlannerNet inference
                preds, fear = self.net(self.current_depth_tensor, goal_tensor)
                
                # Generate trajectory and command velocities
                waypoints = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
                cmd_vels = self.waypoints_to_cmd_vel(waypoints)
                
                # Clamp velocities
                linear_x = torch.clamp(cmd_vels[0, 0, 0], -1.0, 1.0).item()
                angular_z = torch.clamp(cmd_vels[0, 0, 1], -1.0, 1.0).item()
                
                fear_val = fear.cpu().item()
                
                # Fear-based safety stop/reverse
                if fear_val < 0.3: # High fear -> obstacle is very close
                     linear_x = 0.0
                     angular_z = 0.0
                elif fear_val < 0.1: # Moderate fear -> slow down
                    #  linear_x *= (1 - (fear_val - 0.1)/0.2)
                     linear_x = -0.13
                     angular_z = 0.0

            # Update visualization with path and direction
            with self.data_lock:
                if self.visualization_image is not None:
                    # Create a copy to draw on, then update the shared image
                    img_to_draw = self.visualization_image.copy()
                    final_img = self.draw_path_and_direction(img_to_draw, waypoints, angular_z)
                    if final_img is not None:
                        self.visualization_image = final_img
                        
            # Publish Twist message
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)
            
            self.get_logger().info(f"Cmd: linear_x={linear_x:.3f}, angular_z={angular_z:.3f}, fear={fear_val:.3f}")
            
        except Exception as e:
            # correctly log the full traceback in ROS2 
            self.get_logger().error(f"Control loop error: {e}\n {traceback.format_exc()}")
            # Safety stop on error
            self.cmd_pub.publish(Twist())

    def destroy_node(self):
        """Safely shut down the node and visualization thread."""
        self.get_logger().info("Shutting down...")
        self.running = False # Signal the visualization thread to exit
        self.vis_thread.join() # Wait for the thread to finish
        self.cmd_pub.publish(Twist()) # Send a final stop command
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealSensePlannerControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
