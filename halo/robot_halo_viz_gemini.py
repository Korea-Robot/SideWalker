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
import pandas as pd
import torchvision.transforms as transforms
import os
import threading
import time
import math
import traceback
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# --- Import your model classes ---
# Make sure these files (reward_estimation_model.py, etc.) are in the same directory
# or your Python path.
from reward_estimation_model import HALORewardModel

# ==============================================================================
# --- Bresenham's line algorithm (no skimage dependency) ---
# ==============================================================================
def draw_line_bresenham(x0, y0, x1, y1):
    """
    Calculates pixel coordinates between two points using Bresenham's line algorithm.
    Returns row (rr) and column (cc) coordinates for numpy array indexing.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    rr, cc = [], []
    
    while True:
        rr.append(x0)
        cc.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
            
    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)

class HaloRewardControlNode(Node):
    def __init__(self):
        super().__init__('halo_reward_control_node')

        # === Model & Control Configuration ===
        self.MODEL_PATH = './gamma-0.5_lr_1e-05/best_model.pth'
        self.NUM_CANDIDATE_ACTIONS = 17
        self.FIXED_LINEAR_V = 0.4  # Max linear velocity
        self.IMG_SIZE_MASK = 32

        # === ROS2 Setup ===
        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # === State & Waypoint Variables ===
        self.current_pose = None
        self.current_rgb_tensor = None
        self.waypoints = [(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        self.waypoint_index = 0
        self.goal_threshold = 0.5

        # === Model & Candidate Masks Setup ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_reward_model()
        self.candidate_masks, self.angular_velocities = self.generate_candidate_masks()

        # === Visualization Setup ===
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_lock = threading.Lock()
        self.visualization_image = None
        self.latest_rewards = np.zeros(self.NUM_CANDIDATE_ACTIONS)
        self.best_action_idx = 0
        self.running = True
        self.vis_thread.start()
        
        self.get_logger().info(f"✅ HALO Reward Control Node has started on {self.device}.")

    def setup_reward_model(self):
        """Loads the HALORewardModel and sets up image transformations."""
        self.model = HALORewardModel(freeze_dino=True).to(self.device)
        if not os.path.exists(self.MODEL_PATH):
            self.get_logger().error(f"Model checkpoint not found at {self.MODEL_PATH}")
            raise FileNotFoundError(f"Model checkpoint not found at {self.MODEL_PATH}")
        
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.get_logger().info("HALORewardModel loaded successfully.")

    def image_callback(self, msg: Image):
        """Processes incoming RGB images."""
        try:
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
            
            with self.vis_lock:
                # For model inference
                self.current_rgb_tensor = self.transform(cv_image_rgb).unsqueeze(0).to(self.device)
                # For visualization
                self.visualization_image = cv_image_rgb.copy()

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def odom_callback(self, msg: Odometry):
        """Updates the robot's current pose."""
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def control_callback(self):
        """Main control loop for model inference and command publishing."""
        if self.current_rgb_tensor is None or self.current_pose is None:
            return

        # --- Waypoint Navigation Logic ---
        if self.waypoint_index >= len(self.waypoints):
            self.publish_stop_command()
            return
            
        target_wp = self.waypoints[self.waypoint_index]
        current_x, current_y, _ = self.current_pose
        distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)

        if distance_to_goal < self.goal_threshold:
            self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("✅ All waypoints reached. Stopping.")
                self.publish_stop_command()
                return

        # --- Model Inference and Action Selection ---
        try:
            with torch.no_grad():
                rgb_expanded = self.current_rgb_tensor.repeat(self.NUM_CANDIDATE_ACTIONS, 1, 1, 1)
                predicted_rewards, _ = self.model(rgb_expanded, self.candidate_masks)
                rewards = predicted_rewards.squeeze().cpu().numpy()

            best_action_idx = np.argmax(rewards)
            chosen_angular_z = self.angular_velocities[best_action_idx]
            
            # Slow down when turning
            chosen_linear_x = self.FIXED_LINEAR_V / (1.0 + 0.8 * abs(chosen_angular_z))

            with self.vis_lock:
                self.latest_rewards = rewards
                self.best_action_idx = best_action_idx

            # --- Publish Command ---
            twist = Twist()
            twist.linear.x = float(chosen_linear_x)
            twist.angular.z = float(chosen_angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(f"WP[{self.waypoint_index}] | Best Action Idx: {best_action_idx}, Reward: {rewards[best_action_idx]:.3f} -> v:{chosen_linear_x:.2f}, w:{chosen_angular_z:.2f}")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.publish_stop_command()

    def publish_stop_command(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _visualization_thread(self):
        """Displays the RGB image with reward-colored candidate trajectories."""
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.vis_lock:
                if self.visualization_image is None:
                    time.sleep(0.1)
                    continue
                display_image = self.visualization_image.copy()
                rewards = self.latest_rewards.copy()
                best_idx = self.best_action_idx
            
            final_image = self.draw_trajectories_on_image(display_image, rewards, best_idx)
            
            cv2.imshow("HALO Reward Vision", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(30) == 27: # ESC key
                self.running = False
                break
        
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")
    
    def draw_trajectories_on_image(self, image, rewards, best_idx):
        """Draws all candidate trajectories on the image, colored by reward."""
        h, w, _ = image.shape
        
        # Normalize rewards for color mapping
        norm = Normalize(vmin=np.min(rewards), vmax=np.max(rewards))
        cmap = cm.get_cmap('viridis') # Colormap from red (low) to green/yellow (high)

        # Draw each candidate trajectory
        for i, w_val in enumerate(self.angular_velocities):
            # Regenerate points for this trajectory to draw on the large image
            mask_points = self._generate_trajectory_points(w_val, h, w)
            
            color_rgba = cmap(norm(rewards[i]))
            color_bgr = tuple(int(c * 255) for c in color_rgba[:3])
            
            thickness = 4 if i == best_idx else 2
            if i == best_idx:
                color_bgr = (255, 0, 0) # Highlight best trajectory in Blue

            for j in range(len(mask_points) - 1):
                p1 = tuple(mask_points[j])
                p2 = tuple(mask_points[j+1])
                cv2.line(image, p1, p2, color_bgr, thickness)
        
        # Add text info
        text = f"Best Action: w={self.angular_velocities[best_idx]:.2f}, reward={rewards[best_idx]:.3f}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return image

    def _generate_trajectory_points(self, angular_vel, img_h, img_w):
        """Helper to generate pixel coordinates for a single trajectory."""
        duration = 1.0; hz = 10; num_points = int(duration * hz)
        timestamps = np.arange(num_points) * (1000 / hz)
        linear_v = self.FIXED_LINEAR_V / (1 + 0.5 * abs(angular_vel))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'manual_linear_x': [linear_v] * num_points,
            'manual_angular_z': [angular_vel] * num_points,
        })
        
        delta_t = df['timestamp'].diff().fillna(0) / 1000.0
        x, y, theta = 0.0, 0.0, 0.0
        odom_list = []
        for i in range(1, len(df)):
            v = df['manual_linear_x'].iloc[i]; w = df['manual_angular_z'].iloc[i]; dt = delta_t.iloc[i]
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            odom_list.append([x, y])
        
        if not odom_list: return []

        odom_segment = np.array(odom_list)
        
        # --- Projection onto image plane ---
        # Scale trajectory to fit nicely in the bottom half of the image
        # X (forward) is mapped to image v (vertical), Y (left) is mapped to image u (horizontal)
        max_dist_x = 1.5 # meters
        
        # Invert v to have the path go "up" from the bottom
        v_coords = img_h - 1 - (odom_segment[:, 0] / max_dist_x * (img_h * 0.75))
        
        # Center u at the middle of the image
        u_coords = (img_w / 2) - (odom_segment[:, 1] / max_dist_x * (img_w * 1.5))
        
        points = np.vstack((u_coords, v_coords)).T.astype(np.int32)
        
        # Add the starting point (robot's position at the bottom center)
        start_point = np.array([[img_w // 2, img_h - 1]], dtype=np.int32)
        return np.vstack((start_point, points))

    # =====================================================================
    # --- Candidate Mask Generation (from your inference script) ---
    # =====================================================================
    def generate_candidate_masks(self):
        """Generates candidate trajectory masks for the model."""
        angular_velocities = np.linspace(-1.0, 1.0, self.NUM_CANDIDATE_ACTIONS)
        candidate_masks = []
        
        for w in angular_velocities:
            duration = 2.0; hz = 10; num_points = int(duration * hz)
            timestamps = np.arange(num_points) * (1000 / hz)
            linear_v = self.FIXED_LINEAR_V / (1 + 0.5 * abs(w))
            
            dummy_df = pd.DataFrame({
                'timestamp': timestamps,
                'manual_linear_x': [linear_v] * num_points,
                'manual_angular_z': [w] * num_points,
            })
            
            mask_np = self.generate_trajectory_mask_from_df(dummy_df, img_size=self.IMG_SIZE_MASK)
            candidate_masks.append(torch.from_numpy(mask_np).float())
            
        self.get_logger().info(f"Generated {len(candidate_masks)} candidate masks.")
        return torch.stack(candidate_masks).unsqueeze(1).to(self.device), angular_velocities

    def generate_trajectory_mask_from_df(self, df, img_size):
        if len(df) < 2:
            return np.zeros((img_size, img_size), dtype=np.uint8)

        delta_t = df['timestamp'].diff().fillna(0) / 1000.0
        x, y, theta = 0.0, 0.0, 0.0
        odom_list = [[x, y]]
        for i in range(1, len(df)):
            v = df['manual_linear_x'].iloc[i]
            w = df['manual_angular_z'].iloc[i]
            dt = delta_t.iloc[i]
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            odom_list.append([x, y])
        
        ego_coords = np.array(odom_list)

        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        max_range = 1.0
        
        u = np.clip(((ego_coords[:, 0] / max_range) * (img_size - 1)).astype(int), 0, img_size - 1)
        v = np.clip((ego_coords[:, 1] / (max_range / 1.3) * (img_size - 1) / 2 + (img_size / 2)).astype(int), 0, img_size - 1)

        for i in range(len(u) - 1):
            rr, cc = draw_line_bresenham(u[i], v[i], u[i+1], v[i+1])
            mask[rr, cc] = 1
        
        mask[0, img_size // 2] = 1
        return mask

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.publish_stop_command()
        self.vis_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HaloRewardControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
