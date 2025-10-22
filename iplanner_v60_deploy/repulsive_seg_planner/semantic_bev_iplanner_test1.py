#!/usr/bin/env python3

# ==============================================================================
# --- 🚀 Combined Imports ---
# ==============================================================================
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
from PIL import Image as PILImage # ✅ NEW: For image conversion
import os
import json
import threading
import time
import math
import traceback

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

# ✅ NEW: Imports from BEV Generation Script
from transforms3d.quaternions import quat2mat
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from matplotlib.colors import ListedColormap

print('✅ BEV Repulsive Force Planner Started!')

# ==============================================================================
# --- 🚀 Geometric, BEV, Model Functions (From First Script) ---
# ==============================================================================
# ✅ NEW: All helper functions from the BEV script are placed here.
def intrinsics_from_fov(width, height, fov_h_deg, fov_v_deg):
    fov_h, fov_v = math.radians(fov_h_deg), math.radians(fov_v_deg)
    fx = width / (2.0 * math.tan(fov_h / 2.0)); fy = height / (2.0 * math.tan(fov_v / 2.0))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

def calculate_crop_box(rgb_intrinsics, rgb_dims, depth_fov_h_deg, depth_fov_v_deg):
    fx_rgb, fy_rgb = rgb_intrinsics[0, 0], rgb_intrinsics[1, 1]; w_rgb, h_rgb = rgb_dims
    fov_h_depth_rad, fov_v_depth_rad = math.radians(depth_fov_h_deg), math.radians(depth_fov_v_deg)
    new_w = 2 * fx_rgb * math.tan(fov_h_depth_rad / 2.0); new_h = 2 * fy_rgb * math.tan(fov_v_depth_rad / 2.0)
    left, top = (w_rgb - new_w) / 2, (h_rgb - new_h) / 2
    return int(top), int(left), int(new_h), int(new_w)

def create_hmt(translation, quaternion):
    rot_matrix = quat2mat([quaternion[3], *quaternion[:3]]); hmt = np.eye(4); hmt[:3, :3] = rot_matrix; hmt[:3, 3] = translation; return hmt

def unproject_depth_to_pointcloud(depth_map, camera_k):
    fx, fy, cx, cy = camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2]; h, w = depth_map.shape; u, v = np.meshgrid(np.arange(w), np.arange(h)); valid_mask = depth_map > 0; z = np.where(valid_mask, depth_map, 0); x = (u - cx) * z / fx; y = (v - cy) * z / fy; return np.stack((x, y, z), axis=-1).reshape(-1, 3), valid_mask.flatten()

def apply_transform_to_pointcloud(points, hmt):
    ones = np.ones((points.shape[0], 1)); transformed_points = np.hstack((points, ones)) @ hmt.T; return transformed_points[:, :3]

def project_points_to_image_plane(points_3d, camera_k_rgb):
    fx, fy, cx, cy = camera_k_rgb[0, 0], camera_k_rgb[1, 1], camera_k_rgb[0, 2], camera_k_rgb[1, 2]; x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]; valid_z_mask = z > 1e-6; u, v = np.zeros_like(z), np.zeros_like(z); u[valid_z_mask] = (fx * x[valid_z_mask] / z[valid_z_mask]) + cx; v[valid_z_mask] = (fy * y[valid_z_mask] / z[valid_z_mask]) + cy; return u, v, valid_z_mask

def create_semantic_bev_from_pointcloud(depth_image, semantic_mask, depth_intrinsics, rgb_intrinsics_cropped, extrinsics_hmt, generic_obstacle_id, bev_resolution=0.05, bev_size_m=10.0, z_min=-1.0, z_max=1.0):
    points_cam_frame, depth_valid_mask = unproject_depth_to_pointcloud(depth_image, depth_intrinsics)
    rgb_h, rgb_w = semantic_mask.shape
    u_rgb, v_rgb, proj_valid_mask = project_points_to_image_plane(points_cam_frame, rgb_intrinsics_cropped)
    combined_mask = depth_valid_mask & proj_valid_mask
    u_rgb_int, v_rgb_int = u_rgb.astype(int), v_rgb.astype(int)
    bounds_mask = (u_rgb_int >= 0) & (u_rgb_int < rgb_w) & (v_rgb_int >= 0) & (v_rgb_int < rgb_h)
    final_mask = combined_mask & bounds_mask
    valid_points_cam_frame = points_cam_frame[final_mask]
    valid_u_rgb, valid_v_rgb = u_rgb_int[final_mask], v_rgb_int[final_mask]
    semantic_labels = semantic_mask[valid_v_rgb, valid_u_rgb]
    points_robot_frame = apply_transform_to_pointcloud(valid_points_cam_frame, extrinsics_hmt)
    height_filter = (points_robot_frame[:, 2] > z_min) & (points_robot_frame[:, 2] < z_max)
    points_filtered, labels_filtered = points_robot_frame[height_filter], semantic_labels[height_filter]
    labels_for_bev = np.where(labels_filtered == 0, 0, generic_obstacle_id) # Treat all non-background as generic obstacles
    bev_pixel_size = int(bev_size_m / bev_resolution)
    bev_image = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)
    x_robot, y_robot = points_filtered[:, 0], points_filtered[:, 1]
    u_bev = (bev_pixel_size // 2 - y_robot / bev_resolution).astype(int)
    v_bev = (bev_pixel_size - 1 - x_robot / bev_resolution).astype(int)
    valid_bev_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & (v_bev >= 0) & (v_bev < bev_pixel_size)
    u_bev_valid, v_bev_valid = u_bev[valid_bev_indices], v_bev[valid_bev_indices]
    labels_valid = labels_for_bev[valid_bev_indices]
    bev_image[v_bev_valid, u_bev_valid] = labels_valid
    return bev_image

id2label = {0: 'background', 1: 'barricade', 2: 'bench', 3: 'bicycle', 4: 'bollard', 5: 'bus', 6: 'car', 7: 'carrier', 8: 'cat', 9: 'chair', 10: 'dog', 11: 'fire_hydrant', 12: 'kiosk', 13: 'motorcycle', 14: 'movable_signage', 15: 'parking_meter', 16: 'person', 17: 'pole', 18: 'potted_plant', 19: 'power_controller', 20: 'scooter', 21: 'stop', 22: 'stroller', 23: 'table', 24: 'traffic_light', 25: 'traffic_light_controller', 26: 'traffic_sign', 27: 'tree_trunk', 28: 'truck', 29: 'wheelchair'}

class SegFormer(nn.Module):
    def __init__(self, num_classes=30): super().__init__(); self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_classes, id2label=id2label, label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True, torch_dtype=torch.float32, use_safetensors=True)
    def forward(self, x): return self.model(pixel_values=x).logits

def load_seg_model(model_path, device, num_classes):
    model = SegFormer(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {'model.' + k if not k.startswith('model.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=False); model.to(device); model.eval(); return model


# ==============================================================================
# --- 🚀 Main ROS2 Node Class ---
# ==============================================================================
class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10) # ✅ NEW
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)

        # Waypoints and State
        self.current_pose = None
        self.waypoints = [(0.0, 0.0),(5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)] # Example square
        self.waypoint_index = 0
        self.goal_threshold = 0.6
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # Data Holders
        self.current_depth_tensor = None
        self.current_depth_numpy = None # ✅ NEW: For BEV
        self.current_rgb_image = None # ✅ NEW: For BEV

        # Setup Models and Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_planner()
        self.setup_bev_generator() # ✅ NEW

        # Visualization
        self.visualization_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # Plotting Data
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_preds = np.array([])
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])
        self.latest_bev = np.zeros((200, 200), dtype=np.uint8) # ✅ NEW: For visualization

        self.get_logger().info("✅ RealSense PlannerNet with BEV Repulsion has started.")

    def setup_planner(self):
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"✅ PlannerNet model loaded successfully on {self.device}")

    def setup_bev_generator(self): # ✅ NEW
        self.bev_params = {}
        # --- BEV Constants ---
        self.bev_params['NUM_LABELS'] = 30
        self.bev_params['GENERIC_OBSTACLE_ID'] = self.bev_params['NUM_LABELS']
        self.bev_params['RGB_WIDTH_ORIG'], self.bev_params['RGB_HEIGHT_ORIG'] = 640, 480 # Match RealSense
        
        # --- Camera Intrinsics and Extrinsics ---
        # NOTE: These values should be calibrated for your specific robot setup.
        # These are typical values for D435 at 640x480.
        self.bev_params['RGB_INTRINSICS_ORIG'] = np.array([[616.0, 0.0, 320.0], [0.0, 616.0, 240.0], [0.0, 0.0, 1.0]])
        self.bev_params['DEPTH_INTRINSICS'] = np.array([[384.0, 0.0, 320.0], [0.0, 384.0, 240.0], [0.0, 0.0, 1.0]])
        
        # This HMT describes the depth camera's pose relative to the robot's base_link frame.
        # T = [-0.015, 0.22, 0.05] (x,y,z), Q = [0.49, -0.51, 0.5, -0.5] (x,y,z,w)
        self.bev_params['EXTRINSIC_HMT'] = create_hmt([-0.015, 0.22, 0.05], [0.49, -0.51, 0.5, -0.5])

        # --- Crop Box Calculation ---
        # Assuming RGB FoV is wider than Depth FoV. We crop RGB to match Depth FoV.
        DEPTH_FOV_H_DEG, DEPTH_FOV_V_DEG = 87.0, 58.0
        self.bev_params['crop_top'], self.bev_params['crop_left'], self.bev_params['crop_h'], self.bev_params['crop_w'] = calculate_crop_box(
            self.bev_params['RGB_INTRINSICS_ORIG'], (self.bev_params['RGB_WIDTH_ORIG'], self.bev_params['RGB_HEIGHT_ORIG']), DEPTH_FOV_H_DEG, DEPTH_FOV_V_DEG
        )
        self.bev_params['RGB_INTRINSICS_CROPPED'] = intrinsics_from_fov(
            self.bev_params['crop_w'], self.bev_params['crop_h'], DEPTH_FOV_H_DEG, DEPTH_FOV_V_DEG
        )

        # --- Semantic Model & Transforms ---
        MODEL_PATH = 'best_model2.pth'
        self.seg_model = load_seg_model(MODEL_PATH, self.device, self.bev_params['NUM_LABELS'])
        self.bev_params['inference_transforms'] = transforms.Compose([
            transforms.Resize((self.bev_params['RGB_HEIGHT_ORIG'], self.bev_params['RGB_WIDTH_ORIG'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- Repulsive Force Parameters ---
        self.bev_params['bev_resolution'] = 0.05  # 5 cm per pixel
        self.bev_params['bev_pixel_size'] = 200   # 10m x 10m map (200 * 0.05)
        self.repulsion_strength = 0.8  # How strongly points are pushed
        self.influence_radius = 8      # Radius in pixels (8 * 5cm = 40cm) to check for obstacles

        self.get_logger().info("✅ SegFormer model and BEV parameters initialized.")

    def rgb_callback(self, msg): # ✅ NEW
        try:
            # ROS Image -> CV2 BGR
            self.current_rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB processing error: {e}")

    def depth_callback(self, msg): # MODIFIED
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # --- For BEV Generation (meters, float32) ---
            # Using 1000.0 as the scaling factor for 16-bit depth
            self.current_depth_numpy = (depth_cv.astype(np.float32) / 1000.0)

            # --- For PlannerNet (normalized, 0-10m) ---
            max_depth_value = 10.0
            depth_clipped = np.clip(self.current_depth_numpy, 0, max_depth_value)
            depth_normalized_planner = depth_clipped / max_depth_value
            
            # For visualization
            depth_display_normalized = (depth_normalized_planner * 255).astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_display_normalized, cv2.COLORMAP_JET)

            # Create tensor for PlannerNet
            depth_tensor = torch.from_numpy(depth_normalized_planner).unsqueeze(0)
            depth_tensor = depth_tensor.repeat(3, 1, 1)
            depth_tensor = TF.resize(depth_tensor, [360, 640])
            self.current_depth_tensor = depth_tensor.unsqueeze(0).to(self.device)

            with self.plot_data_lock:
                self.visualization_image = depth_display
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def control_callback(self):
        # MODIFIED: Added check for RGB image
        if self.current_depth_tensor is None or self.current_pose is None or self.current_rgb_image is None or self.current_depth_numpy is None:
            return

        try:
            # (Waypoint following logic is the same as before)
            if self.waypoint_index >= len(self.waypoints):
                twist = Twist(); twist.linear.x = 0.0; twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return
            # ... (rest of waypoint logic) ...
            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    return
            target_wp = self.waypoints[self.waypoint_index]
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            

            # ==========================================================
            # --- ✅ NEW: BEV Generation & Repulsive Force Pipeline ---
            # ==========================================================
            
            ## --- Step 1: Generate Semantic BEV ---
            # Prepare RGB image for SegFormer
            rgb_pil = PILImage.fromarray(cv2.cvtColor(self.current_rgb_image, cv2.COLOR_BGR2RGB))
            rgb_tensor_inf = self.bev_params['inference_transforms'](rgb_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get semantic mask
                logits = self.seg_model(rgb_tensor_inf)
                upsampled_logits = F.interpolate(logits, size=(self.bev_params['RGB_HEIGHT_ORIG'], self.bev_params['RGB_WIDTH_ORIG']), mode='bilinear', align_corners=False)
                full_pred_mask = torch.argmax(upsampled_logits, dim=1)[0].cpu().numpy()
                
                # Crop mask to match depth FoV
                ct, cl, ch, cw = self.bev_params['crop_top'], self.bev_params['crop_left'], self.bev_params['crop_h'], self.bev_params['crop_w']
                pred_mask_cropped = full_pred_mask[ct : ct + ch, cl : cl + cw]

                # Create the BEV map
                semantic_bev = create_semantic_bev_from_pointcloud(
                    depth_image=self.current_depth_numpy,
                    semantic_mask=pred_mask_cropped,
                    depth_intrinsics=self.bev_params['DEPTH_INTRINSICS'],
                    rgb_intrinsics_cropped=self.bev_params['RGB_INTRINSICS_CROPPED'],
                    extrinsics_hmt=self.bev_params['EXTRINSIC_HMT'],
                    generic_obstacle_id=self.bev_params['GENERIC_OBSTACLE_ID'],
                    bev_resolution=self.bev_params['bev_resolution'],
                    bev_size_m=self.bev_params['bev_pixel_size'] * self.bev_params['bev_resolution']
                )

            ## --- Step 2: Get Initial PlannerNet Prediction ---
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)

            ## --- Step 3: Apply Repulsive Force from BEV ---
            preds_tensor_repuled = self.apply_repulsive_force(preds_tensor, semantic_bev)
            
            # ==========================================================
            # --- End of New Pipeline ---
            # ==========================================================

            # MODIFIED: Use the repulsed tensor for control
            with torch.no_grad():
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor_repuled, step=0.1)
                cmd_vels = preds_tensor_repuled[:,:,:2]
                fear_val = fear.cpu().item()

                # (Control logic is the same, but now uses modified waypoints)
                k = 2; h = 3
                angular_z = torch.clamp(cmd_vels[0, k:k+h, 1], -1.0, 1.0).mean().cpu().item()
                angular_z = self._discretize_value(angular_z, 0.2)
                linear_x = 0.4
                if abs(angular_z) >= 0.4: linear_x = 0.1 # Slow down on sharp turns
                
                # Update visualization data with final tensors
                with self.plot_data_lock:
                    self.latest_preds = preds_tensor_repuled.squeeze().cpu().numpy() # Show repulsed preds
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                    self.latest_bev = semantic_bev # Store for visualization
                
                # Publish commands
                twist = Twist()
                twist.linear.x = float(linear_x)
                twist.angular.z = float(angular_z)
                self.cmd_pub.publish(twist)

                self.get_logger().info(f"WP[{self.waypoint_index}] | CMD: lx={linear_x:.2f}, az={angular_z:.2f} | Fear:{fear_val:.2f}")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            
    def apply_repulsive_force(self, preds_tensor, semantic_bev): # ✅ NEW
        """
        Modifies predicted waypoints to steer them away from obstacles in the BEV map.
        """
        # Clone the tensor to avoid modifying the original
        preds_repulsed = preds_tensor.clone().squeeze(0) # Shape: [5, 3]
        
        # Get locations of all obstacles in the BEV map (pixels)
        # Note: In BEV, rows (v) correspond to x_robot, cols (u) to y_robot
        obstacle_pixels = np.argwhere(semantic_bev > 0)
        if obstacle_pixels.size == 0:
            return preds_tensor # No obstacles, no change

        # BEV parameters
        res = self.bev_params['bev_resolution']
        size = self.bev_params['bev_pixel_size']

        for i in range(preds_repulsed.shape[0]):
            point_robot = preds_repulsed[i, :2] # x, y in meters
            x_r, y_r = point_robot[0].item(), point_robot[1].item()
            
            # Convert robot coords (m) to BEV pixel coords
            u_p = int(size // 2 - y_r / res)
            v_p = int(size - 1 - x_r / res)
            
            # Check if point is within BEV map bounds
            if not (0 <= u_p < size and 0 <= v_p < size):
                continue
            
            # Find nearby obstacles
            dist_sq = np.sum((obstacle_pixels - np.array([v_p, u_p]))**2, axis=1)
            nearby_mask = dist_sq < self.influence_radius**2
            
            if not np.any(nearby_mask):
                continue # No obstacles within influence radius
            
            nearby_obstacles = obstacle_pixels[nearby_mask]
            
            # Calculate average repulsive vector in pixel space
            total_force_vec = np.zeros(2, dtype=np.float32)
            for obs_pixel in nearby_obstacles:
                # Vector from obstacle to the point
                vec = np.array([v_p, u_p]) - obs_pixel
                dist = np.linalg.norm(vec)
                if dist < 1e-5: continue # Avoid division by zero
                
                # Force is stronger when closer (1/dist)
                force_magnitude = self.repulsion_strength * (1.0 / dist)
                force_vec = (vec / dist) * force_magnitude
                total_force_vec += force_vec

            # Apply the summed force to the point's pixel coordinates
            v_p_new = v_p + total_force_vec[0]
            u_p_new = u_p + total_force_vec[1]
            
            # Convert new pixel coords back to robot coords (m)
            x_r_new = (size - 1 - v_p_new) * res
            y_r_new = (size // 2 - u_p_new) * res
            
            # Update the tensor with the new, repulsed coordinates
            preds_repulsed[i, 0] = x_r_new
            preds_repulsed[i, 1] = y_r_new

        return preds_repulsed.unsqueeze(0)


    # (The rest of the class methods: _discretize_value, _visualization_thread, quaternion_to_yaw, odom_callback, etc. remain the same)
    def _discretize_value(self,value,step):
        return round(value/step)*step
    def _visualization_thread(self):
        self.get_logger().info("Starting CV2 visualization thread.")
        bev_window_name = "Semantic BEV"
        planner_window_name = "PlannerNet Vision"
        cv2.namedWindow(bev_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(planner_window_name, cv2.WINDOW_NORMAL)

        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
                bev_image = self.latest_bev.copy()

            if display_image is not None:
                cv2.imshow(planner_window_name, display_image)
            
            if bev_image is not None:
                # Make BEV more visible
                bev_display = (bev_image > 0).astype(np.uint8) * 255
                bev_display_color = cv2.cvtColor(bev_display, cv2.COLOR_GRAY2BGR)
                
                # Draw robot position
                robot_pos_x = bev_display_color.shape[1] // 2
                robot_pos_y = bev_display_color.shape[0] - 1
                cv2.drawMarker(bev_display_color, (robot_pos_x, robot_pos_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
                
                cv2.imshow(bev_window_name, bev_display_color)

            key = cv2.waitKey(30)
            if key == 27: # ESC
                self.running = False

        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y); cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z); return math.atan2(siny_cosp, cosy_cosp)
    def odom_callback(self, msg: Odometry):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        with self.plot_data_lock: self.current_pose = [x, y, yaw]; self.trajectory_data.append([x, y])
    def destroy_node(self):
        self.get_logger().info("Shutting down..."); self.running = False; self.vis_thread.join(); super().destroy_node()


# (Matplotlib and main execution functions remain the same)
def update_plot(frame, node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    with node.plot_data_lock: traj = list(node.trajectory_data); pose = node.current_pose; preds_local = node.latest_preds.copy(); waypoints_local = node.latest_waypoints.copy(); goal_local = node.latest_local_goal.copy(); all_wps = np.array(node.waypoints); wp_idx = node.waypoint_index
    if not traj: return []
    reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:];
    if reached_wps.size > 0: reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else: reached_wps_plot.set_data([], [])
    if pending_wps.size > 0: pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else: pending_wps_plot.set_data([], [])
    traj_arr = np.array(traj); traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0]); current_x, current_y, current_yaw = pose; current_point.set_data([-current_y], [current_x]); heading_len = 0.5; heading_end_x = current_x + heading_len * math.cos(current_yaw); heading_end_y = current_y + heading_len * math.sin(current_yaw); heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])
    if preds_local.size > 0 and waypoints_local.size > 0 and goal_local.size > 0:
        rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)], [math.sin(current_yaw),  math.cos(current_yaw)]])
        waypoints_global = (rot_matrix @ waypoints_local[:, :2].T).T + np.array([current_x, current_y]); preds_global = (rot_matrix @ preds_local[:, :2].T).T + np.array([current_x, current_y]); goal_global = rot_matrix @ goal_local + np.array([current_x, current_y]); waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0]); preds_points.set_data(-preds_global[:, 1], preds_global[:, 0]); goal_point.set_data([-goal_global[1]], [goal_global[0]])
    return [traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot]

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    fig, ax = plt.subplots(figsize=(10, 10),constrained_layout=True)
    ax.set_title('Real-time Trajectory and PlannerNet Prediction'); ax.set_xlabel('-Y Position (m)'); ax.set_ylabel('X Position (m)'); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    wps_array = np.array(node.waypoints); x_min, y_min = wps_array.min(axis=0) - 1.5; x_max, y_max = wps_array.max(axis=0) + 1.5; ax.set_ylim(x_min, x_max); ax.set_xlim(-y_max, -y_min)
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory'); current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position'); heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading'); preds_points, = ax.plot([], [], 'ro', markersize=5, label='Preds (Model Output)'); waypoints_line, = ax.plot([], [], 'y.-', lw=1, label='Waypoints (Path)'); goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal'); reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints'); pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints'); ax.legend()
    ani = FuncAnimation(fig, update_plot, fargs=(node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot), interval=100, blit=True)
    try: plt.show()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); ros_thread.join()

if __name__ == '__main__':
    main()
