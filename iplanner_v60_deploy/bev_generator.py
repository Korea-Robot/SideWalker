# bev_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import numpy as np
import math
from transforms3d.quaternions import quat2mat
import os
from PIL import Image

# ==============================================================================
# --- 🚀 Configuration & Constants (from your first script) ---
# ==============================================================================
NUM_LABELS = 30
GENERIC_OBSTACLE_ID = NUM_LABELS

class_to_idx = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
    'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
    'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
    'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
    'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
    'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}
id2label = {idx: label for label, idx in class_to_idx.items()}

# Camera parameters
def intrinsics_from_fov(w, h, fov_h_deg, fov_v_deg):
    fov_h, fov_v = math.radians(fov_h_deg), math.radians(fov_v_deg)
    fx = w / (2.0 * math.tan(fov_h / 2.0))
    fy = h / (2.0 * math.tan(fov_v / 2.0))
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])

RGB_INTRINSICS = intrinsics_from_fov(224, 224, 90.0, 65.0)
DEPTH_INTRINSICS = intrinsics_from_fov(640, 480, 87.0, 58.0) # Assuming RealSense D435 default FOV

def create_hmt(translation, quaternion):
    rot_matrix = quat2mat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    hmt = np.eye(4); hmt[:3, :3] = rot_matrix; hmt[:3, 3] = translation
    return hmt

EXTRINSIC_HMT = create_hmt([-0.015, 0.22, 0.05], [0.49, -0.51, 0.5, -0.5])


# ==============================================================================
# --- 🚀 SegFormer Model Definition ---
# ==============================================================================
class SegFormer(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=num_classes, id2label=id2label,
            label2id={l: i for i, l in id2label.items()},
            ignore_mismatched_sizes=True, torch_dtype=torch.float32,use_safetensors=True
        )
    def forward(self, x):
        return self.model(pixel_values=x).logits


# ==============================================================================
# --- 🚀 Main BEV Generator Class ---
# ==============================================================================
class SemanticBEVGenerator:
    """
    Encapsulates the entire process of generating a semantic BEV map
    from an RGB image and a depth map.
    """
    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path, NUM_LABELS).to(self.device).eval()
        from torchvision.transforms import v2 as T
        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, model_path, num_classes):
        model = SegFormer(num_classes=num_classes)
        if os.path.exists(model_path):
            try:
                # Handle both state_dict and full model saves
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                # Adjust keys if they are prefixed (e.g., 'model.')
                new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Successfully loaded semantic model weights from '{model_path}'.")
            except Exception as e:
                print(f"⚠️ Model weight loading error: {e}. Using pre-trained weights.")
        else:
            print(f"⚠️ Warning: Model file '{model_path}' not found. Using pre-trained weights.")
        return model

    def generate_bev(self, rgb_image_np, depth_image_np, bev_resolution=0.05, bev_size_m=10.0):
        """
        Main function to generate the semantic BEV.
        Args:
            rgb_image_np (np.ndarray): RGB image (H, W, 3).
            depth_image_np (np.ndarray): Depth map in meters (H, W).
        Returns:
            np.ndarray: The generated semantic BEV map.
        """
        # 1. Get semantic mask from RGB image
        semantic_mask = self._predict_semantic_mask(rgb_image_np)

        # 2. Create BEV from depth and semantic mask
        semantic_bev = self._create_semantic_bev_from_pointcloud(
            depth_image=depth_image_np,
            semantic_mask=semantic_mask,
            generic_obstacle_id=GENERIC_OBSTACLE_ID,
            bev_resolution=bev_resolution,
            bev_size_m=bev_size_m
        )
        return semantic_bev

    def _predict_semantic_mask(self, rgb_image_np):
        """Performs inference to get the semantic mask."""
        with torch.no_grad():
            # The model expects a batch, so we add a dimension
            input_tensor = self.transform(rgb_image_np).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            
            # Upsample logits to original image size for better alignment with depth
            upsampled_logits = F.interpolate(
                logits, size=rgb_image_np.shape[:2], mode='bilinear', align_corners=False
            )
            predictions = torch.argmax(upsampled_logits, dim=1)
            return predictions[0].cpu().numpy()

    # --- Geometric and Camera Functions (from your script) ---
    def _create_semantic_bev_from_pointcloud(self, depth_image, semantic_mask, generic_obstacle_id,
                                             bev_resolution=0.05, bev_size_m=10.0, z_min=0.2, z_max=1.5):
        # 1. Unproject depth to 3D point cloud
        points_cam, valid_mask_depth = self._unproject_depth(depth_image, DEPTH_INTRINSICS)
        
        # 2. Project points onto the RGB image plane to find their semantic label
        # Note: This step assumes RGB and Depth cameras are roughly aligned.
        # For perfect alignment, use the extrinsic calibration between cameras.
        h_rgb, w_rgb = semantic_mask.shape
        # Resize semantic mask to depth image size for direct mapping
        semantic_mask_resized = np.array(Image.fromarray(semantic_mask.astype(np.uint8)).resize((depth_image.shape[1], depth_image.shape[0]), Image.NEAREST))
        semantic_labels = semantic_mask_resized.flatten()[valid_mask_depth]
        valid_points_cam = points_cam[valid_mask_depth]
        
        # 3. Transform points from camera frame to robot's base frame
        points_robot = self._apply_transform(valid_points_cam, EXTRINSIC_HMT)

        # 4. Filter points by height (z-axis in robot frame)
        height_filter = (points_robot[:, 2] > z_min) & (points_robot[:, 2] < z_max)
        points_filtered = points_robot[height_filter]
        labels_filtered = semantic_labels[height_filter]

        # 5. Re-label background points as generic obstacles
        labels_for_bev = np.where(labels_filtered == 0, generic_obstacle_id, labels_filtered)

        # 6. Project filtered 3D points onto the 2D BEV grid
        bev_pixel_size = int(bev_size_m / bev_resolution)
        bev_image = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)
        
        x_robot, y_robot = points_filtered[:, 0], points_filtered[:, 1]
        
        # Convert robot coordinates (X forward, Y left) to BEV image coordinates (v down, u right)
        u_bev = (bev_pixel_size // 2 - y_robot / bev_resolution).astype(int)
        v_bev = (bev_pixel_size - 1 - x_robot / bev_resolution).astype(int)
        
        valid_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & (v_bev >= 0) & (v_bev < bev_pixel_size)
        
        # Assign labels to the BEV map
        bev_image[v_bev[valid_indices], u_bev[valid_indices]] = labels_for_bev[valid_indices]
        
        return bev_image

    def _unproject_depth(self, depth_map, K):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        valid = depth_map > 0
        z = np.where(valid, depth_map, 0)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_3d = np.stack((x, y, z), axis=-1)
        return points_3d.reshape(-1, 3), valid.flatten()

    def _apply_transform(self, points, hmt):
        homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed = homo_points @ hmt.T
        return transformed[:, :3]
