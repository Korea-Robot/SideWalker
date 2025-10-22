#!/usr/bin/env python3

# ==============================================================================
# --- 🚀 Imports ---
# ==============================================================================
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import numpy as np
import math
import os
from transforms3d.quaternions import quat2mat
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import v2 as T

# ==============================================================================
# --- 🚀 Configuration ---
# ==============================================================================
MODEL_PATH = 'best_model2.pth'  # 훈련된 모델 가중치 경로
NUM_LABELS = 30
GENERIC_OBSTACLE_ID = NUM_LABELS

# ==============================================================================
# --- 🚀 Class Definitions, Labels, and Colormap ---
# ==============================================================================
# (기존 코드와 동일)
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

def create_bev_colormap(num_base_classes, generic_obstacle_id):
    """BEV 시각화를 위한 컬러맵을 생성합니다."""
    # Matplotlib의 'hsv' 컬러맵을 기반으로 색상 생성
    hues = np.linspace(0.0, 1.0, num_base_classes, endpoint=False)
    colors = [plt.cm.hsv(h) for h in hues]
    # 배경(ID 0)은 검은색으로
    colors[0] = (0, 0, 0, 1)
    # 일반 장애물(ID 30)은 흰색으로
    colors.append((1.0, 1.0, 1.0, 1.0))
    # 컬러맵 객체 생성
    cmap_mpl = ListedColormap(colors)


    # OpenCV에서 사용할 수 있도록 (B, G, R) 형태의 룩업 테이블로 변환
    # modified 31 colors => 256 size table extended 
    full_colormap_cv = np.zeros((256,3),dtype=np.uint8) # 256 size black table generate 

    # # ❗️ 수정된 부분: list를 np.array로 변환 후 슬라이싱
    # # OpenCV에서 사용할 수 있도록 (B, G, R) 형태의 룩업 테이블로 변환
    colors_np = np.array(cmap_mpl.colors) # 리스트를 NumPy 배열로 변환
    num_defined_colors = len(colors_np)
    # colormap_cv = (colors_np[:, :3] * 255).astype(np.uint8)[:, ::-1] # RGB to BGR

    # return colormap_cv

    # 정의된 색상(31개)을 복사
    defined_colors_bgr = (colors_np[:, :3] * 255).astype(np.uint8)[:, ::-1] # RGB to BGR
    full_colormap_cv[:num_defined_colors] = defined_colors_bgr

    return full_colormap_cv

# ==============================================================================
# --- 🚀 Geometric and Camera Functions ---
# ==============================================================================
# (기존 코드와 동일)
def intrinsics_from_fov(w, h, fov_h_deg, fov_v_deg):
    fov_h, fov_v = math.radians(fov_h_deg), math.radians(fov_v_deg)
    fx = w / (2.0 * math.tan(fov_h / 2.0))
    fy = h / (2.0 * math.tan(fov_v / 2.0))
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])

# RealSense D435 카메라의 일반적인 사양에 맞춘 값
RGB_INTRINSICS = intrinsics_from_fov(640, 480, 69.0, 42.0)
DEPTH_INTRINSICS = intrinsics_from_fov(640, 480, 87.0, 58.0)

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
# --- 🚀 ROS2 Node Class ---
# ==============================================================================
class RealtimeSemanticBEVNode(Node):
    def __init__(self):
        super().__init__('realtime_semantic_bev_node')
        
        # --- ROS2 Setup ---
        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        # --- Member Variables ---
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- AI Model and Preprocessing ---
        self.model = self._load_model(MODEL_PATH, NUM_LABELS).to(self.device).eval()
        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- Visualization ---
        self.bev_colormap_cv = create_bev_colormap(NUM_LABELS, GENERIC_OBSTACLE_ID)
        
        # --- Main Processing Loop ---
        self.timer = self.create_timer(0.1, self.process_and_visualize) # 10 Hz

    def _load_model(self, model_path, num_classes):
        """모델 가중치를 로드하는 함수."""
        model = SegFormer(num_classes=num_classes)
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                self.get_logger().info(f"✅ Successfully loaded model weights from '{model_path}'.")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Model loading error: {e}. Using pre-trained weights.")
        else:
            self.get_logger().warn(f"⚠️ Model file '{model_path}' not found. Using pre-trained weights.")
        return model

    def rgb_callback(self, msg):
        """RGB 이미지 토픽을 수신하는 콜백 함수."""
        try:
            # ROS Image 메시지를 OpenCV BGR 형식으로 변환
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image: {e}")

    def depth_callback(self, msg):
        """Depth 이미지 토픽을 수신하는 콜백 함수."""
        try:
            # 16UC1 형식으로 변환 (mm 단위) 후 미터 단위로 변경
            depth_mm = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.latest_depth_image = (depth_mm / 1000.0).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

    def process_and_visualize(self):
        """주기적으로 호출되어 BEV 생성 및 시각화를 수행하는 메인 함수."""
        if self.latest_rgb_image is None or self.latest_depth_image is None:
            self.get_logger().info("Waiting for RGB and Depth images...")
            return

        # --- 1. Semantic Mask 생성 ---
        # OpenCV(BGR) 이미지를 모델 입력에 맞게 RGB로 변환
        rgb_for_model = cv2.cvtColor(self.latest_rgb_image, cv2.COLOR_BGR2RGB)
        semantic_mask = self._predict_semantic_mask(rgb_for_model)
        
        # --- 2. Semantic BEV 생성 ---
        semantic_bev = self._create_semantic_bev(self.latest_depth_image, semantic_mask)

        # --- 3. 시각화 ---
        # Mask와 BEV에 컬러맵 적용
        mask_colored = cv2.applyColorMap((semantic_mask * (255 // NUM_LABELS)).astype(np.uint8), cv2.COLORMAP_JET)
        # bev_colored = cv2.LUT(cv2.cvtColor(semantic_bev, cv2.COLOR_GRAY2BGR), self.bev_colormap_cv)
        bev_colored = cv2.LUT(semantic_bev,self.bev_colormap_cv)

        # 로봇 위치 표시
        bev_h, bev_w, _ = bev_colored.shape
        cv2.drawMarker(bev_colored, (bev_w // 2, bev_h -1), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 3)

        # 결과 이미지를 하나의 창에 표시하기 위해 크기 조절 및 병합
        h, w, _ = self.latest_rgb_image.shape
        mask_resized = cv2.resize(mask_colored, (w, h))
        bev_resized = cv2.resize(bev_colored, (w, h))
        
        top_row = np.hstack((self.latest_rgb_image, mask_resized))
        # BEV를 아래쪽에 크게 표시하기 위해 빈 공간 생성
        bottom_row_placeholder = np.zeros_like(top_row)
        bev_large = cv2.resize(bev_colored, (top_row.shape[1], top_row.shape[0]))
        
        # 제목 추가
        cv2.putText(self.latest_rgb_image, "RGB Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(mask_resized, "Semantic Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bev_large, "Semantic BEV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        final_display = np.vstack((np.hstack((self.latest_rgb_image, mask_resized)), bev_large))

        cv2.imshow("Realtime Semantic BEV", final_display)
        cv2.waitKey(1)
        
    def _predict_semantic_mask(self, rgb_image_np):
        """RGB 이미지로 Semantic Mask를 추론하는 함수."""
        with torch.no_grad():
            input_tensor = self.transform(rgb_image_np).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            
            # 원본 이미지 크기로 업샘플링
            upsampled_logits = F.interpolate(
                logits, size=rgb_image_np.shape[:2], mode='bilinear', align_corners=False
            )
            predictions = torch.argmax(upsampled_logits, dim=1)
            return predictions[0].cpu().numpy()

    def _create_semantic_bev(self, depth_image, semantic_mask, bev_resolution=0.05, bev_size_m=10.0, z_min=0.2, z_max=1.5):
        """BEV 맵을 생성하는 핵심 로직."""
        # 1. Depth 이미지에서 3D 포인트 클라우드 생성 (카메라 좌표계)
        points_cam, valid_mask_depth = self._unproject_depth(depth_image, DEPTH_INTRINSICS)
        
        # 2. 유효한 3D 포인트에 해당하는 Semantic Label 찾기
        h_rgb, w_rgb = semantic_mask.shape
        # semantic_mask를 depth 이미지 크기로 리사이즈
        semantic_mask_resized = np.array(PILImage.fromarray(semantic_mask.astype(np.uint8)).resize((depth_image.shape[1], depth_image.shape[0]), PILImage.NEAREST))
        
        semantic_labels = semantic_mask_resized.flatten()[valid_mask_depth]
        valid_points_cam = points_cam[valid_mask_depth]
        
        # 3. 포인트 클라우드를 카메라 좌표계에서 로봇 베이스 좌표계로 변환
        points_robot = self._apply_transform(valid_points_cam, EXTRINSIC_HMT)

        # 4. 로봇 기준 높이(z)로 포인트 필터링 (바닥, 천장 등 제외)
        height_filter = (points_robot[:, 2] > z_min) & (points_robot[:, 2] < z_max)
        points_filtered = points_robot[height_filter]
        labels_filtered = semantic_labels[height_filter]

        # 5. 배경(ID 0)으로 분류된 포인트를 '일반 장애물'로 재지정
        labels_for_bev = np.where(labels_filtered == 0, GENERIC_OBSTACLE_ID, labels_filtered)

        # 6. 필터링된 3D 포인트를 2D BEV 그리드로 투영
        bev_pixel_size = int(bev_size_m / bev_resolution)
        bev_image = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)
        
        x_robot, y_robot = points_filtered[:, 0], points_filtered[:, 1]
        
        # 로봇 좌표(X: 전방, Y: 좌측) -> BEV 이미지 좌표(v: 아래, u: 우측)
        u_bev = (bev_pixel_size // 2 - y_robot / bev_resolution).astype(int)
        v_bev = (bev_pixel_size - 1 - x_robot / bev_resolution).astype(int)
        
        valid_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & (v_bev >= 0) & (v_bev < bev_pixel_size)
        
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

# ==============================================================================
# --- 🚀 Main Execution ---
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = RealtimeSemanticBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
