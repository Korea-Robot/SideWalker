#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from transforms3d.quaternions import quat2mat
from PIL import Image as PILImage

# ==============================================================================
# --- ⚙️ 설정 변수 (Configuration) ---
# ==============================================================================
# 1. ROS 토픽 이름
RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"

# 2. 모델 및 추론 관련 설정
MODEL_PATH = "best_model2.pth"  # 사용하던 'best_model2.pth' 또는 가장 성능이 좋았던 모델
INFERENCE_SIZE = (512, 512)     # 학습 시 사용했던 이미지 크기 (너비, 높이)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. BEV(Bird's-Eye-View) 맵 설정
BEV_RESOLUTION = 0.05  # BEV 맵의 해상도 (미터/픽셀)
BEV_SIZE_M = 10.0      # BEV 맵의 크기 (정방형, 미터)
Z_MIN_ROBOT = 0.15     # 로봇 좌표계 기준, 장애물로 인식할 최소 높이 (미터)
Z_MAX_ROBOT = 1.8      # 로봇 좌표계 기준, 장애물로 인식할 최대 높이 (미터)

# 4. 카메라 파라미터 (RealSense D435 기준, 필요시 수정)
# 참고: 이 값들은 'ros2 topic echo /camera/camera/camera_info' 등으로 확인하는 것이 가장 정확합니다.
CAMERA_PARAMS = {
    'rgb': {'w': 640, 'h': 480, 'fx': 615.3, 'fy': 615.3, 'cx': 320.0, 'cy': 240.0},
    'depth': {'w': 640, 'h': 480, 'fx': 386.0, 'fy': 386.0, 'cx': 321.4, 'cy': 241.2}
}

# 5. 카메라의 물리적 위치 (로봇 베이스 좌표계 기준)
# [x, y, z] (m) / [x, y, z, w]
CAM_TRANSLATION = [0.2, 0.0, 0.5]  # 예시: 로봇 중심에서 전방 20cm, 높이 50cm
CAM_QUATERNION = [0.5, -0.5, 0.5, 0.5] # 예시: 아래를 45도 바라보는 방향

# ==============================================================================
# --- 🎨 클래스 정보 및 컬러맵 (Labels & Colormaps) ---
# ==============================================================================
CLASS_TO_IDX = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
    'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
    'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
    'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
    'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
    'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}
NUM_LABELS = len(CLASS_TO_IDX)

# ==============================================================================
# --- 🤖 모델 클래스 정의 (Model Definition) ---
# ==============================================================================
# 첫 번째 코드에서 잘 작동했던 모델 클래스를 그대로 사용합니다.
class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=30):
        super().__init__()
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
    def forward(self, x):   
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

# ==============================================================================
# --- 📐 ROS2 노드 클래스 (ROS2 Node Class) ---
# ==============================================================================
class SemanticBEVNode(Node):
    def __init__(self):
        super().__init__('semantic_bev_node')
        
        # --- 변수 초기화 ---
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.device = DEVICE
        self.bridge = CvBridge()
        self.get_logger().info(f"Using device: {self.device}")

        # --- 모델 및 전처리기 로딩 (첫 번째 코드 기반) ---
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- 컬러맵 생성 ---
        self.color_palette_2d = self.create_color_palette_2d() # 2D 마스크용
        self.color_palette_bev = self.create_color_palette_bev() # BEV 맵용

        # --- 카메라 파라미터 및 변환 행렬 계산 ---
        self.depth_intrinsics = np.array([
            [CAMERA_PARAMS['depth']['fx'], 0, CAMERA_PARAMS['depth']['cx']],
            [0, CAMERA_PARAMS['depth']['fy'], CAMERA_PARAMS['depth']['cy']],
            [0, 0, 1]
        ])
        self.extrinsic_hmt = self.create_hmt(CAM_TRANSLATION, CAM_QUATERNION)

        # --- ROS2 구독 및 타이머 설정 ---
        self.rgb_sub = self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.timer = self.create_timer(0.1, self.process_and_visualize) # 10Hz
        self.get_logger().info('Semantic BEV node started. Waiting for images...')

    # --- 모델 로딩 함수 (첫 번째 코드와 동일) ---
    def load_model(self):
        model = DirectSegFormer(num_classes=NUM_LABELS)
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            new_state_dict = {}
            # state_dict 키 이름이 달라 발생하는 오류를 해결하기 위한 로직
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info(f"Model loaded successfully from '{MODEL_PATH}'")
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}. Using pre-trained weights.")
        model.to(self.device)
        model.eval()
        return model

    # --- 컬러맵 생성 함수들 ---
    def create_color_palette_2d(self):
        cmap = plt.cm.get_cmap('jet', NUM_LABELS)
        palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
        for i in range(NUM_LABELS):
            if i == 0: palette[i] = [0, 0, 0]
            else:
                rgba = cmap(i)
                palette[i] = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        return palette

    def create_color_palette_bev(self):
        cmap = plt.cm.get_cmap('hsv', NUM_LABELS)
        palette = np.zeros((NUM_LABELS + 1, 3), dtype=np.uint8) # +1 for generic obstacle
        for i in range(NUM_LABELS):
            if i == 0: palette[i] = [0, 0, 0] # Background: Black
            else:
                rgba = cmap(i / NUM_LABELS)
                palette[i] = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        return palette

    # --- 이미지 콜백 함수들 ---
    def rgb_callback(self, msg):
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        try:
            depth_mm = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.latest_depth_image = depth_mm.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    # --- 메인 처리 및 시각화 함수 ---
    def process_and_visualize(self):
        if self.latest_rgb_image is None or self.latest_depth_image is None:
            return

        img_bgr = self.latest_rgb_image.copy()
        depth_image = self.latest_depth_image.copy()
        original_h, original_w, _ = img_bgr.shape

        # 1. 이미지 전처리 및 Semantic Mask 추론
        resized_img = cv2.resize(img_bgr, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb_for_model = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_for_model).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
        
        upsampled_logits = F.interpolate(logits, size=(original_h, original_w), mode='bilinear', align_corners=False)
        semantic_mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # 2. Semantic BEV 맵 생성
        bev_map = self.create_semantic_bev(depth_image, semantic_mask)

        # 3. 시각화
        # 2D Semantic Mask 시각화
        mask_colored = self.color_palette_2d[semantic_mask]
        overlay_2d = cv2.addWeighted(img_bgr, 0.6, mask_colored, 0.4, 0)

        # BEV 맵 시각화
        bev_colored = self.color_palette_bev[bev_map]
        
        # 로봇 위치 표시 (중앙 하단)
        bev_h, bev_w, _ = bev_colored.shape
        cv2.drawMarker(bev_colored, (bev_w // 2, bev_h -1), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 3)

        # 결과 이미지 병합 및 표시
        # 2D 오버레이 이미지 크기를 BEV 맵 크기에 맞춤 (500x500)
        display_size = (bev_h, bev_w)
        overlay_resized = cv2.resize(overlay_2d, display_size)
        
        # 좌: 2D 오버레이, 우: BEV 맵
        final_display = np.hstack((overlay_resized, bev_colored))
        
        cv2.imshow("Semantic BEV", final_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            self.destroy_node()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # --- BEV 생성 관련 함수들 ---
    def create_semantic_bev(self, depth_image, semantic_mask):
        # 1. 3D 포인트 클라우드 생성 (카메라 좌표계)
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        valid = depth_image > 0
        z_cam = np.where(valid, depth_image, 0)
        x_cam = (u - self.depth_intrinsics[0, 2]) * z_cam / self.depth_intrinsics[0, 0]
        y_cam = (v - self.depth_intrinsics[1, 2]) * z_cam / self.depth_intrinsics[1, 1]
        
        points_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
        
        # 2. Semantic Mask를 Depth 이미지 크기에 맞게 리사이즈 (최근접 이웃 보간)
        mask_resized = cv2.resize(semantic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        labels = mask_resized.flatten()
        
        # 3. 유효한 포인트만 필터링 (Depth 값이 있고, 배경이 아닌 포인트)
        # ✨ 핵심 수정: 배경(ID=0)인 포인트는 BEV 맵에 그리지 않음
        valid_indices = (valid.flatten()) & (labels != 0)
        
        points_cam_valid = points_cam[valid_indices]
        labels_valid = labels[valid_indices]
        
        # 4. 카메라 좌표계 -> 로봇 베이스 좌표계로 변환
        homo_points = np.hstack((points_cam_valid, np.ones((points_cam_valid.shape[0], 1))))
        points_robot = (self.extrinsic_hmt @ homo_points.T).T[:, :3]

        # 5. 로봇 기준 높이 필터링 (바닥, 천장 등 제외)
        height_filter = (points_robot[:, 2] > Z_MIN_ROBOT) & (points_robot[:, 2] < Z_MAX_ROBOT)
        points_filtered = points_robot[height_filter]
        labels_filtered = labels_valid[height_filter]

        # 6. 3D 포인트를 2D BEV 그리드로 투영
        bev_pixel_size = int(BEV_SIZE_M / BEV_RESOLUTION)
        bev_map = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)
        
        x_robot, y_robot = points_filtered[:, 0], points_filtered[:, 1]
        
        # 로봇 좌표(X: 전방, Y: 좌측) -> BEV 이미지 좌표(v: 아래, u: 우측)
        u_bev = (bev_pixel_size // 2 - y_robot / BEV_RESOLUTION).astype(int)
        v_bev = (bev_pixel_size - 1 - x_robot / BEV_RESOLUTION).astype(int)
        
        # BEV 맵 경계 내에 있는 유효한 인덱스만 사용
        valid_bev_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & \
                            (v_bev >= 0) & (v_bev < bev_pixel_size)
        
        bev_map[v_bev[valid_bev_indices], u_bev[valid_bev_indices]] = labels_filtered[valid_bev_indices]
        
        return bev_map
        
    def create_hmt(self, translation, quaternion):
        # 동차 변환 행렬 (Homogeneous Transformation Matrix) 생성
        rot_matrix = quat2mat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]]) # w, x, y, z
        hmt = np.eye(4)
        hmt[:3, :3] = rot_matrix
        hmt[:3, 3] = translation
        return hmt

# ==============================================================================
# --- 🏁 메인 실행 함수 (Main Execution) ---
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SemanticBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
