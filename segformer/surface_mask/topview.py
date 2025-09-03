#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import math

# --- 설정 변수 ---
# 토픽 이름들
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
POINTCLOUD_TOPIC = "/camera/camera/depth/color/points"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

# 모델 경로
MODEL_PATH = "surface_mask_best_lrup.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Top view 설정
TOP_VIEW_WIDTH = 800
TOP_VIEW_HEIGHT = 600
MAX_DISTANCE = 10.0  # 최대 거리 (미터)
MIN_DISTANCE = 0.5   # 최소 거리 (미터)
PIXELS_PER_METER = 40  # 미터당 픽셀 수

# 클래스 정보
CLASS_TO_IDX = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_block': 5, 'sidewalk': 6
}
NUM_LABELS = len(CLASS_TO_IDX)

class DirectSegFormer(nn.Module):
    """학습 시 사용된 모델과 동일한 구조의 클래스"""
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        try:
            self.original_model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        except ValueError as e:
            if "torch.load" in str(e):
                print(f"Warning: {e}")
                print("Creating model architecture without pretrained weights...")
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(pretrained_model_name)
                config.num_labels = num_classes
                self.original_model = SegformerForSemanticSegmentation(config)
            else:
                raise e
        
    def forward(self, x):   
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

class SegformerTopViewNode(Node):
    def __init__(self):
        super().__init__('segformer_topview_node')
        
        self.device = DEVICE
        self.get_logger().info(f"Using device: {self.device}")

        # 모델 로드
        self.model = self.load_model()
        self.get_logger().info(f"Model loaded from '{MODEL_PATH}'")

        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 색상 팔레트
        self.color_palette = self.create_color_palette()
        self.legend_image = self.create_legend_image()

        self.bridge = CvBridge()
        
        # 카메라 내부 파라미터
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 메시지 동기화를 위한 구독자들
        self.color_sub = Subscriber(self, Image, COLOR_TOPIC)
        self.depth_sub = Subscriber(self, Image, DEPTH_TOPIC)
        
        # 카메라 정보 구독자
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10)

        # 시간 동기화
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        self.get_logger().info('SegformerTopView node started. Waiting for synchronized images...')

    def load_model(self):
        """DirectSegFormer 모델을 로드"""
        model = DirectSegFormer(num_classes=NUM_LABELS)
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info("Model loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")
            self.get_logger().warn("Using model without pretrained weights")
        
        model.to(self.device)
        model.eval()
        return model

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성"""
        cmap = plt.cm.get_cmap('jet', NUM_LABELS)
        palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
        
        for i in range(NUM_LABELS):
            if i == 0:
                palette[i] = [0, 0, 0]
                continue
            rgba = cmap(i)
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            palette[i] = bgr
        return palette

    def create_legend_image(self):
        """범례 이미지 생성"""
        legend_width = 180
        legend_height_per_class = 50
        legend_height = legend_height_per_class * NUM_LABELS
        
        legend_img = np.full((legend_height, legend_width, 3), 255, dtype=np.uint8)
        idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

        for i in range(NUM_LABELS):
            class_name = idx_to_class.get(i, 'Unknown')
            color_bgr = self.color_palette[i]
            
            y_pos = i * legend_height_per_class
            swatch_start = (10, y_pos + 5)
            swatch_end = (40, y_pos + 25)
            text_pos = (50, y_pos + 20)
            
            cv2.rectangle(legend_img, swatch_start, swatch_end, 
                         (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
            cv2.putText(legend_img, class_name, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        return legend_img

    def camera_info_callback(self, msg):
        """카메라 내부 파라미터 수신"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info("Camera parameters received")

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image)
        return input_tensor.unsqueeze(0).to(self.device)

    def depth_to_pointcloud(self, depth_image, color_image):
        """Depth 이미지를 Point Cloud로 변환"""
        if self.camera_matrix is None:
            self.get_logger().warn("Camera parameters not available")
            return None, None, None

        h, w = depth_image.shape
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 깊이 값이 0이 아닌 픽셀만 선택
        valid_mask = (depth_image > 0) & (depth_image < 65535)
        y_coords, x_coords = np.where(valid_mask)
        
        if len(x_coords) == 0:
            return None, None, None

        # Depth 값을 미터 단위로 변환 (일반적으로 mm 단위로 저장됨)
        depths = depth_image[y_coords, x_coords].astype(np.float32) / 1000.0
        
        # 3D 좌표 계산
        x_3d = (x_coords - cx) * depths / fx
        y_3d = (y_coords - cy) * depths / fy
        z_3d = depths

        # 색상 정보 추출
        colors = color_image[y_coords, x_coords]

        return x_3d, y_3d, z_3d, colors, x_coords, y_coords

    def create_top_view(self, x_3d, y_3d, z_3d, segmentation_mask, x_coords, y_coords):
        """3D 좌표를 top view로 변환"""
        # 거리 필터링
        distance_mask = (z_3d >= MIN_DISTANCE) & (z_3d <= MAX_DISTANCE)
        
        if np.sum(distance_mask) == 0:
            return np.zeros((TOP_VIEW_HEIGHT, TOP_VIEW_WIDTH, 3), dtype=np.uint8)

        # 유효한 점들만 선택
        valid_x = x_3d[distance_mask]
        valid_z = z_3d[distance_mask]
        valid_x_coords = x_coords[distance_mask]
        valid_y_coords = y_coords[distance_mask]

        # Top view 좌표 변환
        # X축: 좌우 (-MAX_DISTANCE/2 ~ MAX_DISTANCE/2)
        # Z축: 앞뒤 (MIN_DISTANCE ~ MAX_DISTANCE)
        
        top_view_x = ((valid_x + MAX_DISTANCE/2) * PIXELS_PER_METER).astype(int)
        top_view_y = ((MAX_DISTANCE - valid_z) * PIXELS_PER_METER).astype(int)

        # 이미지 경계 내의 점들만 선택
        valid_indices = (
            (top_view_x >= 0) & (top_view_x < TOP_VIEW_WIDTH) &
            (top_view_y >= 0) & (top_view_y < TOP_VIEW_HEIGHT)
        )

        if np.sum(valid_indices) == 0:
            return np.zeros((TOP_VIEW_HEIGHT, TOP_VIEW_WIDTH, 3), dtype=np.uint8)

        # Top view 이미지 생성
        top_view = np.zeros((TOP_VIEW_HEIGHT, TOP_VIEW_WIDTH, 3), dtype=np.uint8)
        
        final_x = top_view_x[valid_indices]
        final_y = top_view_y[valid_indices]
        final_img_x = valid_x_coords[valid_indices]
        final_img_y = valid_y_coords[valid_indices]

        # 세그멘테이션 결과를 top view에 매핑
        for i in range(len(final_x)):
            seg_class = segmentation_mask[final_img_y[i], final_img_x[i]]
            color = self.color_palette[seg_class]
            top_view[final_y[i], final_x[i]] = color

        return top_view

    def add_grid_and_labels(self, top_view):
        """Top view에 격자와 라벨 추가"""
        result = top_view.copy()
        
        # 격자 그리기 (1미터 간격)
        for i in range(0, TOP_VIEW_WIDTH, PIXELS_PER_METER):
            cv2.line(result, (i, 0), (i, TOP_VIEW_HEIGHT), (128, 128, 128), 1)
        
        for i in range(0, TOP_VIEW_HEIGHT, PIXELS_PER_METER):
            cv2.line(result, (0, i), (TOP_VIEW_WIDTH, i), (128, 128, 128), 1)

        # 중심선 그리기 (카메라 위치)
        center_x = TOP_VIEW_WIDTH // 2
        cv2.line(result, (center_x, 0), (center_x, TOP_VIEW_HEIGHT), (0, 255, 0), 2)
        
        # 거리 라벨 추가
        for dist in range(1, int(MAX_DISTANCE) + 1):
            y_pos = int((MAX_DISTANCE - dist) * PIXELS_PER_METER)
            if 0 <= y_pos < TOP_VIEW_HEIGHT:
                cv2.putText(result, f"{dist}m", (5, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def synchronized_callback(self, color_msg, depth_msg):
        """동기화된 컬러/깊이 이미지 처리"""
        try:
            # 이미지 변환
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # 세그멘테이션 수행
        original_h, original_w, _ = color_image.shape
        input_tensor = self.preprocess_image(color_image)

        with torch.no_grad():
            logits = self.model(input_tensor)

        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # 오버레이 생성
        segmentation_image = self.color_palette[pred_mask_np]
        overlay = cv2.addWeighted(color_image, 0.6, segmentation_image, 0.4, 0)

        # 3D 포인트 생성
        point_data = self.depth_to_pointcloud(depth_image, color_image)
        
        if point_data[0] is not None:
            x_3d, y_3d, z_3d, colors, x_coords, y_coords = point_data
            
            # Top view 생성
            top_view = self.create_top_view(x_3d, y_3d, z_3d, pred_mask_np, x_coords, y_coords)
            top_view_with_grid = self.add_grid_and_labels(top_view)
        else:
            top_view_with_grid = np.zeros((TOP_VIEW_HEIGHT, TOP_VIEW_WIDTH, 3), dtype=np.uint8)
            cv2.putText(top_view_with_grid, "No valid depth data", (50, TOP_VIEW_HEIGHT//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 결과 표시
        # 범례 크기 조정
        legend_resized = cv2.resize(self.legend_image, (150, original_h))
        
        # 이미지들을 나란히 배치
        top_row = np.hstack((color_image, overlay, legend_resized))
        
        # Top view를 원본 이미지 크기에 맞게 조정
        scale_factor = original_w / TOP_VIEW_WIDTH
        new_height = int(TOP_VIEW_HEIGHT * scale_factor)
        top_view_resized = cv2.resize(top_view_with_grid, (original_w, new_height))
        
        # Scale top view to original_w width
        scale_factor = original_w / TOP_VIEW_WIDTH
        new_height = int(TOP_VIEW_HEIGHT * scale_factor)
        top_view_resized_to_original_w = cv2.resize(top_view_with_grid, (original_w, new_height))
        
        # If the resized top view is shorter than original_h, add padding
        if new_height < original_h:
            padding = np.zeros((original_h - new_height, original_w, 3), dtype=np.uint8)
            bottom_row_first_panel = np.vstack((top_view_resized_to_original_w, padding))
        else:
            bottom_row_first_panel = cv2.resize(top_view_with_grid, (original_w, original_h)) # Ensure it's original_h tall

        # Create a blank space for the middle panel (corresponding to overlay)
        middle_blank_panel = np.zeros((original_h, original_w, 3), dtype=np.uint8)

        # Create the empty space for the legend label (150 wide, original_h tall)
        legend_label_panel = np.zeros((original_h, 150, 3), dtype=np.uint8)
        cv2.putText(legend_label_panel, "Top View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine all panels for the bottom row
        bottom_row_combined = np.hstack((bottom_row_first_panel, middle_blank_panel, legend_label_panel))
        
        # 최종 결과 조합
        final_result = np.vstack((top_row, bottom_row_combined))
        
        cv2.imshow("SegFormer Top View Visualization", final_result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SegformerTopViewNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
