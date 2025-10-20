#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber

# --- 설정 변수 ---
# 토픽 이름들
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

# 모델 경로
MODEL_PATH = "surface_mask_best_lrup.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 깊이 시각화 설정
MAX_DEPTH = 5.0  # 최대 깊이 (미터)
MIN_DEPTH = 0.3  # 최소 깊이 (미터)

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

class DepthSemanticVisualizerNode(Node):
    def __init__(self):
        super().__init__('depth_semantic_visualizer_node')
        
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
        
        self.get_logger().info('Depth Semantic Visualizer node started. Waiting for synchronized images...')

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
        legend_width = 200
        legend_height_per_class = 40
        legend_height = legend_height_per_class * NUM_LABELS + 50  # 추가 공간
        
        legend_img = np.full((legend_height, legend_width, 3), 255, dtype=np.uint8)
        idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

        # 제목 추가
        cv2.putText(legend_img, "Semantic Classes", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for i in range(NUM_LABELS):
            class_name = idx_to_class.get(i, 'Unknown')
            color_bgr = self.color_palette[i]
            
            y_pos = i * legend_height_per_class + 50
            swatch_start = (10, y_pos)
            swatch_end = (35, y_pos + 20)
            text_pos = (45, y_pos + 15)
            
            cv2.rectangle(legend_img, swatch_start, swatch_end, 
                         (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
            cv2.putText(legend_img, class_name, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
        return legend_img

    def camera_info_callback(self, msg):
        """카메라 내부 파라미터 수신"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.get_logger().info("Camera parameters received")

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image)
        return input_tensor.unsqueeze(0).to(self.device)

    def create_depth_semantic_costmap(self, depth_image, semantic_mask):
        """깊이 정보와 시맨틱 정보를 결합한 costmap 생성"""
        h, w = depth_image.shape
        costmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 깊이를 미터 단위로 변환 (mm -> m)
        depth_m = depth_image.astype(np.float32) / 1000.0
        
        # 유효한 깊이 범위 마스크
        valid_depth_mask = (depth_m > MIN_DEPTH) & (depth_m < MAX_DEPTH) & (depth_m > 0)
        
        # 깊이를 0-1 범위로 정규화 (가까운 곳이 밝게)
        normalized_depth = np.zeros_like(depth_m)
        normalized_depth[valid_depth_mask] = 1.0 - (
            (depth_m[valid_depth_mask] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        )
        normalized_depth = np.clip(normalized_depth, 0, 1)
        
        # 시맨틱 색상과 깊이 정보 결합
        for i in range(h):
            for j in range(w):
                if valid_depth_mask[i, j]:
                    # 시맨틱 클래스 색상 가져오기
                    semantic_color = self.color_palette[semantic_mask[i, j]]
                    
                    # 깊이에 따른 밝기 조절 (가까울수록 밝게)
                    depth_factor = normalized_depth[i, j]
                    
                    # 색상과 깊이 정보 결합
                    costmap[i, j] = (semantic_color * depth_factor).astype(np.uint8)
                else:
                    # 유효하지 않은 깊이는 검정색
                    costmap[i, j] = [0, 0, 0]
        
        return costmap

    def create_depth_colormap(self, depth_image):
        """깊이 이미지를 컬러맵으로 변환"""
        # 깊이를 미터 단위로 변환
        depth_m = depth_image.astype(np.float32) / 1000.0
        
        # 유효한 깊이 범위로 제한
        depth_clipped = np.clip(depth_m, MIN_DEPTH, MAX_DEPTH)
        
        # 0-255 범위로 정규화
        depth_normalized = ((depth_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255).astype(np.uint8)
        
        # 유효하지 않은 픽셀은 0으로 설정
        valid_mask = (depth_m > MIN_DEPTH) & (depth_m < MAX_DEPTH) & (depth_m > 0)
        depth_normalized[~valid_mask] = 0
        
        # 컬러맵 적용
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colormap[~valid_mask] = [0, 0, 0]  # 무효한 픽셀은 검정색
        
        return depth_colormap

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

        # 세그멘테이션 오버레이 생성
        segmentation_image = self.color_palette[pred_mask_np]
        semantic_overlay = cv2.addWeighted(color_image, 0.6, segmentation_image, 0.4, 0)

        # 깊이 semantic costmap 생성
        depth_semantic_costmap = self.create_depth_semantic_costmap(depth_image, pred_mask_np)
        
        # 깊이 컬러맵 생성 (참고용)
        depth_colormap = self.create_depth_colormap(depth_image)
        # Resize depth_colormap to match the width of legend_resized
        depth_colormap_resized = cv2.resize(depth_colormap, (200, original_h))

        # 범례 크기 조정
        legend_resized = cv2.resize(self.legend_image, (200, original_h))
        
        # 상단: Semantic Overlay + Legend
        top_row = np.hstack((semantic_overlay, legend_resized))
        
        # 하단: Depth Semantic Costmap + Depth Colormap
        bottom_row = np.hstack((depth_semantic_costmap, depth_colormap_resized))
        
        # 라벨 추가를 위한 패딩
        label_height = 30
        top_label = np.full((label_height, top_row.shape[1], 3), 50, dtype=np.uint8)
        bottom_label = np.full((label_height, bottom_row.shape[1], 3), 50, dtype=np.uint8)
        
        # 라벨 텍스트 추가
        cv2.putText(top_label, "Semantic Segmentation", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bottom_label, "Depth-Semantic Costmap", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bottom_label, "Depth Colormap", (depth_semantic_costmap.shape[1] + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 최종 결과 조합
        final_result = np.vstack((
            top_label,
            top_row,
            bottom_label,
            bottom_row
        ))
        
        cv2.imshow("Depth Semantic Visualization", final_result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DepthSemanticVisualizerNode()
    
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
