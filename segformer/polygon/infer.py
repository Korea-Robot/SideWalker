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

# --- 설정 변수 ---
# 1. Realsense 카메라 이미지 토픽 이름
REALSENSE_TOPIC = "/camera/camera/color/image_raw"
# REALSENSE_TOPIC = "/argus/ar0234_front_left/image_raw"


# 2. 학습된 모델 가중치 파일 경로
MODEL_PATH = "best_model2.pth"

# MODEL_PATH = "best_seg_model.pth" # best accuracy feel

# MODEL_PATH = "seg_model_epoch_100.pth"


# best_model2.pth  best_model.pth  best_seg_model.pth seg_model_epoch_100.pth
# 3. 추론에 사용할 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 변경 사항 1: 추론에 사용할 이미지 크기 설정 ---
INFERENCE_SIZE = (512, 512) # (너비, 높이)

# --- 학습 스크립트에서 가져온 클래스 정보 ---
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

# --- 학습 스크립트에서 가져온 모델 클래스 정의 ---
class DirectSegFormer(nn.Module):
    """학습 시 사용된 모델과 동일한 구조의 클래스"""
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=30):
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
                # PyTorch 버전 문제 우회: 로컬에서 모델 생성 후 가중치만 로드
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

class SegformerViewerNode(Node):
    def __init__(self):
        super().__init__('segformer_viewer_node')
        
        # 1) Set the device
        self.device = DEVICE
        self.get_logger().info(f"Using device: {self.device}")

        # 2) Load DirectSegFormer model from checkpoint
        self.model = self.load_model()
        self.get_logger().info(f"Model loaded from '{MODEL_PATH}'")

        # 3) Image preprocessing pipeline (학습 스크립트와 동일한 정규화 사용)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 4) Create a color palette for visualization
        self.color_palette = self.create_color_palette()

        # ROS 2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            REALSENSE_TOPIC,
            self.image_callback,
            10)
        
        self.get_logger().info('DirectSegFormer viewer node has been started. Waiting for images...')

    def load_model(self):
        """DirectSegFormer 모델을 로드하고 state_dict를 적용"""
        model = DirectSegFormer(num_classes=NUM_LABELS)
        
        try:
            # 체크포인트 로드 (weights_only=False로 시도)
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

            
            # 키 이름 매핑 처리
            new_state_dict = {}
            for key, value in checkpoint.items():
                # 체크포인트의 키가 "segformer."로 시작하면 "original_model." 접두사 추가
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # 매핑된 state_dict로 모델 로드
            model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info("Model loaded successfully with key mapping")
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")
            self.get_logger().warn("Using model without pretrained weights")
        
        model.to(self.device)
        model.eval()  # 추론 모드로 설정
        return model

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성 (OpenCV BGR 형식)"""
        # Matplotlib의 'jet' 컬러맵 사용
        cmap = plt.cm.get_cmap('jet', NUM_LABELS)
        palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
        
        for i in range(NUM_LABELS):
            if i == 0:  # 배경은 검은색
                palette[i] = [0, 0, 0]
                continue
            # RGBA (0-1) -> BGR (0-255) 변환
            rgba = cmap(i)
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            palette[i] = bgr
        return palette

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 텐서로 변환 및 정규화
        input_tensor = self.transform(rgb_image)
        
        # 배치 차원 추가 및 디바이스로 전송
        return input_tensor.unsqueeze(0).to(self.device)

    def image_callback(self, msg):
        """Callback function for processing image messages and displaying the result."""
        try:
            # Convert ROS Image message to an OpenCV image
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return

        original_h, original_w, _ = img_bgr.shape

        # --- 변경 사항 2: 이미지 리사이즈 ---
        # 모델 입력 전에 이미지를 학습 크기(512x512)로 리사이즈
        resized_img = cv2.resize(img_bgr, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 1) 리사이즈된 이미지 전처리
        input_tensor = self.preprocess_image(resized_img)

        # 2) 추론 수행
        with torch.no_grad():
            # DirectSegFormer는 logits를 직접 반환
            logits = self.model(input_tensor)

        # 3) 결과 후처리
        # --- 변경 사항 3: 원본 크기로 업샘플링 ---
        # 로짓을 원본 이미지 크기로 업샘플링하여 오버레이
        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w), # 여기서 원본 크기 사용
            mode='bilinear',
            align_corners=False
        )
        
        # 가장 확률이 높은 클래스로 예측 마스크 생성
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # 4) 시각화
        # 세그멘테이션 마스크에 색상 입히기
        segmentation_image = self.color_palette[pred_mask_np]

        # 원본 이미지와 마스크를 오버레이
        overlay = cv2.addWeighted(img_bgr, 0.6, segmentation_image, 0.4, 0)
        
        # Stack the original and segmented images side-by-side for comparison
        images_to_show = np.hstack((img_bgr, overlay))

        # 5) Display the images
        cv2.imshow("DirectSegFormer Segmentation Viewer", images_to_show)
        
        # Check for key press to close the window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            rclpy.shutdown()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    segformer_viewer_node = SegformerViewerNode()
    
    # Use a try-finally block to ensure cv2 windows are closed
    try:
        rclpy.spin(segformer_viewer_node)
    finally:
        # Cleanup
        segformer_viewer_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
