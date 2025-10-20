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
REALSENSE_TOPIC = "/argus/ar0234_front_left/image_raw"


# 2. 학습된 모델 가중치 파일 경로
MODEL_PATH = "surface_mask_best_lrup.pt"

# 3. 추론에 사용할 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 학습 스크립트에서 가져온 클래스 정보 ---
CLASS_TO_IDX = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_block': 5, 'sidewalk': 6
}
NUM_LABELS = len(CLASS_TO_IDX)

# --- 학습 스크립트에서 가져온 모델 클래스 정의 ---
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

class SegformerViewerNode(Node):
    def __init__(self):
        super().__init__('segformer_viewer_node')
        
        self.device = DEVICE
        self.get_logger().info(f"Using device: {self.device}")

        self.model = self.load_model()
        self.get_logger().info(f"Model loaded from '{MODEL_PATH}'")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.color_palette = self.create_color_palette()
        
        # --- [추가] 범례 이미지 생성 ---
        self.legend_image = self.create_legend_image()
        # -----------------------------

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
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info("Model loaded successfully with key mapping")
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")
            self.get_logger().warn("Using model without pretrained weights")
        
        model.to(self.device)
        model.eval()
        return model

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성 (OpenCV BGR 형식)"""
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

    # --- [함수 추가] 범례 이미지 생성 ---
    def create_legend_image(self):
        """클래스 색상 구분을 위한 범례 이미지를 생성합니다."""
        legend_width = 180
        legend_height_per_class = 70
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
            
            cv2.rectangle(legend_img, swatch_start, swatch_end, (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
            cv2.putText(legend_img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
        return legend_img
    # ------------------------------------

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image)
        return input_tensor.unsqueeze(0).to(self.device)

    def image_callback(self, msg):
        """이미지 메시지를 처리하고 결과를 표시하는 콜백 함수"""
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return

        original_h, original_w, _ = img_bgr.shape

        input_tensor = self.preprocess_image(img_bgr)

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

        segmentation_image = self.color_palette[pred_mask_np]
        overlay = cv2.addWeighted(img_bgr, 0.6, segmentation_image, 0.4, 0)
        
        # --- [로직 수정] 범례를 포함하여 이미지 표시 ---
        # 1. 범례 이미지의 높이를 원본 이미지 높이에 맞게 조절
        h, w, _ = img_bgr.shape
        # 가로-세로 비율을 유지하면서 높이 조절
        scale_factor = h / self.legend_image.shape[0]
        new_legend_w = int(self.legend_image.shape[1] * scale_factor)
        resized_legend = cv2.resize(self.legend_image, (new_legend_w, h))

        # 2. 원본, 오버레이, 범례 이미지를 가로로 연결
        images_to_show = np.hstack((img_bgr, overlay, resized_legend))
        # ---------------------------------------------

        cv2.imshow("DirectSegFormer Segmentation Viewer", images_to_show)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rclpy.shutdown()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    segformer_viewer_node = SegformerViewerNode()
    
    try:
        rclpy.spin(segformer_viewer_node)
    finally:
        segformer_viewer_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
