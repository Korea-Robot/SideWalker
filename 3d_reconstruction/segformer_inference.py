#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- 필요한 라이브러리들을 불러옵니다 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# [추가] Hugging Face 모델 및 프로세서
from transformers import (
    SegformerForSemanticSegmentation, 
    SegformerImageProcessor,
    MaskFormerForInstanceSegmentation,
    AutoImageProcessor
)
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# --- 📜 1. 기본 설정 변수들 ---

# 구독할 ROS 2 이미지 토픽 이름
REALSENSE_TOPIC = "/camera/camera/color/image_raw"
# REALSENSE_TOPIC = "/argus/ar0234_front_left/image_raw"
# 로컬 모델(surface, object) 추론 시 입력 크기
INFERENCE_SIZE = 512

# 🧠 2. 추론에 사용할 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔄 3. 사용할 모델 타입 선택 (중요!)
# 0: surface (로컬), 1: object (로컬), 2: ade20k (HF), 3: coco (HF)
model_list = ['surface', 'object', 'ade20k', 'coco']
MODEL_TYPE_INDEX = 1  # <--- 🚀 여기를 변경하여 모델 선택 (예: ade20k)
MODEL_TYPE = model_list[MODEL_TYPE_INDEX]

# 🗂️ 4. 선택된 모델에 따른 설정 (로컬 모델 경로)
# (ade20k, coco는 이 경로를 사용하지 않고 HF에서 직접 다운로드)

LOCAL_MODEL_PATHS = {
    'surface': "models/surface/surface_mask_best_lrup.pt",
    'object': "models/dynamic_object/best_model2.pth"
}

# 로컬 모델용 클래스 정보
LOCAL_CLASS_INFO = {
    'surface': {
        'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
        'roadway': 4, 'braille_block': 5, 'sidewalk': 6
    },
    'object': {
        'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
        'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
        'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
        'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
        'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
        'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
        'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
    }
}


# --- 🤖 5. Segformer 모델 클래스 정의 (로컬 모델 로딩용) ---
# 'surface', 'object' 모델을 로드할 때만 사용됩니다.
class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        try:
            self.original_model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        except ValueError as e:
            if "torch.load" in str(e):
                print(f"Warning: {e}")
                print("사전 학습 가중치 없이 모델 구조만 생성합니다...")
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(pretrained_model_name)
                config.num_labels = num_classes
                self.original_model = SegformerForSemanticSegmentation(config)
            else:
                raise e

    def forward(self, x):
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

# --- 🚀 6. 메인 ROS 2 노드 클래스 ---
class SegformerViewerNode(Node):
    def __init__(self):
        super().__init__('segformer_viewer_node')

        self.device = DEVICE
        self.model_type = MODEL_TYPE
        self.get_logger().info(f"사용 디바이스: {self.device} 💻")
        self.get_logger().info(f"선택된 모델 타입: {self.model_type}")

        # 🧠 [개선] 모델과 프로세서를 동적으로 로드합니다.
        # id2label_map: {0: 'class_a', 1: 'class_b', ...} 형태
        self.model, self.processor, id2label_map = self.load_model_and_processor()
        
        # 로드된 맵을 기반으로 클래스 정보 설정
        self.IDX_TO_CLASS = id2label_map
        self.CLASS_TO_IDX = {v: k for k, v in self.IDX_TO_CLASS.items()}
        self.NUM_LABELS = len(self.IDX_TO_CLASS)
        
        self.get_logger().info(f"모델 로드 완료. 총 {self.NUM_LABELS}개 클래스 감지.")

        # 🎨 클래스별 색상 팔레트를 생성합니다. (클래스 개수에 맞춰 동적 생성)
        self.color_palette = self.create_color_palette()
        
        # 🔖 범례 이미지를 미리 생성해둡니다. (클래스 개수에 맞춰 동적 생성)
        self.legend_image = self.create_legend_image()

        # 🔄 ROS 이미지와 OpenCV 이미지를 변환할 CvBridge 객체 생성
        self.bridge = CvBridge()
        
        # 📨 이미지 토픽 구독 설정
        self.subscription = self.create_subscription(
            Image,
            REALSENSE_TOPIC,
            self.image_callback,
            10)

        self.get_logger().info('Segformer 뷰어 노드가 시작되었습니다. 이미지를 기다립니다... 📸')

    def load_model_and_processor(self):
        """
        [개선된 함수]
        MODEL_TYPE에 따라 적절한 모델, 프로세서, 클래스 맵을 로드합니다.
        """
        if self.model_type in ['surface', 'object']:
            # --- 1. 로컬 모델 (surface, object) 로드 ---
            self.get_logger().info(f"로컬 모델 '{self.model_type}' 로딩 중...")
            class_map_idx_first = LOCAL_CLASS_INFO[self.model_type]
            num_classes = len(class_map_idx_first)
            
            # 1-1. 모델 로드 (DirectSegFormer 래퍼 사용)
            model = DirectSegFormer(num_classes=num_classes)
            model_path = LOCAL_MODEL_PATHS[self.model_type]
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                new_state_dict = {}
                for key, value in checkpoint.items():
                    new_key = 'original_model.' + key if key.startswith('segformer.') or key.startswith('decode_head.') else key
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
                self.get_logger().info(f"로컬 가중치 로드 성공: '{model_path}'")
            except Exception as e:
                self.get_logger().error(f"로컬 모델 로딩 실패: {e}")
                self.get_logger().warn("경고: 학습된 가중치 없이 모델을 사용합니다.")
            
            # 1-2. 프로세서 설정 (torchvision.transforms 사용)
            processor = transforms.Compose([
                transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # 1-3. 클래스 맵 반환 (id -> label 형태)
            id2label_map = {v: k for k, v in class_map_idx_first.items()}
            
            model.to(self.device).eval()
            return model, processor, id2label_map

        elif self.model_type == 'ade20k':
            # --- 2. [신규] ade20k (Segformer) 모델 로드 ---
            self.get_logger().info("Hugging Face 'ade20k' (Segformer) 모델 로딩 중...")
            model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            
            processor = SegformerImageProcessor.from_pretrained(model_name)
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
            model.to(self.device).eval()
            # model.config.id2label에 {0: 'wall', 1: 'building', ...} 정보가 들어있음
            return model, processor, model.config.id2label

        elif self.model_type == 'coco':
            # --- 3. [신규] coco (MaskFormer) 모델 로드 ---
            self.get_logger().info("Hugging Face 'coco' (MaskFormer) 모델 로딩 중...")
            # model_name = "facebook/maskformer-swin-base-coco"
            model_name = "facebook/maskformer-swin-tiny-coco"

            
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = MaskFormerForInstanceSegmentation.from_pretrained(model_name)
            
            model.to(self.device).eval()
            # model.config.id2label에 {0: 'unlabeled', 1: 'person', ...} 정보가 들어있음
            return model, processor, model.config.id2label
        
        else:
            raise ValueError(f"알 수 없는 MODEL_TYPE입니다: {self.model_type}")

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성 (OpenCV BGR 형식)"""
        # 'jet' 컬러맵은 색상 구분이 잘 됩니다.
        cmap = plt.cm.get_cmap('jet', self.NUM_LABELS)
        palette = np.zeros((self.NUM_LABELS, 3), dtype=np.uint8)

        for i in range(self.NUM_LABELS):
            # 0번 클래스(배경/unlabeled)는 검은색으로 고정
            if i == 0: 
                palette[i] = [0, 0, 0]
                continue
            
            rgba = cmap(i)
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            palette[i] = bgr
        
        self.get_logger().info(f"총 {self.NUM_LABELS}개의 클래스용 컬러 팔레트 생성 완료.")
        return palette

    def create_legend_image(self):
        """[동적] 클래스 개수에 맞춰 범례 이미지를 생성합니다."""
        legend_width = 250  # 클래스 이름이 길 수 있으므로 폭을 넓힘
        legend_height_per_class = 20 # 높이를 줄여 더 많은 클래스 표시
        legend_height = legend_height_per_class * self.NUM_LABELS
        
        # 클래스가 너무 많으면(예: 150개) 최대 높이 제한
        max_height = 1080 # (FHD 높이)
        if legend_height > max_height:
            legend_height = max_height
            legend_height_per_class = legend_height / self.NUM_LABELS

        legend_img = np.full((legend_height, legend_width, 3), 255, dtype=np.uint8)

        for i in range(self.NUM_LABELS):
            # 범례가 이미지를 초과하면 중단
            y_pos = int(i * legend_height_per_class)
            if y_pos > legend_height - legend_height_per_class:
                break

            class_name = self.IDX_TO_CLASS.get(i, 'Unknown')
            color_bgr = self.color_palette[i]
            
            swatch_start = (10, y_pos + 2)
            swatch_end = (30, y_pos + int(legend_height_per_class * 0.8))
            text_pos = (35, y_pos + int(legend_height_per_class * 0.7))

            cv2.rectangle(legend_img, swatch_start, swatch_end, 
                          (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
            
            cv2.putText(legend_img, f"{i}: {class_name}", text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
        return legend_img

    def preprocess_image(self, cv_image):
        """[개선] 모델 타입에 맞는 프로세서로 이미지를 변환"""
        
        # 1. OpenCV(BGR) -> PIL(RGB) 이미지로 변환 (모든 프로세서가 선호)
        rgb_image_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_np)

        if self.model_type in ['surface', 'object']:
            # 2-1. 로컬 모델 (torchvision.transforms)
            input_tensor = self.processor(pil_image)
            # [C, H, W] -> [1, C, H, W] 배치 차원 추가 및 디바이스 전송
            return input_tensor.unsqueeze(0).to(self.device)
        
        else:
            # 2-2. HF 모델 (ImageProcessor)
            # processor가 텐서 변환, 정규화, 배치 차원 추가까지 모두 처리
            inputs = self.processor(images=pil_image, return_tensors="pt")
            # 딕셔너리 형태의 입력을 디바이스로 전송
            return inputs.to(self.device)

    def image_callback(self, msg):
        """[핵심] 이미지 수신, 추론, 후처리, 시각화"""
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"ROS 이미지 변환 실패: {e}")
            return

        original_h, original_w, _ = img_bgr.shape

        # 1. 전처리 (모델 타입에 맞게 자동 수행)
        inputs = self.preprocess_image(img_bgr)

        # 2. [추론 및 후처리] (모델 타입별 분기)
        # 최종 결과물: segmentation_image (HxWx3 BGR 컬러맵)
        segmentation_image = np.zeros((original_h, original_w, 3), dtype=np.uint8)

        with torch.no_grad():
            if self.model_type in ['surface', 'object', 'ade20k']:
                # --- [Semantic Segmentation] ---
                
                # 'surface', 'object'는 텐서 입력
                # 'ade20k'는 딕셔너리 입력
                if self.model_type == 'ade20k':
                    outputs = self.model(**inputs)
                else: 
                    outputs = self.model(inputs) # inputs = 텐서
                
                # 공통: logits 추출
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # 업샘플링 (원본 이미지 크기로 복원)
                upsampled_logits = F.interpolate(
                    logits,
                    size=(original_h, original_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # 가장 점수가 높은 클래스 ID(0~N)를 픽셀별로 선택
                pred_mask_np = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                # [시각화] ID 맵 -> 컬러 BGR 이미지로 변환
                segmentation_image = self.color_palette[pred_mask_np]

            elif self.model_type == 'coco':
                # --- [Panoptic Segmentation (MaskFormer)] ---
                
                # 1. 추론 (딕셔너리 입력)
                outputs = self.model(**inputs)
                
                # 2. [중요] Panoptic 후처리 (프로세서 사용)
                # target_sizes를 원본 크기로 지정해야 함
                result = self.processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[(original_h, original_w)]
                )[0] # 0번 = 첫 번째(유일한) 이미지 결과
                
                # (H, W) 크기의 텐서. 각 픽셀은 '인스턴스 ID'를 가짐
                pred_mask_np = result["segmentation"].cpu().numpy()
                # 인스턴스 ID별 정보 리스트 (예: {id: 1, label_id: 15, ...})
                segment_info = result["segments_info"]

                # 3. [시각화] 인스턴스 ID 맵 -> 클래스 컬러 맵으로 변환
                # (배경은 어차피 0(검은색)이므로 객체들만 순회하며 색칠)
                for info in segment_info:
                    instance_id = info['id']
                    class_id = info['label_id'] # 0=unlabeled, 1=person ...
                    
                    # 팔레트에서 해당 '클래스'의 색상 조회
                    color = self.color_palette[class_id % self.NUM_LABELS]
                    
                    # ID 맵에서 이 인스턴스 ID에 해당하는 픽셀들만 골라 색칠
                    segmentation_image[pred_mask_np == instance_id] = color

        # 3. [시각화] (모든 모델 공통)
        # 원본 이미지(60%) + 세그멘테이션(40%)
        overlay = cv2.addWeighted(img_bgr, 0.6, segmentation_image, 0.4, 0)
        
        # 범례 이미지 높이 맞추기
        h, w, _ = img_bgr.shape
        # 가로-세로 비율 유지하며 리사이즈 (FHD 등 세로가 긴 범례도 처리)
        scale_factor = h / self.legend_image.shape[0] 
        new_legend_w = int(self.legend_image.shape[1] * scale_factor)
        
        # 이미지가 너무 작으면(scale_factor < 0) 에러 날 수 있으므로 최소 1 픽셀 보장
        if new_legend_w <= 0: new_legend_w = 1
        if h <= 0: h = 1
            
        resized_legend = cv2.resize(self.legend_image, (new_legend_w, h), interpolation=cv2.INTER_AREA)

        # [원본], [오버레이], [범례] 가로로 연결
        images_to_show = np.hstack((img_bgr, overlay, resized_legend))

        # 화면에 표시
        cv2.imshow("Multi-Model Segmentation Viewer", images_to_show)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' 키
            self.get_logger().info("Esc 키 입력 감지. 노드를 종료합니다.")
            rclpy.shutdown()
            cv2.destroyAllWindows()

# --- 🏁 7. 메인 실행 함수 ---
def main(args=None):
    rclpy.init(args=args)
    segformer_viewer_node = SegformerViewerNode()
    try:
        rclpy.spin(segformer_viewer_node)
    finally:
        segformer_viewer_node.get_logger().info("노드 정리 및 종료 중...")
        segformer_viewer_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
