#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast # Half Precision을 위해 import
import numpy as np
import cv2
from PIL import Image as PILImage
import time
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict
from tqdm import tqdm
import os
import logging

# transformers 라이브러리에서 필요한 모듈 임포트
from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerConfig
)

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 1. 의존성 코드 (Config)
# (이전과 동일)
# ============================================================================
OBJECT_CLASSES = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3,
    'bollard': 4, 'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8,
    'chair': 9, 'dog': 10, 'fire_hydrant': 11, 'kiosk': 12,
    'motorcycle': 13, 'movable_signage': 14, 'parking_meter': 15,
    'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21,
    'stroller': 22, 'table': 23, 'traffic_light': 24,
    'traffic_light_controller': 25, 'traffic_sign': 26,
    'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}
SURFACE_CLASSES = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_block': 5, 'sidewalk': 6
}
CITYSCAPES_CLASSES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3,
    "fence": 4, "pole": 5, "traffic light": 6, "traffic sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10, "person": 11,
    "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18
}

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters (이 스크립트에서는 사용되지 않음)"""
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class ReconstructionConfig:
    """Main configuration for segmentation"""
    use_semantic: bool = True
    model_type: str ="maskformer-cityscapes"
    
    # --- 중요 ---
    # 사용자 정의 모델의 경로를 실제 파일 위치로 수정해야 합니다.
    custom_object_model_path: str = "models/dynamic_object/best_model2.pth"
    custom_surface_model_path: str = "models/surface/surface_mask_best_lrup.pt"

    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    
    active_model_name: str = field(init=False)
    inference_size: int = field(init=False)
    custom_class_names: Dict[str, int] = field(init=False)
    
    def __post_init__(self):
        # (참고) custom-object의 inference_size는 더 이상 전처리에 사용되지 않습니다.
        if self.model_type == "custom-object":
            self.active_model_name = self.custom_object_model_path
            self.custom_class_names = OBJECT_CLASSES.copy()
            self.inference_size = 512 
        elif self.model_type == "custom-surface":
            self.active_model_name = self.custom_surface_model_path
            self.custom_class_names = SURFACE_CLASSES.copy()
            self.inference_size = 512
        elif self.model_type == "segformer-cityscapes":
            self.active_model_name = self.segformer_checkpoint
            self.custom_class_names = CITYSCAPES_CLASSES.copy()
            self.inference_size = 512
        elif self.model_type == "maskformer-cityscapes":
            self.active_model_name = self.maskformer_checkpoint
            self.custom_class_names = CITYSCAPES_CLASSES.copy()
            self.inference_size = 384
        else:
            raise ValueError(f"알 수 없는 model_type입니다: {self.model_type}")

    @property
    def num_custom_classes(self) -> int:
        return len(self.custom_class_names)
    
    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.custom_class_names.items()}

# ============================================================================
# 2. 의존성 코드 (Model)
# ============================================================================

# --- ⬇️ 수정된 부분 1: CustomSegFormer 클래스 수정 ⬇️ ---
class CustomSegFormer(nn.Module):
    """Custom trained SegFormer model (ROS 코드의 DirectSegFormer와 동일 구조)"""
    def __init__(self, num_classes: int = 30, pretrained_name: str = "nvidia/mit-b0"):
        super().__init__()
        try:
            # 속성 이름을 'original_model'로 변경
            self.original_model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes, 
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        except (ValueError, OSError) as e:
            if "torch.load" in str(e) or "is not a local folder and is not a valid model identifier" in str(e):
                logger.warning(f"Warning: {e}")
                logger.warning("Creating model architecture without pretrained weights...")
                config = SegformerConfig.from_pretrained(pretrained_name)
                config.num_labels = num_classes
                # 속성 이름을 'original_model'로 변경
                self.original_model = SegformerForSemanticSegmentation(config)
            else:
                raise e
                
    def forward(self, x):
        # 'original_model'을 사용하여 forward
        outputs = self.original_model(pixel_values=x)
        return outputs.logits
# --- ⬆️ 수정된 부분 1 ⬆️ ---


class SemanticModel:
    """Unified interface for different semantic segmentation models"""
    def __init__(self, config, device, logger_instance=None):
        self.config = config
        self.device = device
        self.logger = logger_instance
        self.model = None
        self.image_processor = None
        self.enable_half = (self.device.type == 'cuda') # Half Precision 사용 여부
        self.inference_size_hw = (config.inference_size, config.inference_size)
        self.inference_size_wh = (config.inference_size, config.inference_size)
        
        if not config.use_semantic:
            self._log("Semantic segmentation disabled - using RGB only")
            return
            
        self.model_type = config.model_type
        self._load_model()
        
        if self.enable_half:
            self._log("⚡ Half Precision (FP16) enabled for inference")

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _load_model(self):
        """Load the specified model"""
        if self.model_type == "custom-object":
            self._load_custom_model()
        elif self.model_type == "custom-surface":
            # (참고) custom-surface가 custom-object와 동일한 구조/가중치 로딩을 쓴다면
            # 여기도 동일하게 수정되어야 합니다. 일단 custom-object만 수정합니다.
            self._load_custom_model() 
        elif self.model_type == "segformer-cityscapes":
            self._load_segformer()
        elif self.model_type == "maskformer-cityscapes":
            self._load_maskformer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # --- ⬇️ 수정된 부분 2: _load_custom_model 키 매핑 수정 ⬇️ ---
    def _load_custom_model(self):
        """Load custom trained SegFormer (ROS-compatible)"""
        # 1. CustomSegFormer (내부에 original_model 보유) 인스턴스 생성
        self.model = CustomSegFormer(num_classes=self.config.num_custom_classes)
        
        model_path = self.config.active_model_name

        if not os.path.exists(model_path):
            self._log(f"⚠️ 모델 파일이 존재하지 않습니다: {model_path}")
            self._log("경고: 모델 가중치 없이 초기화합니다.")
            self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return

        try:
            # 2. 체크포인트 로드
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )
            
            # 'model' 키가 있는지 확인 (학습 시 저장 방식에 따라)
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            
            # 3. state_dict 키 정리 (ROS 코드와 동일하게)
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    # 키 접두사를 'original_model.'로 변경
                    new_key = 'original_model.' + key
                else:
                    # 'original_model.'이 이미 붙어있거나 다른 키일 경우 그대로 사용
                    new_key = key
            
                new_state_dict[new_key] = value

            # 4. self.model (CustomSegFormer)에 가중치 로드
            self.model.load_state_dict(new_state_dict, strict=False)

            self._log(f"✅ Custom model loaded from {model_path} (ROS-compatible keys)")

        except Exception as e:
            self._log(f"⚠️ Model loading failed: {e}")
            self._log("Using model without pretrained weights")
            
        self.model.to(self.device)
        self.model.eval()
        # 5. 전처리 정의 (리사이즈 없음)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # --- ⬆️ 수정된 부분 2 ⬆️ ---

    def _load_segformer(self):
        """Load SegFormer model"""
        model_name = self.config.active_model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._log("✅ SegFormer model loaded")

    def _load_maskformer(self):
        """Load MaskFormer model"""
        model_name = self.config.active_model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._log("✅ MaskFormer-COCO model loaded")

    def predict(self, rgb_image):
        """Run semantic segmentation on RGB image (BGR OpenCV format)"""
        if not self.config.use_semantic:
            return None
        
        # 모델 타입에 따라 분기
        if self.model_type == "custom-object":
            return self._predict_custom(rgb_image)
        elif self.model_type == "custom-surface":
            return self._predict_custom(rgb_image) # custom-surface도 동일 로직 가정
        elif self.model_type == "segformer-cityscapes":
            return self._predict_segformer(rgb_image)
        elif self.model_type == "maskformer-cityscapes":
            return self._predict_maskformer(rgb_image)

    # --- ⬇️ 수정된 부분 3: _predict_custom 리사이즈 제거 ⬇️ ---
    def _predict_custom(self, rgb_image):
        h_orig, w_orig = rgb_image.shape[:2]
        
        # 1. (제거) 사전 리사이즈 로직 삭제
        # 2. 원본 이미지를 바로 BGR -> RGB 변환
        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        # 3. 전처리 적용 (리사이즈 없음)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 4. Half Precision (autocast) 적용
            with autocast(enabled=self.enable_half):
                logits = self.model(input_tensor) # self.model은 CustomSegFormer
            
            # FP16 출력 -> FP32로 변환 (F.interpolate 호환성)
            if self.enable_half:
                logits = logits.float()
                
        # 5. 원본 해상도로 업샘플링 (ROS 코드와 동일)
        logits = F.interpolate(
            logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        
        pred_mask = torch.argmax(logits, dim=1).squeeze()
        return pred_mask.cpu().numpy().astype(np.uint8)
    # --- ⬆️ 수정된 부분 3 ⬆️ ---

    def _predict_segformer(self, rgb_image):
        h_orig, w_orig = rgb_image.shape[:2]
        
        # (참고) Segformer/Maskformer는 inference_size를 사용하지 않고,
        # AutoImageProcessor가 내부적으로 리사이즈를 처리합니다.
        
        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Half Precision (autocast) 적용
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)
        
        if self.enable_half:
            outputs.logits = outputs.logits.float()
            
        result = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h_orig, w_orig)]
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)

    def _predict_maskformer(self, rgb_image):
        h_orig, w_orig = rgb_image.shape[:2]

        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)

        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Half Precision (autocast) 적용
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)
                
        if self.enable_half:
            if outputs.class_queries_logits is not None:
                outputs.class_queries_logits = outputs.class_queries_logits.float()
            if outputs.masks_queries_logits is not None:
                outputs.masks_queries_logits = outputs.masks_queries_logits.float()
                
        result = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h_orig, w_orig)]
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)

# ============================================================================
# 3. 시맨틱 마스크 시각화 클래스
# (이전과 동일)
# ============================================================================

class SemanticVisualizer:
    """시맨틱 마스크를 GPU에서 RGB 컬러로 변환하고 원본 이미지와 블렌딩"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_classes = config.num_custom_classes
        self._init_semantic_colormap()
        logger.info(f'GPU 시맨틱 컬러맵 생성 완료 ({self.num_classes} classes)')

    def _init_semantic_colormap(self):
        """시맨틱 라벨을 RGB로 변환하기 위한 GPU 컬러맵 생성"""
        # Cityscapes (19 classes) 예시 컬러맵 (R, G, B)
        cityscapes_palette = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
            [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
            [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ]
        
        colors = torch.zeros((self.num_classes, 3), dtype=torch.uint8, device=self.device)
        
        # 클래스 이름과 인덱스 매핑
        for i, (name, idx) in enumerate(self.config.custom_class_names.items()):
            if idx >= self.num_classes:
                continue 
                
            if i < len(cityscapes_palette):
                colors[idx] = torch.tensor(cityscapes_palette[i], dtype=torch.uint8, device=self.device)
            else:
                r = (i * 50) % 255
                g = (i * 90) % 255
                b = (i * 120) % 255
                colors[idx] = torch.tensor([r, g, b], dtype=torch.uint8, device=self.device)
        
        if 0 < self.num_classes:
            colors[0] = torch.tensor([0, 0, 0], dtype=torch.uint8, device=self.device)
        
        self.semantic_colormap_gpu = colors

    def apply_colormap(self, mask_tensor, original_image_bgr_tensor, alpha=0.6):
        """
        GPU에서 마스크에 컬러맵을 적용하고 원본 이미지(BGR)와 블렌딩합니다.
        """
        colors_rgb = self.semantic_colormap_gpu[mask_tensor.long()]
        colors_bgr = colors_rgb[..., [2, 1, 0]] 
        
        blended_gpu = (
            original_image_bgr_tensor.float() * (1.0 - alpha) + 
            colors_bgr.float() * alpha
        )
        
        return blended_gpu.to(torch.uint8)

# ============================================================================
# 4. 비디오 처리 메인 클래스
# (이전과 동일)
# ============================================================================

class VideoProcessor:
    """비디오를 로드하고, 모델 추론을 실행하며, 결과 비디오를 저장"""
    
    def __init__(self, config: ReconstructionConfig, input_path: str, output_path: str):
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        self.model = SemanticModel(config, self.device, logger_instance=self)
        self.visualizer = SemanticVisualizer(config, self.device)
        self.timings = deque(maxlen=200)

    # SemanticModel이 사용할 로깅 메소드
    def info(self, msg):
        logger.info(msg)

    def process_video(self):
        """비디오 처리를 시작합니다."""
        logger.info(f"--- 🚀 모델 [{self.config.model_type}] 처리 시작 ---")
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            logger.error(f"오류: 비디오 파일을 열 수 없습니다. {self.input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"입력 비디오: {width}x{height} @ {fps:.2f} FPS, 총 {frame_count} 프레임")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error(f"오류: 비디오 파일을 쓸 수 없습니다. {self.output_path}")
            cap.release()
            return
            
        logger.info(f"출력 비디오 저장 위치: {self.output_path}")

        try:
            pbar = tqdm(total=frame_count, desc=f"Processing {self.config.model_type}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                t_start = time.perf_counter()
                
                # 5. 시맨틱 예측 (NumPy BGR 입력 -> NumPy (H, W) 라벨 출력)
                pred_mask_np = self.model.predict(frame)
                
                if pred_mask_np is None:
                    logger.warning("시맨틱 마스크 생성 실패, 프레임 건너뜀")
                    writer.write(frame) # 원본 프레임 저장
                    continue

                # 6. 시각화 (GPU 가속)
                frame_gpu = torch.from_numpy(frame).to(self.device)
                mask_gpu = torch.from_numpy(pred_mask_np).to(self.device)
                
                blended_frame_gpu = self.visualizer.apply_colormap(
                    mask_gpu, frame_gpu, alpha=0.6
                )
                
                blended_frame_np = blended_frame_gpu.cpu().numpy()
                
                writer.write(blended_frame_np)
                
                self.timings.append(time.perf_counter() - t_start)
                pbar.update(1)

            pbar.close()

        except Exception as e:
            logger.error(f"비디오 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            writer.release()
            logger.info("비디오 캡처 및 쓰기 객체 해제 완료")

        if self.timings:
            avg_time_ms = np.mean(self.timings) * 1000
            avg_fps = 1000.0 / avg_time_ms
            logger.info(f"✅ 처리 완료: {self.config.model_type}")
            logger.info(f"   평균 처리 속도: {avg_fps:.2f} FPS ({avg_time_ms:.2f} ms/frame)")

# ============================================================================
# 5. 메인 실행 함수
# (이전과 동일)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MP4 비디오에 4가지 시맨틱 세그멘테이션 모델을 적용합니다."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="입력 MP4 비디오 파일 경로"
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return

    # 처리할 모델 타입 리스트
    model_types_to_run = [
        # "custom-object",
        # "custom-surface",
        "segformer-cityscapes",
        "maskformer-cityscapes"
    ]
    
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.dirname(input_path) or "." 

    for model_type in model_types_to_run:
        try:
            config = ReconstructionConfig(model_type=model_type)
            
            output_filename = f"{name_without_ext}_{model_type}_output.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            processor = VideoProcessor(config, input_path, output_path)
            processor.process_video()
            
            # GPU 캐시 클리어 (메모리 부족 방지)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"--- ❌ 모델 [{model_type}] 처리 중 심각한 오류 발생 ---")
            logger.error(e)
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
