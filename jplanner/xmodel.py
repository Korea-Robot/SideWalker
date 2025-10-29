#!/usr/bin/env python3

"""
semantic_model.py

PyTorch 기반 시맨틱 세그멘테이션 모델을 로드하고 추론을 실행하는 모듈.
- CustomSegFormer: 커스텀 학습된 SegFormer 모델 정의
- SemanticModel: HuggingFace 및 커스텀 모델을 위한 통합 인터페이스
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)
from torch.cuda.amp import autocast

# 설정 파일에서 ReconstructionConfig 클래스를 임포트
from xconfig import ReconstructionConfig
import cv2 
import numpy as np
import torch.nn.functional as F 

class CustomSegFormer(nn.Module):
    """Custom trained SegFormer model"""
    def __init__(self, num_classes: int = 30, pretrained_name: str = "nvidia/mit-b0"):
        super().__init__()
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes, # (수정) 원본 코드의 'config' 대신 'num_classes' 사용
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        except ValueError as e:
            # 가중치 파일(.safetensors)이 없을 때 발생하는 오류 처리
            if "torch.load" in str(e):
                print(f"Warning: {e}")
                print("Creating model architecture without pretrained weights...")
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(pretrained_name)
                config.num_labels = num_classes
                self.model = SegformerForSemanticSegmentation(config)
            else:
                raise e
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits

class SemanticModel:
    """Unified interface for different semantic segmentation models"""
    
    def __init__(self, config: ReconstructionConfig, device, logger=None):
        self.config = config
        self.device = device
        self.logger = logger
        self.model = None
        self.image_processor = None
        self.enable_half = (self.device.type == 'cuda' and self.config.use_gpu)
        
        if not config.use_semantic:
            self._log("Semantic segmentation disabled - using RGB only")
            return

        self.inference_size_hw = (config.inference_size, config.inference_size)
        self.inference_size_wh = (config.inference_size, config.inference_size)
        self.model_type = config.model_type
        
        self._load_model()
        
        if self.enable_half:
            self._log("⚡ Half Precision (FP16) enabled for inference")

    def _log(self, msg):
        """로거가 있으면 로깅, 없으면 print"""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _load_model(self):
        """Load the specified model"""
        if self.model_type == "custom-object":
            self._load_custom_model()
        elif self.model_type == "custom-surface":
            self._load_custom_model()
        elif self.model_type == "segformer-cityscapes":
            self._load_segformer()
        elif self.model_type == "maskformer-cityscapes":
            self._load_maskformer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_custom_model(self):
        """Load custom trained SegFormer"""
        self.model = CustomSegFormer(num_classes=self.config.num_custom_classes)
        try:
            checkpoint = torch.load(
                self.config.custom_model_path,
                map_location=self.device,
                weights_only=False
            )
            # 체크포인트 키 이름 보정
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'model.' + key
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self._log(f"✅ Custom model loaded from {self.config.custom_model_path}")
        except Exception as e:
            self._log(f"⚠️ Model loading failed: {e}")
            self._log("Using model without pretrained weights")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 커스텀 모델용 이미지 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
        """
        Run semantic segmentation on RGB image (BGR OpenCV format)
        Returns: NumPy array (H, W) of class indices
        """
        if not self.config.use_semantic:
            return None
        
        # 모델 타입에 따라 적절한 추론 함수 호출
        if self.model_type in ["custom-object", "custom-surface"]:
            return self._predict_custom(rgb_image)
        elif self.model_type == "segformer-cityscapes":
            return self._predict_segformer(rgb_image)
        elif self.model_type == "maskformer-cityscapes":
            return self._predict_maskformer(rgb_image)
        else:
            return None # Should not happen

    def _predict_custom(self, rgb_image):
        """추론: 커스텀 SegFormer 모델"""
        h_orig, w_orig = rgb_image.shape[:2]
        
        # 리사이징
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, self.inference_size_wh, interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image
        
        # BGR -> RGB -> PIL -> Tensor
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast(enabled=self.enable_half):
                logits = self.model(input_tensor)
            if self.enable_half:
                logits = logits.float() # FP32로 복원

        # 원본 크기로 업샘플링
        logits = F.interpolate(
            logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        pred_mask = torch.argmax(logits, dim=1).squeeze()
        
        return pred_mask.cpu().numpy().astype(np.uint8)

    def _predict_segformer(self, rgb_image):
        """추론: HuggingFace SegFormer 모델"""
        h_orig, w_orig = rgb_image.shape[:2]
        
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, self.inference_size_wh, interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image

        # BGR -> RGB -> PIL
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        # ImageProcessor 사용
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)
        
        if self.enable_half:
            outputs.logits = outputs.logits.float()

        # 후처리 (업샘플링 포함)
        result = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h_orig, w_orig)]
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)

    def _predict_maskformer(self, rgb_image):
        """추론: HuggingFace MaskFormer 모델"""
        h_orig, w_orig = rgb_image.shape[:2]

        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, self.inference_size_wh, interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image

        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)

        # FP32로 복원
        if self.enable_half:
            if outputs.class_queries_logits is not None:
                outputs.class_queries_logits = outputs.class_queries_logits.float()
            if outputs.masks_queries_logits is not None:
                outputs.masks_queries_logits = outputs.masks_queries_logits.float()

        # 후처리
        result = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h_orig, w_orig)]
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)
