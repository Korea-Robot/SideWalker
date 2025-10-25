#!/usr/bin/env python3
"""
Semantic Segmentation Models for Point Cloud Reconstruction
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage
from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    # MaskFormerForInstanceSegmentation
    Mask2FormerForUniversalSegmentation
)
from torch.cuda.amp import autocast ### MODIFIED: Import autocast

class CustomSegFormer(nn.Module):
    """Custom trained SegFormer model"""
    
    def __init__(self, num_classes: int = 30, pretrained_name: str = "nvidia/mit-b0"):
        super().__init__()
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=config.num_custom_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        except ValueError as e:
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
    
    def __init__(self, config, device, logger=None):
        self.config = config
        self.device = device
        self.logger = logger
        self.model = None
        self.image_processor = None

        ### MODIFIED: Add flags for optimization ###
        self.enable_half = (self.device.type == 'cuda')
        # (H, W) for transforms/logic
        self.inference_size_hw = (config.inference_size, config.inference_size) 
        # (W, H) for cv2.resize
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
            
            # Map checkpoint keys
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

        ### MODIFIED: Remove Resize from transform, as we pre-resize ###
        self.transform = transforms.Compose([
            # transforms.Resize((self.config.inference_size, self.config.inference_size)), # Removed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_segformer(self):
        """Load SegFormer model"""

        model_name = self.config.active_model_name

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, #self.config.segformer_checkpoint,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, #self.config.segformer_checkpoint,
            # num_labels=self.config.num_custom_classes
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
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, #self.config.maskformer_checkpoint
        )
        # self.model = MaskFormerForInstanceSegmentation.from_pretrained(
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, #self.config.maskformer_checkpoint
        )
        self.model.to(self.device)
        self.model.eval()
        self._log("✅ MaskFormer-COCO model loaded")
    
    def predict(self, rgb_image):
        """
        Run semantic segmentation on RGB image
        
        Args:
            rgb_image: BGR image (OpenCV format)
            
        Returns:
            semantic_mask: (H, W) numpy array with class labels, or None if semantic disabled
        """
        if not self.config.use_semantic:
            return None
        
        if self.model_type == "custom-object":
            return self._predict_custom(rgb_image)
        elif self.model_type == "custom-surface":
            return self._predict_custom(rgb_image)
        elif self.model_type == "segformer-cityscapes":
            return self._predict_segformer(rgb_image)
        elif self.model_type == "maskformer-cityscapes":
            return self._predict_maskformer(rgb_image)
    
    def _predict_custom(self, rgb_image):
        """Custom model inference"""
        
        ### MODIFIED: Pre-resize image ###
        h_orig, w_orig = rgb_image.shape[:2]
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, 
                self.inference_size_wh, # (W, H) for cv2
                interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image
        
        # BGR -> RGB
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            ### MODIFIED: Use autocast for FP16 ###
            with autocast(enabled=self.enable_half):
                logits = self.model(input_tensor)
            
            # Cast back to float32 for interpolation safety
            if self.enable_half:
                logits = logits.float()
        
        ### MODIFIED: Upsample to original size ###
        logits = F.interpolate(
            logits,
            size=(h_orig, w_orig), # Use original shape
            mode='bilinear',
            align_corners=False
        )
        
        # Get class predictions
        pred_mask = torch.argmax(logits, dim=1).squeeze()
        return pred_mask.cpu().numpy().astype(np.uint8)
    
    def _predict_segformer(self, rgb_image):
        """SegFormer-ADE20k inference"""

        ### MODIFIED: Pre-resize image ###
        h_orig, w_orig = rgb_image.shape[:2]
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, 
                self.inference_size_wh, # (W, H) for cv2
                interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image
        
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            ### MODIFIED: Use autocast for FP16 ###
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)
        
        ### MODIFIED: Cast back to float32 for post-processing ###
        if self.enable_half:
            outputs.logits = outputs.logits.float()

        result = self.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(h_orig, w_orig)] ### MODIFIED: Use original shape
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)
    
    def _predict_maskformer(self, rgb_image):
        """MaskFormer-COCO inference"""

        ### MODIFIED: Pre-resize image ###
        h_orig, w_orig = rgb_image.shape[:2]
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, 
                self.inference_size_wh, # (W, H) for cv2
                interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image
            
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            ### MODIFIED: Use autocast for FP16 ###
            with autocast(enabled=self.enable_half):
                outputs = self.model(**inputs)
        
        ### MODIFIED: Cast back to float32 for post-processing ###
        if self.enable_half:
            # MaskFormer has multiple outputs, cast all relevant ones
            if outputs.class_queries_logits is not None:
                outputs.class_queries_logits = outputs.class_queries_logits.float()
            if outputs.masks_queries_logits is not None:
                outputs.masks_queries_logits = outputs.masks_queries_logits.float()

        # result = self.image_processor.post_process_panoptic_segmentation(
        result = self.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(h_orig, w_orig)] ### MODIFIED: Use original shape
        )[0]
        
        return result.cpu().numpy().astype(np.uint8)
        # return result["segmentation"].cpu().numpy().astype(np.uint8)

"""
네! 이번에 적용된 최적화는 `SemanticModel`의 추론 속도를 높이는 데 초점을 맞춘, 매우 효과적인 두 가지 기법입니다.

어떤 부분이 바뀌었고 왜 그렇게 했는지 하나하나 설명해 드릴게요.

---

## 1. ⚡️ Half Precision (FP16) 추론 적용

**"연산 정밀도를 낮춰서 속도를 2배 가까이 올립니다."**

* **WHAT (무엇이 바뀌었나?)**
    1.  `from torch.cuda.amp import autocast`를 임포트했습니다.
    2.  `__init__` 함수에 `self.enable_half = (self.device.type == 'cuda')`라는 플래그를 추가했습니다. (GPU를 사용할 때만 활성화됩니다.)
    3.  모든 `_predict_...` 함수의 `with torch.no_grad():` 내부에 `with autocast(enabled=self.enable_half):` 블록을 추가했습니다.
    4.  모델 추론(`self.model(...)`)이 이 `autocast` 블록 안에서 실행됩니다.
    5.  `autocast` 블록 직후, `logits.float()`처럼 모델의 출력값을 다시 `.float()` (FP32)로 변환했습니다.

* **WHY (왜 바뀌었나?)**
    * **속도 향상:** `autocast`는 PyTorch의 **Automatic Mixed Precision** 기능입니다. 이 블록 안에서 실행되는 연산(특히 CNN, Linear 레이어)은 기본 정밀도인 **FP32 (32비트)** 대신 **FP16 (16비트)**으로 수행됩니다.
    * NVIDIA GPU의 **Tensor Core**는 FP16 연산에 특화되어 있어, FP32 연산보다 훨씬 빠릅니다. (이론적으로 최대 2배 이상)
    * **메모리 절약:** 모델 가중치와 중간 계산 값들이 사용하는 GPU 메모리도 절반으로 줄어듭니다.
    * **안정성:** `logits.float()`로 다시 변환하는 이유는, `F.interpolate`나 `argmax` 같은 후속 연산이 CPU에서 실행되거나 FP32 정밀도를 요구할 때 발생할 수 있는 오류를 방지하기 위해서입니다. 가장 무거운 연산(모델 추론)은 이미 FP16으로 빠르게 끝난 상태입니다.

---

## 2. 🖼️ 입력 이미지 리사이즈 (전처리)

**"모델에 작은 이미지를 입력해서 연산량을 획기적으로 줄입니다."**

* **WHAT (무엇이 바뀌었나?)**
    1.  `__init__` 함수에 `self.inference_size_hw` (높이, 너비)와 `self.inference_size_wh` (너비, 높이) 변수를 추가했습니다. (OpenCV는 `(W, H)` 순서를 사용하므로 명확히 구분)
    2.  모든 `_predict_...` 함수가 시작될 때, 원본 이미지의 크기(`h_orig`, `w_orig`)를 먼저 저장합니다.
    3.  `cv2.resize` 함수를 사용해, 원본 이미지를 `self.inference_size_wh` (예: 512x512) 크기로 **먼저 축소**합니다. (`rgb_image_resized` 생성)
    4.  `cv2.cvtColor`, `PILImage.fromarray`, `self.transform` (또는 `image_processor`) 등 모든 전처리가 이 **축소된 이미지**를 기반으로 수행됩니다.

* **WHY (왜 바뀌었나?)**
    * **병목 현상 제거:** RealSense에서 들어오는 1280x720 (약 92만 픽셀) 또는 1920x1080 (약 207만 픽셀) 고해상도 이미지를 그대로 모델에 입력하는 것은 엄청난 병목입니다.
    * **연산량 감소:** 모델의 연산량(FLOPs)은 입력 해상도에 비례(또는 그 이상)합니다. 이미지를 512x512 (약 26만 픽셀)로 줄여서 입력하면, 모델이 처리할 픽셀 수가 **1/3 ~ 1/8** 수준으로 줄어들어 추론 속도가 압도적으로 빨라집니다.

---

## 3. 📈 마스크 업샘플링 (후처리)

**"작게 예측된 마스크를 다시 원본 크기로 확대합니다."**

* **WHAT (무엇이 바뀌었나?)**
    1.  `_predict_custom`의 `F.interpolate` 함수의 `size` 인자를 `(h_orig, w_orig)` (원본 크기)로 변경했습니다.
    2.  `_predict_segformer`, `_predict_maskformer`의 `post_process_...` 함수의 `target_sizes` 인자를 `[(h_orig, w_orig)]` (원본 크기)로 변경했습니다.

* **WHY (왜 바뀌었나?)**
    * **해상도 일치:** 2번에서 우리는 **작은 이미지**로 추론했기 때문에, 결과물로 나온 마스크도 **작은 크기**입니다.
    * 하지만 `ReconstructionNode`의 `_align_to_depth` 함수는 **원본 이미지**와 동일한 해상도의 마스크를 기대합니다.
    * 따라서 추론 직후, 모델이 출력한 작은 마스크(예: 512x512)를 `F.interpolate` (bilinear 보간) 또는 `post_process` 기능을 이용해 다시 원본 크기(예: 1280x720)로 **확대**하는 과정이 필요합니다.

---

## 4. 🧹 기타 정리 (중복 제거)

* **WHAT (무엇이 바뀌었나?)**
    * `_load_custom_model`의 `self.transform` (PyTorch `transforms.Compose`) 리스트에서 `transforms.Resize(...)` 단계를 **제거**했습니다.

* **WHY (왜 바뀌었나?)**
    * **중복 제거:** 2번에서 `cv2.resize`를 사용해 이미지를 미리 리사이즈하기 때문에, PyTorch의 `transforms.Resize`가 또 실행될 필요가 없습니다. (OpenCV의 `cv2.resize`가 PIL을 이용하는 `transforms.Resize`보다 일반적으로 더 빠릅니다.)

---

## 🎯 결론

이 최적화들을 통해 `semantic_model.predict()` 함수는 이제 다음과 같이 동작합니다.

1.  **[Pre-process]** 큰 (720p) BGR 이미지를 받음.
2.  `cv2.resize`로 작은 (512x512) 이미지로 축소 (속도 향상 ⬆️).
3.  **[Inference]** `autocast`를 사용해 FP16으로 모델 추론 (속도 향상 ⬆️, 메모리 감소 ⬇️).
4.  **[Post-process]** 작은 (512x512) 마스크 결과를 얻음.
5.  `F.interpolate`로 큰 (720p) 마스크로 확대.
6.  `ReconstructionNode`로 최종 마스크 반환.

결과적으로 **시맨틱 세그멘테이션 단계의 속도가 크게 향상**되어, `rgbd_callback`의 전체적인 FPS(초당 프레임 수)가 올라가게 됩니다.
"""
