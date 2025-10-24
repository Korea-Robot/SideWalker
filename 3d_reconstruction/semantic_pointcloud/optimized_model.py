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
    MaskFormerForInstanceSegmentation
)
from torch.cuda.amp import autocast ### MODIFIED: Import autocast

class CustomSegFormer(nn.Module):
    """Custom trained SegFormer model"""
    
    def __init__(self, num_classes: int = 30, pretrained_name: str = "nvidia/mit-b0"):
        super().__init__()
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes,
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
        if self.model_type == "custom":
            self._load_custom_model()
        elif self.model_type == "segformer-ade20k":
            self._load_segformer_ade20k()
        elif self.model_type == "maskformer-coco":
            self._load_maskformer_coco()
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
    
    def _load_segformer_ade20k(self):
        """Load SegFormer-ADE20k model"""
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.segformer_checkpoint,
            do_reduce_labels=True
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.config.segformer_checkpoint
        )
        self.model.to(self.device)
        self.model.eval()
        self._log("✅ SegFormer-ADE20k model loaded")
    
    def _load_maskformer_coco(self):
        """Load MaskFormer-COCO model"""
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.maskformer_checkpoint
        )
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            self.config.maskformer_checkpoint
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
        
        if self.model_type == "custom":
            return self._predict_custom(rgb_image)
        elif self.model_type == "segformer-ade20k":
            return self._predict_segformer(rgb_image)
        elif self.model_type == "maskformer-coco":
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

        result = self.image_processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(h_orig, w_orig)] ### MODIFIED: Use original shape
        )[0]
        
        return result["segmentation"].cpu().numpy().astype(np.uint8)
