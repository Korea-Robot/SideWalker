#!/usr/bin/env python3
"""
Configuration file for Semantic Point Cloud Reconstruction
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ReconstructionConfig:
    """Main configuration for point cloud reconstruction"""
    
    # ============================================================================
    # Semantic Model Settings
    # ============================================================================
    use_semantic: bool = True  # False로 설정하면 RGB만 사용
    # model_type: str = "custom"  # "custom", "segformer-ade20k", "maskformer-coco"
    model_type: str = "segformer-ade20k" 
    # model_type: str = "maskformer-coco" 

    # Custom model settings
    custom_model_path: str = "best_model2.pth"

    if model_type=="maskformer-coco":
        inference_size: int = 384
    else:
        inference_size: int = 1024 # 512 # 256 
    
    # Pre-trained model checkpoints
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    # segformer_checkpoint: str ="peldrak/segformer-b0-cityscapes-512-512-finetuned-coastTrain"
    segformer_checkpoint: str = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    segformer_checkpoint: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

    
    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # model = SegformerForSemanticSegmentation.from_pretrained("segments-tobias/segformer-b0-finetuned-segments-sidewalk")
    
    # maskformer_checkpoint: str = "facebook/maskformer-swin-base-coco"
    # maskformer_checkpoint: str ="facebook/mask2former-swin-tiny-coco-panoptic"
    
    maskformer_checkpoint: str ="facebook/mask2former-swin-tiny-cityscapes-panoptic" # best
    # maskformer_checkpoint: str ="facebook/mask2former-swin-large-cityscapes-semantic"
    # maskformer_checkpoint: str ="facebook/mask2former-swin-tiny-cityscapes-semantic" # best
    
    # ============================================================================
    # Camera Parameters (Intel RealSense D455)
    # ============================================================================
    depth_intrinsics: CameraIntrinsics = CameraIntrinsics(
        fx=431.0625,
        fy=431.0625,
        cx=434.492,
        cy=242.764
    )
    
    rgb_intrinsics: CameraIntrinsics = CameraIntrinsics(
        fx=645.4923,
        fy=644.4183,
        cx=653.03259,
        cy=352.28909
    )
    
    # ============================================================================
    # ROS Topics
    # ============================================================================
    depth_topic: str = '/camera/camera/depth/image_rect_raw'
    rgb_topic: str = '/camera/camera/color/image_raw'
    pointcloud_topic: str = '/semantic_pointcloud'
    
    # ============================================================================
    # Frame IDs
    # ============================================================================
    source_frame: str = 'camera_depth_optical_frame'
    # target_frame: str = 'body'
    target_frame: str = 'camera_link'
    
    # ============================================================================
    # Processing Parameters
    # ============================================================================
    downsample_y: int = 6 #9  # Vertical downsampling factor
    downsample_x: int = 4 #6  # Horizontal downsampling factor
    sync_slop: float = 0.1  # Time synchronization tolerance (seconds)
    
    # Performance
    use_gpu: bool = True  # Automatically detect if False
    
    # ============================================================================
    # Custom Model Class Definitions
    # ============================================================================
    # custom_class_names: Dict[str, int] = None
    
    custom_class_names = {
        "road": 0,
        "sidewalk": 1,
        "building": 2,
        "wall":3,
        "fence":4,
        "pole":5,
        "traffic light":6,
        "traffic sign":7,
        "vegetation":8,
        "terrain":9,
        "sky":10,
        "person":11,
        "rider":12,
        "car":13,
        "truck":14,
        "bus":15,
        "train":16,
        "motorcycle":17,
        "bicycle":18
    }
    def __post_init__(self):
        if self.custom_class_names is None:
            self.custom_class_names = {
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
    
    @property
    def num_custom_classes(self) -> int:
        return len(self.custom_class_names)
    
    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.custom_class_names.items()}
