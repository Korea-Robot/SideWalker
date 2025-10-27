#!/usr/bin/env python3
"""
Configuration file for Semantic Point Cloud Reconstruction
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

# ============================================================================
# 기본 클래스 맵 상수
# ============================================================================

# 1. Custom Object Classes
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

# 2. Custom Surface Classes
SURFACE_CLASSES = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_block': 5, 'sidewalk': 6
}

# 3. Cityscapes Classes (Segformer, Maskformer 공통 사용)
CITYSCAPES_CLASSES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3,
    "fence": 4, "pole": 5, "traffic light": 6, "traffic sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10, "person": 11,
    "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18
}

# ============================================================================
# 데이터 클래스 정의
# ============================================================================

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
    use_semantic: bool =  True  # False로 설정하면 RGB만 사용
    
    # 사용할 모델 타입을 선택합니다.
    # __post_init__에서 이 값에 따라 다른 설정들이 동적으로 로드됩니다.
    # model_type: str = "custom-object"  
    # 사용 가능 옵션: 
    # "custom-object", "custom-surface", 
    # "segformer-cityscapes", "maskformer-cityscapes",
    # "segformer-ade20k", "maskformer-coco"
    model_type: str ="maskformer-cityscapes"
    # model_type: str ="segformer-cityscapes"

    # --- Custom Model Paths ---
    custom_model_path: str = "best_object_model.pth"
    custom_model_path: str = "best_surface_model.pth"

    # --- Pre-trained Model Checkpoints (필요시 수정) ---
    # segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"   # good 

    # segformer_checkpoint: str = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
    # segformer_checkpoint: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    # segformer_checkpoint: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" 
    # segformer_checkpoint: str = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" # bad 
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" # good

    maskformer_checkpoint: str = "facebook/mask2former-swin-base-coco"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-coco"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-panoptic"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic" # best
    # maskformer_checkpoint: str = "facebook/mask2former-swin-small-cityscapes-semantic"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-large-cityscapes-semantic"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-coco-panoptic"

    # --- Dynamically Set Fields (init=False) ---
    # __post_init__에서 model_type에 따라 자동으로 설정되는 필드들
    
    # 실제 로드될 모델의 경로 또는 체크포인트 이름
    active_model_name: str = field(init=False) 
    
    # 모델에 맞는 추론 이미지 크기
    inference_size: int = field(init=False)
    
    # 모델에 맞는 클래스 이름 맵
    custom_class_names: Dict[str, int] = field(init=False)

    # ============================================================================
    # Camera Parameters (Intel RealSense D455)
    # ============================================================================
    # default_factory를 사용하여 각 인스턴스가 고유한 객체를 갖도록 함
    depth_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=431.0625,
        fy=431.0625,
        cx=434.492,
        cy=242.764
    ))

    rgb_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=645.4923,
        fy=644.4183,
        cx=653.03259,
        cy=352.28909
    ))

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
    target_frame: str = 'camera_link'

    # ============================================================================
    # Processing Parameters
    # ============================================================================
    downsample_y: int = 9  # Vertical downsampling factor
    downsample_x: int = 6  # Horizontal downsampling factor
    sync_slop: float = 0.1  # Time synchronization tolerance (seconds)

    # Performance
    use_gpu: bool = True  # Automatically detect if False

    # ============================================================================
    # Post Initialization
    # ============================================================================
    def __post_init__(self):
        """
        model_type에 기반하여 active_model_name, custom_class_names, 
        inference_size를 동적으로 설정합니다.
        """
        if self.model_type == "custom-object":
            self.active_model_name = self.custom_object_model_path
            self.custom_class_names = OBJECT_CLASSES.copy()
            self.inference_size = 512  # 커스텀 모델의 추론 크기 (예시)

        elif self.model_type == "custom-surface":
            self.active_model_name = self.custom_surface_model_path
            self.custom_class_names = SURFACE_CLASSES.copy()
            self.inference_size = 512  # 커스텀 모델의 추론 크기 (예시)

        # 요청대로 Cityscapes 클래스 맵을 공통으로 사용
        elif self.model_type == "segformer-cityscapes":
            self.active_model_name = self.segformer_checkpoint
            self.custom_class_names = CITYSCAPES_CLASSES.copy()
            self.inference_size = 512 #1024 # 512 # 체크포인트 이름(512-1024) 기준

        elif self.model_type == "maskformer-cityscapes":
            self.active_model_name = self.maskformer_checkpoint
            self.custom_class_names = CITYSCAPES_CLASSES.copy()
            self.inference_size = 384 # Cityscapes Maskformer는 1024 사용 (예시)
            
        else:
            raise ValueError(f"알 수 없는 model_type입니다: {self.model_type}")

    # ============================================================================
    # Properties (Getter)
    # ============================================================================
    @property
    def num_custom_classes(self) -> int:
        """동적으로 설정된 클래스 맵의 클래스 개수를 반환합니다."""
        return len(self.custom_class_names)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        """동적으로 설정된 클래스 맵의 (인덱스 -> 이름) 딕셔너리를 반환합니다."""
        return {v: k for k, v in self.custom_class_names.items()}
