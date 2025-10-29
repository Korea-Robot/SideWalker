#!/usr/bin/env python3

"""
reconstruction_config.py

시맨틱 포인트클라우드 및 BEV 생성을 위한 모든 설정 값을 정의하는 모듈입니다.
- 클래스 사전 (Objects, Surfaces, Cityscapes)
- 카메라 내부 파라미터 (Intrinsics)
- 메인 설정 데이터 클래스 (ReconstructionConfig)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

# ============================================================================
# 1. Custom Object Classes
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

# ============================================================================
# 2. Custom Surface Classes
# ============================================================================
SURFACE_CLASSES = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_block': 5, 'sidewalk': 6
}

# ============================================================================
# 3. Cityscapes Classes (Segformer, Maskformer 공통 사용)
# ============================================================================
CITYSCAPES_CLASSES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3,
    "fence": 4, "pole": 5, "traffic light": 6, "traffic sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10, "person": 11,
    "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18
}

# ============================================================================
# 4. 데이터 클래스 정의
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
    
    # --- 모델 설정 ---
    use_semantic: bool = True
    # model_type: str ="maskformer-cityscapes" # "custom-object", "custom-surface", "segformer-cityscapes", "maskformer-cityscapes"
    model_type: str ="segformer-cityscapes"
    # model_type: str ="custom-object"
    # model_type: str ="custom-surface"
    custom_object_model_path: str = "./models/dynamic_object/best_model2.pth.pth"
    custom_surface_model_path: str = "./models/surface/surface_mask_best_lrup.pt.pth"
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic"


    # --- Pre-trained Model Checkpoints (필요시 수정) ---
    # segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"   # good 

    # segformer_checkpoint: str = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
    # segformer_checkpoint: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    # segformer_checkpoint: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" 
    # segformer_checkpoint: str = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" # bad 
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" # good

    # maskformer_checkpoint: str = "facebook/mask2former-swin-base-coco"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-coco"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-panoptic"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic" # best
    # maskformer_checkpoint: str = "facebook/mask2former-swin-small-cityscapes-semantic"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-large-cityscapes-semantic"
    # maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-coco-panoptic"

    
    # --- 카메라 및 토픽 설정 ---
    depth_topic: str = '/camera/camera/depth/image_rect_raw'
    rgb_topic: str = '/camera/camera/color/image_raw'
    pointcloud_topic: str = '/semantic_pointcloud' # 노드에서 재정의됨
    bev_topic: str = '/semantic_bev_map' # 노드에서 재정의됨
    
    # --- 좌표계 설정 ---
    source_frame: str = 'camera_depth_optical_frame'
    target_frame: str = 'camera_link'
    
    # --- PCL/BEV 재구성 파라미터 ---
    downsample_y: int = 9
    downsample_x: int = 6
    sync_slop: float = 0.1
    use_gpu: bool = True

    # --- 카메라 내부 파라미터 (D435 기본값 예시) ---
    depth_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=431.0625, fy=431.0625, cx=434.492, cy=242.764
    ))
    rgb_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=645.4923, fy=644.4183, cx=653.03259, cy=352.28909
    ))
    
    # --- 내부 초기화 변수 ---
    active_model_name: str = field(init=False)
    inference_size: int = field(init=False)
    custom_class_names: Dict[str, int] = field(init=False)

    def __post_init__(self):
        """설정 값에 따라 모델별 파라미터를 자동으로 설정합니다."""
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
            # self.inference_size = 512
            self.inference_size = 1024
        elif self.model_type == "maskformer-cityscapes":
            self.active_model_name = self.maskformer_checkpoint
            self.custom_class_names = CITYSCAPES_CLASSES.copy()
            self.inference_size = 384
        else:
            raise ValueError(f"알 수 없는 model_type입니다: {self.model_type}")

    @property
    def num_custom_classes(self) -> int:
        """활성화된 모델의 클래스 개수를 반환합니다."""
        return len(self.custom_class_names)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        """인덱스 -> 클래스 이름 변환 사전을 반환합니다."""
        return {v: k for k, v in self.custom_class_names.items()}
