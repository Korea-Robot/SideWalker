#!/usr/bin/env python3

### problem

"""

1. RGB + Depth image get
2. Semantic Model Use => Tensor RT & NPU use 
3. Depth 3D point cloud : inverse projection => CPU to GPU optimize
4. semantic + 3D point cloud  => Resolution & depth Error 
5. BEV map 
6. ros2 topic publish  

Message Synchronization & depth accuracy

Process all GPU -> CPU 

rgb camera info topic


# Optimization 

1. Preprocess 
- RGB decrease resolution
- Depth down sampling
- GPU data transfer : pinned memory use 

2. Semantic Segmentation speed up 
- Latency
- FPS 
- ONNX, TensorRT, TorchScript

3. Depth -> Pointcloud 
- GPU based transform 

4. Semantic + BEV vectorize
- loop removal
- GPU process

5. ROS Publish optimize 
- compressed image, publish rate limit, background thread 


---

header:

  stamp:

    sec: 1762930499

    nanosec: 303927246

  frame_id: camera_color_optical_frame

height: 480

width: 640

distortion_model: plumb_bob

d:

- -0.05512524023652077

- 0.06190275400876999

- -0.00023902612156234682

- -0.00012431867071427405

- -0.019336095079779625

k:

- 385.97442626953125

- 0.0

- 322.1943359375

- 0.0

- 385.46087646484375

- 238.75344848632812

- 0.0

- 0.0

- 1.0

r:

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

p:

- 385.97442626953125

- 0.0

- 322.1943359375

- 0.0

- 0.0

- 385.46087646484375

- 238.75344848632812

- 0.0

- 0.0

- 0.0

- 1.0

- 0.0

binning_x: 0

binning_y: 0

roi:

  x_offset: 0

  y_offset: 0

  height: 0

  width: 0

  do_rectify: false

---



depth intrinsic info topic



header:

  stamp:

    sec: 1762930594

    nanosec: 974354492

  frame_id: camera_depth_optical_frame

height: 480

width: 640

distortion_model: plumb_bob

d:

- 0.0

- 0.0

- 0.0

- 0.0

- 0.0

k:

- 395.630859375

- 0.0

- 324.56903076171875

- 0.0

- 395.630859375

- 242.35031127929688

- 0.0

- 0.0

- 1.0

r:

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

p:

- 395.630859375

- 0.0

- 324.56903076171875

- 0.0

- 0.0

- 395.630859375

- 242.35031127929688

- 0.0

- 0.0

- 0.0

- 1.0

- 0.0

binning_x: 0

binning_y: 0

roi:

  x_offset: 0

  y_offset: 0

  height: 0

  width: 0

  do_rectify: false

---

^C



ros2 topic echo /camera/camera/extrinsics/depth_to_color

1762930640.389163 [123]       ros2: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)

rotation:

- 0.9999944567680359

- 0.0004453109868336469

- -0.003304719226434827

- -0.00045781597145833075

- 0.9999927282333374

- -0.0037841906305402517

- 0.003303010016679764

- 0.0037856826093047857

- 0.9999873638153076

translation:

- -0.05908159539103508

- 1.4681237189506646e-05

- 0.00048153731040656567

---



ros2 topic echo /camera/camera/extrinsics/depth_to_depth

1762930669.043295 [123]       ros2: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)

rotation:

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

- 0.0

- 0.0

- 0.0

- 1.0

translation:

- 0.0

- 0.0

- 0.0

---




결 방안
이 문제들을 해결하기 위한 몇 가지 단계가 있습니다.

1. (가장 쉬운 해결책) Realsense의 'Aligned' 토픽 사용
Intel Realsense 카메라는 depth_to_color extrinsics를 이용해 Depth 이미지를 RGB 카메라 시점으로 변환(Warping)시킨 토픽을 이미 발행해 줄 가능성이 높습니다.

Depth 토픽 변경: depth_topic을 /camera/camera/depth/image_rect_raw (현재 값) 대신 /camera/camera/aligned_depth_to_color/image_raw로 변경합니다.

Intrinsics 통일: 이 aligned_depth_to_color 이미지는 RGB 카메라의 시점과 해상도(640x480)를 따릅니다. 따라서 depth_to_pointcloud_gpu 함수에서 사용하는 Intrinsics(fx, fy, cx, cy)는 **Depth 카메라 값(395.63...)이 아닌 RGB 카메라 값(385.97...)**을 사용해야 합니다.

정렬 코드 제거: RGB와 Aligned Depth는 이미 같은 시점과 해상도(640x480)를 가지므로, rgbd_callback의 Step 3. GPU 기반 정렬 (F.interpolate) 부분이 통째로 필요 없어집니다.

파라미터 수정: 코드의 depth_cam 및 rgb_cam 해상도와 Intrinsics 파라미터를 모두 실제 RGB 토픽(640x480, fx=385.97...) 값으로 수정합니다.

2. (Aligned 토픽이 없을 시) 수동 정렬 구현
만약 aligned_depth_to_color 토픽을 사용할 수 없다면, F.interpolate를 사용하는 대신 직접 3D-to-2D 프로젝션을 구현해야 합니다.

올바른 토픽 구독:

RGB: /camera/camera/color/image_rect_raw (왜곡 보정된 이미지)

Depth: /camera/camera/depth/image_rect_raw (왜곡 보정된 이미지)

올바른 파라미터 사용: 코드의 모든 Intrinsics와 해상도 값을 실제 ROS 토픽(640x480) 값으로 수정합니다.

정렬 로직 변경: F.interpolate 대신 다음 로직을 구현합니다.

a. depth_to_pointcloud_gpu를 (올바른 Depth Intrinsics로) 호출하여 camera_depth_optical_frame 기준 3D 포인트 P_d (H, W, 3)를 계산합니다.

b. depth_to_color Extrinsics 행렬 T_c_d를 apply_transform_gpu 함수에 적용하여 P_d를 camera_color_optical_frame 기준 3D 포인트 P_c로 변환합니다.

c. P_c (x, y, z)를 **RGB 카메라의 Intrinsics(P 행렬)**를 사용해 2D 픽셀 좌표 (u_c, v_c)로 프로젝션합니다.

d. (u_c, v_c) 좌표를 이용해 RGB 이미지와 시맨틱 마스크에서 해당 픽셀의 색상과 라벨을 샘플링합니다. (이 과정을 torch.nn.functional.grid_sample로 최적화할 수 있습니다.)

제 생각에는 해결 방안 1이 Realsense 카메라의 표준 기능을 활용하는 가장 효율적이고 정확한 방법입니다.



"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.affines import compose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters # 동기화를 위해
import torch
import torch.nn.functional as F # GPU 기반 이미지 리사이징 (Alignment)
import time
from collections import deque

# ============================================================================
# 의존성 코드 (Config)
# (제공된 'reconstruction_config.py' 내용을 여기에 붙여넣습니다)
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional, Dict

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
    use_semantic: bool = True
    model_type: str ="maskformer-cityscapes"
    
    custom_object_model_path: str = "models/dynamic_object/best_model2.pth.pth"
    custom_surface_model_path: str = "models/surface/surface_mask_best_lrup.pt.pth"

    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    maskformer_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    active_model_name: str = field(init=False)
    inference_size: int = field(init=False)
    custom_class_names: Dict[str, int] = field(init=False)
    
    # --- ⬇️ 수정된 부분 (640x480 실제 값 기준) ⬇️ ---
    depth_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        # K: [395.63, 0.0, 324.56, 0.0, 395.63, 242.35]
        fx=395.630859375, fy=395.630859375, cx=324.56903076171875, cy=242.35031127929688
    ))
    # RGB Intrinsics는 Rectified P 행렬 기준
    # P: [385.97, 0.0, 322.19, 0.0, 0.0, 385.46, 238.75, 0.0]
    rgb_intrinsics: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        fx=385.97442626953125, fy=385.46087646484375, cx=322.1943359375, cy=238.75344848632812
    ))
    
    # 중요: image_raw 대신 왜곡 보정된 image_rect_raw 사용
    depth_topic: str = '/camera/camera/depth/image_rect_raw'
    rgb_topic: str = '/camera/camera/color/image_raw' # <-- 수정됨
    # --- ⬆️ 수정된 부분 ⬆️ ---
    
    # pointcloud_topic: str = '/semantic_pointcloud' # 노드에서 재정의됨
    pointcloud_topic: str = '/pointcloud' # 노드에서 재정의됨
    # bev_topic: str = '/semantic_bev_map' # 노드에서 재정의됨
    bev_topic: str = '/bev_map' # 노드에서 재정의됨
    
    source_frame: str = 'camera_depth_optical_frame'
    target_frame: str = 'camera_link'
    downsample_y: int = 3 # 9
    downsample_x: int = 2 # 6
    sync_slop: float = 0.1
    use_gpu: bool = True
    def __post_init__(self):
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
# 의존성 코드 (Model)
# (제공된 'optimized_model.py' 내용을 여기에 붙여넣습니다)
# ============================================================================
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)
from torch.cuda.amp import autocast

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
        self.enable_half = (self.device.type == 'cuda')
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
        """Run semantic segmentation on RGB image (BGR OpenCV format)"""
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
        h_orig, w_orig = rgb_image.shape[:2]
        if (h_orig, w_orig) != self.inference_size_hw:
            rgb_image_resized = cv2.resize(
                rgb_image, self.inference_size_wh, interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_image_resized = rgb_image
        rgb_image_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with autocast(enabled=self.enable_half):
                logits = self.model(input_tensor)
            if self.enable_half:
                logits = logits.float()
        logits = F.interpolate(
            logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        pred_mask = torch.argmax(logits, dim=1).squeeze()
        return pred_mask.cpu().numpy().astype(np.uint8)
    def _predict_segformer(self, rgb_image):
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
        if self.enable_half:
            outputs.logits = outputs.logits.float()
        result = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h_orig, w_orig)]
        )[0]
        return result.cpu().numpy().astype(np.uint8)
    def _predict_maskformer(self, rgb_image):
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
# 🚀 메인 노드: SemanticPointCloudBEVNode
# ============================================================================

class SemanticPointCloudBEVNode(Node):
    """
    Depth, RGB 이미지를 동기화하여 수신하고,
    Semantic Segmentation을 수행한 뒤,
    GPU 가속을 통해 Semantic Point Cloud와 Semantic BEV Map을 발행하는 노드.
    (Extrinsics를 고려한 Projective Alignment 수행)
    """

    def __init__(self):
        super().__init__('semantic_pointcloud_bev_node')

        # --- 1. 기본 모듈 및 설정 로드 ---
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 제공된 ReconstructionConfig 사용
        self.config = ReconstructionConfig()
        
        self.get_logger().info(f'🚀 CUDA GPU 가속 활성화 (PyTorch, {self.device})')
        if not self.config.use_semantic:
            self.get_logger().warn('시맨틱 모드가 비활성화되었습니다. (config.use_semantic = False)')

        # 시맨틱 모델 로드
        self.semantic_model = SemanticModel(self.config, self.device, self.get_logger())

        # --- 2. ROS 파라미터 선언 (PCL + BEV + Semantic) ---
        
        # --- ⬇️ 수정된 부분 (640x480 및 image_rect_raw 기준) ⬇️ ---
        self.declare_parameter('depth_topic', self.config.depth_topic)
        self.declare_parameter('rgb_topic', self.config.rgb_topic) # /camera/camera/color/image_rect_raw
        self.declare_parameter('source_frame', self.config.source_frame)
        self.declare_parameter('target_frame', self.config.target_frame)
        self.declare_parameter('sync_slop', self.config.sync_slop)

        # Depth 카메라 내부 파라미터 (PCL 재구성용, 640x480)
        self.declare_parameter('depth_cam.fx', self.config.depth_intrinsics.fx) # 395.63
        self.declare_parameter('depth_cam.fy', self.config.depth_intrinsics.fy) # 395.63
        self.declare_parameter('depth_cam.cx', self.config.depth_intrinsics.cx) # 324.56
        self.declare_parameter('depth_cam.cy', self.config.depth_intrinsics.cy) # 242.35
        self.declare_parameter('depth_cam.height', 480)
        # self.declare_parameter('depth_cam.width', 640)
        self.declare_parameter('depth_cam.width', 848)

        # RGB 카메라 내부 파라미터 (Projective Alignment용, 640x480)
        self.declare_parameter('rgb_cam.fx', self.config.rgb_intrinsics.fx) # 385.97
        self.declare_parameter('rgb_cam.fy', self.config.rgb_intrinsics.fy) # 385.46
        self.declare_parameter('rgb_cam.cx', self.config.rgb_intrinsics.cx) # 322.19
        self.declare_parameter('rgb_cam.cy', self.config.rgb_intrinsics.cy) # 238.75
        self.declare_parameter('rgb_cam.height', 480)
        # self.declare_parameter('rgb_cam.width', 640)
        self.declare_parameter('rgb_cam.width', 848)
        # --- ⬆️ 수정된 부분 ⬆️ ---

        # Semantic Point Cloud 파라미터
        self.declare_parameter('semantic_pointcloud_topic', '/semantic_pointcloud')
        self.declare_parameter('pcl.downsample_y', self.config.downsample_y)
        self.declare_parameter('pcl.downsample_x', self.config.downsample_x)

        # Semantic BEV 파라미터
        self.declare_parameter('semantic_bev_topic', '/semantic_bev_map')
        self.declare_parameter('bev.z_min', 0.15)
        self.declare_parameter('bev.z_max', 1.0)
        self.declare_parameter('bev.resolution', 0.1)
        self.declare_parameter('bev.size_x', 30.0)
        self.declare_parameter('bev.size_y', 30.0)
        self.declare_parameter('bev.ignore_labels', [0, 10]) 

        # --- 3. 파라미터 값 할당 ---
        # PCL/BEV 공통
        depth_topic = self.get_parameter('depth_topic').value
        rgb_topic = self.get_parameter('rgb_topic').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        sync_slop = self.get_parameter('sync_slop').value

        # Depth 카메라 (PCL)
        self.fx_d = self.get_parameter('depth_cam.fx').value
        self.fy_d = self.get_parameter('depth_cam.fy').value
        self.cx_d = self.get_parameter('depth_cam.cx').value
        self.cy_d = self.get_parameter('depth_cam.cy').value
        self.depth_height = self.get_parameter('depth_cam.height').value
        self.depth_width = self.get_parameter('depth_cam.width').value
        self.depth_shape_hw = (self.depth_height, self.depth_width)

        # RGB 카메라 (Alignment)
        self.fx_rgb = self.get_parameter('rgb_cam.fx').value
        self.fy_rgb = self.get_parameter('rgb_cam.fy').value
        self.cx_rgb = self.get_parameter('rgb_cam.cx').value
        self.cy_rgb = self.get_parameter('rgb_cam.cy').value
        self.rgb_height = self.get_parameter('rgb_cam.height').value
        self.rgb_width = self.get_parameter('rgb_cam.width').value
        self.rgb_shape_hw = (self.rgb_height, self.rgb_width)

        # PCL 파라미터
        semantic_pointcloud_topic = self.get_parameter('semantic_pointcloud_topic').value
        self.downsample_y = self.get_parameter('pcl.downsample_y').value
        self.downsample_x = self.get_parameter('pcl.downsample_x').value

        # BEV 파라미터
        semantic_bev_topic = self.get_parameter('semantic_bev_topic').value
        self.z_min = self.get_parameter('bev.z_min').value
        self.z_max = self.get_parameter('bev.z_max').value
        self.resolution = self.get_parameter('bev.resolution').value
        self.size_x = self.get_parameter('bev.size_x').value
        self.size_y = self.get_parameter('bev.size_y').value
        self.bev_ignore_labels = self.get_parameter('bev.ignore_labels').value

        # BEV 그리드 설정
        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0

        # --- 4. ROS 통신 설정 ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile)
        rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=qos_profile)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, rgb_sub],
            queue_size=10,
            slop=sync_slop
        )
        self.sync.registerCallback(self.rgbd_callback)

        self.sem_pc_pub = self.create_publisher(PointCloud2, semantic_pointcloud_topic, qos_profile)
        self.sem_bev_pub = self.create_publisher(PointCloud2, semantic_bev_topic, qos_profile)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 5. Point Cloud 필드 정의 (PCL/BEV 공통) ---
        self.semantic_pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
        ]
        self.point_step_pcl = 20 # 4*4 + 4 = 20 bytes
        self.semantic_bev_fields = self.semantic_pointcloud_fields
        self.point_step_bev = self.point_step_pcl

        # --- 6. GPU 파라미터 초기화 ---
        self._init_gpu_parameters()
        self._init_semantic_colormap()

        # 성능 모니터링
        self.timings = {
            'total': deque(maxlen=50),
            'semantic': deque(maxlen=50),
            'align_gpu': deque(maxlen=50),
            'depth_to_pc': deque(maxlen=50),
            'transform': deque(maxlen=50),
            'pcl_pub': deque(maxlen=50),
            'bev_pub': deque(maxlen=50),
        }
        self.last_report_time = time.time()

        self.get_logger().info('✅ Semantic PointCloud + BEV Node initialized (Projective Alignment)')
        self.get_logger().info(f"  RGB Topic: {rgb_topic}")
        self.get_logger().info(f"  Depth Topic: {depth_topic}")
        self.get_logger().info(f"  PCL Topic: {semantic_pointcloud_topic}")
        self.get_logger().info(f"  BEV Topic: {semantic_bev_topic} (Label 필드 포함)")

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성 (콜백 함수 내 부하 감소)"""

        # 1. PCL 재구성을 위한 픽셀 그리드 (Depth 카메라 좌표계, 640x480)
        v, u = torch.meshgrid(
            torch.arange(self.depth_height, device=self.device, dtype=torch.float32),
            torch.arange(self.depth_width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self.u_grid_d = u
        self.v_grid_d = v
        self.fx_d_tensor = torch.tensor(self.fx_d, device=self.device, dtype=torch.float32)
        self.fy_d_tensor = torch.tensor(self.fy_d, device=self.device, dtype=torch.float32)
        self.cx_d_tensor = torch.tensor(self.cx_d, device=self.device, dtype=torch.float32)
        self.cy_d_tensor = torch.tensor(self.cy_d, device=self.device, dtype=torch.float32)

        # 2. Projective Alignment를 위한 파라미터 (RGB 카메라, 640x480)
        self.fx_rgb_tensor = torch.tensor(self.fx_rgb, device=self.device, dtype=torch.float32)
        self.fy_rgb_tensor = torch.tensor(self.fy_rgb, device=self.device, dtype=torch.float32)
        self.cx_rgb_tensor = torch.tensor(self.cx_rgb, device=self.device, dtype=torch.float32)
        self.cy_rgb_tensor = torch.tensor(self.cy_rgb, device=self.device, dtype=torch.float32)

        # 3. 고정된 Extrinsics (Depth -> Color)
        # 제공된 'depth_to_color' 토픽 값 (NumPy)
        rotation_flat = [
            0.9999944567680359, 0.0004453109868336469, -0.003304719226434827,
            -0.00045781597145833075, 0.9999927282333374, -0.0037841906305402517,
            0.003303010016679764, 0.0037856826093047857, 0.9999873638153076
        ]
        translation_vec = [-0.05908159539103508, 1.4681237189506646e-05, 0.00048153731040656567]
        
        T_color_from_depth_np = np.eye(4, dtype=np.float32)
        T_color_from_depth_np[:3, :3] = np.array(rotation_flat).reshape(3, 3)
        T_color_from_depth_np[:3, 3] = np.array(translation_vec)

        # GPU 텐서로 변환
        self.T_color_from_depth_gpu = torch.from_numpy(T_color_from_depth_np).to(self.device)
        self.get_logger().info('고정 Extrinsics (T_color_from_depth) GPU에 로드 완료')

        # 4. BEV 생성을 위한 파라미터 (GPU 텐서)
        self.z_min_t = torch.tensor(self.z_min, device=self.device, dtype=torch.float32)
        self.z_max_t = torch.tensor(self.z_max, device=self.device, dtype=torch.float32)
        self.resolution_t = torch.tensor(self.resolution, device=self.device, dtype=torch.float32)
        self.grid_origin_x_t = torch.tensor(self.grid_origin_x, device=self.device, dtype=torch.float32)
        self.grid_origin_y_t = torch.tensor(self.grid_origin_y, device=self.device, dtype=torch.float32)
        self.bev_ignore_labels_t = torch.tensor(self.bev_ignore_labels, device=self.device, dtype=torch.long)
        self.bev_packed_flat = torch.full(
            (self.cells_y * self.cells_x,), 0, device=self.device, dtype=torch.int64
        )
        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({self.depth_height}x{self.depth_width})')

    def _init_semantic_colormap(self):
        """시맨틱 라벨을 RGB로 변환하기 위한 GPU 컬러맵 생성"""
        num_classes = self.config.num_custom_classes
        cityscapes_palette = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
            [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
            [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ]
        colors = torch.zeros((num_classes, 3), dtype=torch.uint8, device=self.device)
        for i, (name, idx) in enumerate(self.config.custom_class_names.items()):
            if i < len(cityscapes_palette):
                colors[idx] = torch.tensor(cityscapes_palette[i], dtype=torch.uint8, device=self.device)
            else:
                r, g, b = (i * 50) % 255, (i * 90) % 255, (i * 120) % 255
                colors[idx] = torch.tensor([r, g, b], dtype=torch.uint8, device=self.device)
        colors[0] = torch.tensor([0, 0, 0], dtype=torch.uint8, device=self.device)
        self.semantic_colormap_gpu = colors
        self.get_logger().info(f'GPU 시맨틱 컬러맵 생성 완료 ({num_classes} classes)')


    # --- ⬇️ (핵심) 새로운 정렬 함수 ⬇️ ---
    def project_points_to_rgb_grid(self, points_in_color_frame):
        """
        3D 포인트(color frame 기준, H, W, 3)를 RGB 2D 픽셀 그리드로 프로젝션합니다.
        F.grid_sample을 위한 정규화된 좌표(-1 ~ 1)와 유효 마스크를 반환합니다.
        """
        X = points_in_color_frame[..., 0]
        Y = points_in_color_frame[..., 1]
        Z = points_in_color_frame[..., 2]

        # Z > 0 (카메라 앞) 인 포인트만 유효
        z_mask = Z > 1e-6
        # 0으로 나누는 것을 방지
        Z_safe = torch.where(z_mask, Z, 1e-6)

        # 3D -> 2D 프로젝션 (Intrinsics 적용)
        u = self.fx_rgb_tensor * X / Z_safe + self.cx_rgb_tensor
        v = self.fy_rgb_tensor * Y / Z_safe + self.cy_rgb_tensor

        # F.grid_sample을 위한 정규화 (0 ~ W-1) -> (-1 ~ 1)
        norm_u = (u / (self.rgb_width - 1.0)) * 2.0 - 1.0
        norm_v = (v / (self.rgb_height - 1.0)) * 2.0 - 1.0

        # (H, W, 2) 스택
        normalized_grid = torch.stack([norm_u, norm_v], dim=-1)

        # 유효 샘플링 마스크 (카메라 앞 + 이미지 프레임 내부)
        sampling_mask = z_mask & \
                        (norm_u >= -1.0) & (norm_u <= 1.0) & \
                        (norm_v >= -1.0) & (norm_v <= 1.0)

        return normalized_grid, sampling_mask
    # --- ⬆️ (핵심) 새로운 정렬 함수 ⬆️ ---



    def rgbd_callback(self, depth_msg, rgb_msg):
        """Depth, RGB 동시 수신 및 전체 GPU 파이프라인 처리"""
        t_start = time.perf_counter()

        try:
            # --- 1. 시맨틱 예측 (CPU/GPU) ---
            # RGB 메시지 변환 (BGR8, 640x480)
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            if self.config.use_semantic:
                t_sem_start = time.perf_counter()
                # 시맨틱 예측 (결과: H_rgb x W_rgb NumPy)
                semantic_mask_rgb_res = self.semantic_model.predict(rgb_image)
                self.timings['semantic'].append((time.perf_counter() - t_sem_start) * 1000)

                if semantic_mask_rgb_res is None:
                    self.get_logger().warn('시맨틱 마스크 생성 실패', throttle_duration_sec=1.0)
                    return
                
                # GPU 업로드
                mask_tensor_rgb_res = torch.from_numpy(semantic_mask_rgb_res).to(self.device)

            else:
                # 시맨틱 비활성화 시
                mask_tensor_rgb_res = None # 마스크 텐서 없음
                self.timings['semantic'].append(0.0) # 시맨틱 시간 0

            # --- 2. 데이터 GPU 업로드 ---
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding=depth_msg.encoding
            ).astype(np.float32)
            
            depth_tensor = torch.from_numpy(depth_image).to(self.device) / 1000.0 # mm -> m
            
            # RGB는 항상 업로드
            rgb_tensor_rgb_res = torch.from_numpy(rgb_image).to(self.device)

            # --- 3. 3D 재구성 (Depth Frame) ---
            t_depth_start = time.perf_counter()
            pointcloud_cam_depth_frame = self.depth_to_pointcloud_gpu(depth_tensor)
            self.timings['depth_to_pc'].append((time.perf_counter() - t_depth_start) * 1000)

            # --- 4. GPU 기반 정렬 (Projective Alignment) ---
            t_align_start = time.perf_counter()

            # 4.1. 3D 포인트 변환 (Depth Frame -> Color Frame)
            pointcloud_cam_color_frame = self.apply_transform_gpu(
                pointcloud_cam_depth_frame, self.T_color_from_depth_gpu
            )

            # 4.2. 3D -> 2D 프로젝션 (Color Frame -> RGB Image)
            normalized_uv_grid, sampling_mask = self.project_points_to_rgb_grid(
                pointcloud_cam_color_frame
            )
            
            # 4.3. grid_sample을 위한 텐서 준비
            # (1, 3, H_r, W_r) - Bilinear
            rgb_for_interp = rgb_tensor_rgb_res.permute(2, 0, 1).float().unsqueeze(0)
            # (1, H_d, W_d, 2)
            normalized_uv_for_grid_sample = normalized_uv_grid.unsqueeze(0)

            # 4.4. GPU 샘플링 (RGB는 항상 수행)
            rgb_aligned_interp = F.grid_sample(
                rgb_for_interp, normalized_uv_for_grid_sample, 
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            
            # --- ⬇️ 수정된 부분 (조건부 마스크 생성) ⬇️ ---
            if self.config.use_semantic and mask_tensor_rgb_res is not None:
                # (1, 1, H_r, W_r) - Nearest
                mask_for_interp = mask_tensor_rgb_res.float().unsqueeze(0).unsqueeze(0)
                mask_aligned_interp = F.grid_sample(
                    mask_for_interp, normalized_uv_for_grid_sample, 
                    mode='nearest', padding_mode='zeros', align_corners=False
                )
                mask_aligned = mask_aligned_interp.squeeze().long()
            else:
                # 시맨틱이 꺼진 경우, (H_d, W_d) 모양의 0 (배경) 텐서 생성
                mask_aligned = torch.zeros(
                    self.depth_shape_hw, 
                    device=self.device, 
                    dtype=torch.long
                )
            # --- ⬆️ 수정된 부분 ⬆️ ---

            # 4.5. 결과 텐서 (H_d, W_d)
            rgb_aligned_bgr = rgb_aligned_interp.squeeze().permute(1, 2, 0).to(torch.uint8)
            # mask_aligned는 위 조건문에서 항상 정의됨

            # 4.6. 유효하지 않은 포인트/샘플링 마스킹
            invalid_mask = (depth_tensor <= 0.01) | (~sampling_mask)
            
            rgb_aligned_bgr[invalid_mask] = 0
            mask_aligned[invalid_mask] = 0 # 0 = background
            pointcloud_cam_depth_frame[invalid_mask] = 0.0 # Z=0 -> invalid

            self.timings['align_gpu'].append((time.perf_counter() - t_align_start) * 1000)

            # --- 5. TF 조회 (CPU) 및 좌표 변환 (GPU) ---
            t_tf_start = time.perf_counter()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, rclpy.time.Time()
            )
            transform_matrix = self.transform_to_matrix(transform)
            
            transformed_cloud = self.apply_transform_gpu(pointcloud_cam_depth_frame, transform_matrix)
            self.timings['transform'].append((time.perf_counter() - t_tf_start) * 1000)

            # --- 6. 메시지 발행 (PCL, BEV) ---
            stamp = depth_msg.header.stamp

            t_pcl_start = time.perf_counter()
            self.process_and_publish_semantic_pointcloud(
                transformed_cloud, rgb_aligned_bgr, mask_aligned, stamp
            )
            self.timings['pcl_pub'].append((time.perf_counter() - t_pcl_start) * 1000)

            t_bev_start = time.perf_counter()
            self.process_and_publish_semantic_bev(
                transformed_cloud, mask_aligned, stamp
            )
            self.timings['bev_pub'].append((time.perf_counter() - t_bev_start) * 1000)

            # --- 7. 타이밍 기록 ---
            self.timings['total'].append((time.perf_counter() - t_start) * 1000)
            self._report_stats()

        except TransformException as e:
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Semantic PCL/BEV 처리 오류: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())





    def depth_to_pointcloud_gpu(self, depth_tensor):
        """GPU를 이용한 Depth to Point Cloud 변환 (카메라 좌표계)"""
        z = depth_tensor
        x = (self.u_grid_d - self.cx_d_tensor) * z / self.fx_d_tensor
        y = (self.v_grid_d - self.cy_d_tensor) * z / self.fy_d_tensor
        return torch.stack([x, y, z], dim=-1) # (H, W, 3)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # 매트릭스가 NumPy라면 GPU 텐서로 변환
        if isinstance(matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        else:
            matrix_tensor = matrix # 이미 GPU 텐서 (e.g., T_color_from_depth_gpu)

        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        return transformed[:, :3].reshape(original_shape)

    def transform_to_matrix(self, transform):
        """ROS Transform 메시지를 4x4 동차 변환 행렬(NumPy)로 변환"""
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        r = transform.transform.rotation
        quat = [r.w, r.x, r.y, r.z] # transforms3d (w, x, y, z) 순서
        rotation_matrix = quat2mat(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def process_and_publish_semantic_pointcloud(
        self, transformed_cloud, rgb_aligned_bgr, mask_aligned, stamp
    ):
        """Semantic 3D 포인트 클라우드를 다운샘플링, 패킹 후 발행"""

        # 1. 다운샘플링 (GPU)
        points = transformed_cloud[::self.downsample_y, ::self.downsample_x, :]
        colors_bgr = rgb_aligned_bgr[::self.downsample_y, ::self.downsample_x, :]
        labels = mask_aligned[::self.downsample_y, ::self.downsample_x]

        # 2. Flatten (GPU)
        points_flat = points.reshape(-1, 3)
        colors_flat_bgr = colors_bgr.reshape(-1, 3)
        labels_flat = labels.reshape(-1)

        # 3. 유효한 포인트 필터링 (Z > 0)
        # (invalid_mask에서 Z=0으로 설정했으므로, 0.01보다 큰 것만 유효)
        valid_mask = points_flat[:, 2] > 0.01 
        
        points_valid = points_flat[valid_mask]
        colors_valid_bgr = colors_flat_bgr[valid_mask]
        labels_valid = labels_flat[valid_mask]

        num_points = points_valid.shape[0]
        if num_points == 0:
            return

        # 4. RGB 패킹 (GPU)
        r = colors_valid_bgr[:, 2].long()
        g = colors_valid_bgr[:, 1].long()
        b = colors_valid_bgr[:, 0].long()
        
        rgb_packed_gpu = (r << 16) | (g << 8) | b
        rgb_float32_gpu = rgb_packed_gpu.to(torch.uint32).view(torch.float32)

        # 5. Label 패킹 (GPU)
        labels_float32_gpu = labels_valid.long().to(torch.uint32).view(torch.float32)

        # 6. (X, Y, Z, RGB, Label) 데이터 결합 (GPU)
        data_gpu = torch.stack(
            [
                points_valid[:, 0], 
                points_valid[:, 1], 
                points_valid[:, 2], 
                rgb_float32_gpu, 
                labels_float32_gpu
            ],
            dim=-1 # (N, 5)
        )

        # 7. GPU -> CPU 전송
        data_np = data_gpu.cpu().numpy()

        # 8. PointCloud2 메시지 생성 (CPU)
        pointcloud_msg = self._create_semantic_cloud_from_data(
            data_np, stamp, self.target_frame
        )

        # 9. 발행
        self.sem_pc_pub.publish(pointcloud_msg)
    
    def process_and_publish_semantic_bev(
        self, transformed_cloud, mask_aligned, stamp
    ):
        """
        'transformed_cloud' (H, W, 3)와 'mask_aligned' (H, W) GPU 텐서를 사용하여
        GPU에서 Semantic BEV 맵을 생성하고 발행합니다. (Label 필드 포함)
        """

        # 1. Flatten (GPU)
        x_flat = transformed_cloud[..., 0].ravel()
        y_flat = transformed_cloud[..., 1].ravel()
        z_flat = transformed_cloud[..., 2].ravel()
        labels_flat = mask_aligned.ravel().long()

        # 2. Z-필터 마스크 (GPU)
        mask = (z_flat > self.z_min_t) & (z_flat < self.z_max_t)

        # --- ⬇️ 수정된 부분 (조건부 시맨틱 필터링) ⬇️ ---
        # 3. 시맨틱 필터 마스크 (GPU)
        if self.config.use_semantic:
            ignore_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
            for label in self.bev_ignore_labels: 
                ignore_mask |= (labels_flat == label)
            
            mask &= ~ignore_mask
        # --- ⬆️ 수정된 부분 ⬆️ ---

        # 4. 월드 좌표 -> 그리드 인덱스 변환 (GPU)
        grid_c = ((x_flat - self.grid_origin_x_t) / self.resolution_t).long()
        grid_r = ((y_flat - self.grid_origin_y_t) / self.resolution_t).long()

        # 5. 바운더리 체크 마스크 (GPU)
        mask &= (grid_c >= 0) & (grid_c < self.cells_x) & \
                (grid_r >= 0) & (grid_r < self.cells_y)

        # 6. 유효한 포인트만 필터링 (GPU)
        valid_z = z_flat[mask]
        if valid_z.shape[0] == 0:
            return

        valid_labels = labels_flat[mask]
        valid_r = grid_r[mask]
        valid_c = grid_c[mask]

        # 7. 2D 인덱스 -> 1D 선형 인덱스 (GPU)
        linear_indices = valid_r * self.cells_x + valid_c

        # 8. 데이터 패킹 (GPU)
        z_shifted = (valid_z * 1000.0).long() << 16
        packed_data = z_shifted | valid_labels 
        
        # 9. "Highest Point Wins" (GPU Scatter-Max)
        self.bev_packed_flat.fill_(0)
        self.bev_packed_flat.index_reduce_(
            dim=0,
            index=linear_indices,
            source=packed_data,
            reduce="amax",
            include_self=False
        )

        # 10. 유효한 셀만 추출 (GPU)
        valid_bev_mask = self.bev_packed_flat > 0
        valid_indices_flat = torch.where(valid_bev_mask)[0]
        if valid_indices_flat.shape[0] == 0:
            return
        packed_values = self.bev_packed_flat[valid_bev_mask]

        # 11. 데이터 언패킹 (GPU)
        height_values_mm = packed_values >> 16
        label_values = (packed_values & 0xFFFF).long() 
        height_values = height_values_mm.float() / 1000.0

        # 12. 1D 인덱스 -> 2D 인덱스 -> 월드 좌표 (GPU)
        r_idx_bev = torch.div(valid_indices_flat, self.cells_x, rounding_mode='floor')
        c_idx_bev = valid_indices_flat % self.cells_x

        x_world = self.grid_origin_x_t + (c_idx_bev.float() + 0.5) * self.resolution_t
        y_world = self.grid_origin_y_t + (r_idx_bev.float() + 0.5) * self.resolution_t
        z_world = height_values 

        # 13. 라벨 -> RGB 색상 변환 (GPU)
        # (use_semantic=False이면 label_values가 모두 0이므로, 컬러맵 0번(검은색)이 조회됨)
        rgb_float32_gpu = self._label_to_color_gpu(label_values)
        labels_float32_gpu = label_values.long().to(torch.uint32).view(torch.float32)

        # 14. (X, Y, Z, RGB, Label) 데이터 결합 (GPU)
        bev_data_gpu = torch.stack(
            [x_world, y_world, z_world, rgb_float32_gpu, labels_float32_gpu],
            dim=-1 # (N, 5)
        )

        # 15. GPU -> CPU 전송
        bev_data_np = bev_data_gpu.cpu().numpy()

        # 16. PointCloud2 메시지 생성 (CPU)
        bev_msg = self._create_semantic_cloud_from_data(
            bev_data_np, stamp, self.target_frame
        )

        # 17. 발행
        self.sem_bev_pub.publish(bev_msg)


    def _label_to_color_gpu(self, labels):
        """GPU 시맨틱 라벨 텐서(long)를 패킹된 float32 RGB 텐서로 변환"""
        colors_uint8 = self.semantic_colormap_gpu[labels]

        r = colors_uint8[:, 0].long()
        g = colors_uint8[:, 1].long()
        b = colors_uint8[:, 2].long()

        rgb_packed_gpu = (r << 16) | (g << 8) | b
        return rgb_packed_gpu.to(torch.uint32).view(torch.float32)

    def _create_semantic_cloud_from_data(self, data_np, stamp, frame_id):
        """
        (N, 5) [x, y, z, rgb_float32, label_float32] NumPy 배열로
        Semantic PointCloud2 메시지를 생성합니다. (PCL과 BEV 공통)
        """
        header = Header(stamp=stamp, frame_id=frame_id)
        num_points = data_np.shape[0]
        
        # PCL/BEV 모두 5개 필드 (x,y,z,rgb,label) 사용
        fields = self.semantic_pointcloud_fields
        point_step = self.point_step_pcl

        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=point_step * num_points,
            data=data_np.astype(np.float32).tobytes(),
            is_dense=True,
        )

    def _report_stats(self):
        """성능 통계 출력"""
        if time.time() - self.last_report_time < 2.0: # 2초마다
            return
            
        if not self.timings['total']:
            return

        avg_total = np.mean(self.timings['total'])
        fps = 1000.0 / avg_total
        avg_sem = np.mean(self.timings['semantic'])
        avg_align = np.mean(self.timings['align_gpu'])
        avg_depth = np.mean(self.timings['depth_to_pc'])
        avg_tf = np.mean(self.timings['transform'])
        avg_pcl = np.mean(self.timings['pcl_pub'])
        avg_bev = np.mean(self.timings['bev_pub'])

        msg = f"\n📊 [SemanticPCL-BEV] FPS: {fps:.1f} Hz (Total: {avg_total:.1f} ms)\n" \
              f"  ├─ Semantic : {avg_sem:6.1f} ms\n" \
              f"  ├─ Depth→PC : {avg_depth:6.1f} ms\n" \
              f"  ├─ Align GPU: {avg_align:6.1f} ms (Projective)\n" \
              f"  ├─ Transform: {avg_tf:6.1f} ms\n" \
              f"  ├─ PCL Pub  : {avg_pcl:6.1f} ms\n" \
              f"  └─ BEV Pub  : {avg_bev:6.1f} ms"
        
        self.get_logger().info(msg)
        self.last_report_time = time.time()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    node = SemanticPointCloudBEVNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
