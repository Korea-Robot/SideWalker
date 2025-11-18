"""
Config for NPU Semantic PointCloud Node
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    height: int
    width: int

@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str = "./yolov9c-seg.mxq"
    conf_thres: float = 0.5
    iou_thres: float = 0.4

@dataclass
class ROSConfig:
    """ROS topic and frame configuration"""
    depth_topic: str = '/camera/camera/depth/image_rect_raw'
    rgb_topic: str = '/camera/camera/color/image_raw'
    source_frame: str = 'camera_depth_optical_frame'
    target_frame: str = 'camera_link'
    sync_slop: float = 0.1
    semantic_pointcloud_topic: str = '/semantic_pointcloud'


@dataclass
class DepthCameraConfig:
    """Depth camera parameters"""
    fx: float = 395.630859375
    fy: float = 395.630859375
    cx: float = 324.56903076171875
    cy: float = 242.35031127929688
    height: int = 480
    width: int = 848


@dataclass
class RGBCameraConfig:
    """RGB camera parameters"""
    fx: float = 385.97442626953125
    fy: float = 385.46087646484375
    cx: float = 322.1943359375
    cy: float = 238.75344848632812
    height: int = 480
    width: int = 848


@dataclass
class PointCloudConfig:
    """Point cloud processing parameters"""
    downsample_y: int = 3
    downsample_x: int = 2


@dataclass
class SemanticConfig:
    """Semantic segmentation config"""
    num_labels: int = 92  # COCO dataset classes


# COCO Class definitions
COCO_CLASS_TO_IDX = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eye glasses',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet',
    71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
    81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush',
    91: 'hair brush'
}


# Default configuration instance
class Config:
    """Main configuration container"""
    def __init__(self):
        self.model = ModelConfig()
        self.ros = ROSConfig()
        self.depth_cam = DepthCameraConfig()
        self.rgb_cam = RGBCameraConfig()
        self.pointcloud = PointCloudConfig()
        self.semantic = SemanticConfig()