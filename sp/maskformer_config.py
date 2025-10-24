# maskformer config.py
# 384,384 best 

from dataclasses import dataclass, field
from typing import List

@dataclass
class ImageSizeConfig:
    """이미지 크기 설정을 위한 중첩 dataclass"""
    height: int
    width: int

@dataclass
class ImageProcessorConfig:
    """
    제공된 JSON 설정을 기반으로 한
    Image Processor 설정 dataclass
    """
    _max_size: int = 2048
    do_normalize: bool = True
    do_rescale: bool = True
    do_resize: bool = True
    ignore_index: int = 255
    
    # 리스트(mutable) 타입은 default_factory를 사용해야 합니다.
    image_mean: List[float] = field(
        default_factory=lambda: [
            0.48500001430511475,
            0.4560000002384186,
            0.4059999883174896
        ]
    )
    
    image_processor_type: str = "Mask2FormerImageProcessor"
    
    image_std: List[float] = field(
        default_factory=lambda: [
            0.2290000021457672,
            0.2239999920129776,
            0.22499999403953552
        ]
    )
    
    num_labels: int = 19
    reduce_labels: bool = False
    resample: int = 2
    rescale_factor: float = 0.00392156862745098
    
    # 중첩 dataclass 객체도 default_factory를 사용합니다.
    size: ImageSizeConfig = field(
        default_factory=lambda: ImageSizeConfig(height=384, width=384)
    )
    
    size_divisor: int = 32

# --- 사용 예시 ---
# config = ImageProcessorConfig()
# print(config)
# print(f"Image mean: {config.image_mean}")
# print(f"Size: {config.size.height}x{config.size.width}")

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CityscapesLabelsConfig:
    """Cityscapes id2label 매핑을 위한 dataclass"""
    
    id2label: Dict[str, str] = field(default_factory=lambda: {
        "0": "road",
        "1": "sidewalk",
        "2": "building",
        "3": "wall",
        "4": "fence",
        "5": "pole",
        "6": "traffic light",
        "7": "traffic sign",
        "8": "vegetation",
        "9": "terrain",
        "10": "sky",
        "11": "person",
        "12": "rider",
        "13": "car",
        "14": "truck",
        "15": "bus",
        "16": "train",
        "17": "motorcycle",
        "18": "bicycle"
    })

# --- 사용 예시 ---
# label_config = CityscapesLabelsConfig()
# print(label_config.id2label['11'])  # 'person' 출력
