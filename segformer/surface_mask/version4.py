#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 설정 변수 ---
# 1. Realsense 카메라 이미지 토픽 이름
REALSENSE_TOPIC = "/camera/camera/color/image_raw"

# 2. 학습된 모델 가중치 파일 경로
MODEL_PATH = "surface_mask_best.pt"
MODEL_PATH = "surface_mask_best_lrup.pt"


# 3. 추론에 사용할 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 학습 스크립트에서 가져온 클래스 정보 ---
CLASS_TO_IDX = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
}
NUM_LABELS = len(CLASS_TO_IDX)

# --- 학습 스크립트에서 가져온 모델 클래스 정의 ---
class DirectSegFormer(nn.Module):
    """학습 시 사용된 모델과 동일한 구조의 클래스"""
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        try:
            self.original_model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        except ValueError as e:
            if "torch.load" in str(e):
                # PyTorch 버전 문제 우회: 로컬에서 모델 생성 후 가중치만 로드
                print(f"Warning: {e}")
                print("Creating model architecture without pretrained weights...")
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(pretrained_model_name)
                config.num_labels = num_classes
                self.original_model = SegformerForSemanticSegmentation(config)
            else:
                raise e
        
    def forward(self, x):   
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

class SegformerViewerNode(Node):
    def __init__(self):
        super().__init__('segformer_viewer_node')
        
        # 1) Set the device
        self.device = DEVICE
        self.get_logger().info(f"Using device: {self.device}")

        # 2) Load DirectSegFormer model from checkpoint
        self.model = self.load_model()
        self.get_logger().info(f"Model loaded from '{MODEL_PATH}'")

        # 3) Image preprocessing pipeline (학습 스크립트와 동일한 정규화 사용)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 4) Create a color palette for visualization
        self.color_palette = self.create_color_palette()

        # ROS 2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            REALSENSE_TOPIC,
            self.image_callback,
            10)
        
        self.get_logger().info('DirectSegFormer viewer node has been started. Waiting for images...')

    def load_model(self):
        """DirectSegFormer 모델을 로드하고 state_dict를 적용"""
        model = DirectSegFormer(num_classes=NUM_LABELS)
        
        try:
            # 체크포인트 로드 (weights_only=False로 시도)
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            
            # 키 이름 매핑 처리
            new_state_dict = {}
            for key, value in checkpoint.items():
                # 체크포인트의 키가 "segformer."로 시작하면 "original_model." 접두사 추가
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # 매핑된 state_dict로 모델 로드
            model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info("Model loaded successfully with key mapping")
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")
            self.get_logger().warn("Using model without pretrained weights")
        
        model.to(self.device)
        model.eval()  # 추론 모드로 설정
        return model

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성 (OpenCV BGR 형식)"""
        # Matplotlib의 'jet' 컬러맵 사용
        cmap = plt.cm.get_cmap('jet', NUM_LABELS)
        palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
        
        for i in range(NUM_LABELS):
            if i == 0:  # 배경은 검은색
                palette[i] = [0, 0, 0]
                continue
            # RGBA (0-1) -> BGR (0-255) 변환
            rgba = cmap(i)
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            palette[i] = bgr
        return palette

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 텐서로 변환 및 정규화
        input_tensor = self.transform(rgb_image)
        
        # 배치 차원 추가 및 디바이스로 전송
        return input_tensor.unsqueeze(0).to(self.device)

    def image_callback(self, msg):
        """Callback function for processing image messages and displaying the result."""
        try:
            # Convert ROS Image message to an OpenCV image
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return

        original_h, original_w, _ = img_bgr.shape

        # 1) 이미지 전처리
        input_tensor = self.preprocess_image(img_bgr)

        # 2) 추론 수행
        with torch.no_grad():
            # DirectSegFormer는 logits를 직접 반환
            logits = self.model(input_tensor)

        # 3) 결과 후처리
        # 로짓을 원본 이미지 크기로 업샘플링
        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 가장 확률이 높은 클래스로 예측 마스크 생성
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # 4) 시각화
        # 세그멘테이션 마스크에 색상 입히기
        segmentation_image = self.color_palette[pred_mask_np]

        # 원본 이미지와 마스크를 오버레이
        overlay = cv2.addWeighted(img_bgr, 0.6, segmentation_image, 0.4, 0)
        
        # Stack the original and segmented images side-by-side for comparison
        images_to_show = np.hstack((img_bgr, overlay))

        # 5) Display the images
        cv2.imshow("DirectSegFormer Segmentation Viewer", images_to_show)
        
        # Check for key press to close the window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            rclpy.shutdown()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    segformer_viewer_node = SegformerViewerNode()
    
    # Use a try-finally block to ensure cv2 windows are closed
    try:
        rclpy.spin(segformer_viewer_node)
    finally:
        # Cleanup
        segformer_viewer_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

"""
문제가 명확해졌습니다. 저장된 체크포인트의 키 이름과 현재 모델 구조의 키 이름이 맞지 않습니다.
체크포인트는 segformer.encoder...로 시작하는데, 현재 모델은 original_model.segformer.encoder...를 기대하고 있습니다. 이는 학습할 때와 추론할 때 모델 래핑 방식이 다르기 때문입니다.
이를 해결하기 위해 키 매핑을 처리하는 코드로 수정하겠습니다:ROS2 DirectSegFormer Inference NodeCode ∙ Version 4     def load_model(self):
        #DirectSegFormer 모델을 로드하고 state_dict를 적용
        model = DirectSegFormer(num_classes=NUM_LABELS)

        try:
            # 체크포인트 로드 (weights_only=False로 시도)
            checkpoint = torch.load(MODEL_PA이제 체크포인트의 키 이름을 올바르게 매핑하도록 수정했습니다. 주요 변경사항:

키 매핑 처리: 체크포인트에서 segformer.나 decode_head.로 시작하는 키에 original_model. 접두사를 추가합니다.
안전한 로딩: weights_only=False로 설정하여 torch.load 제한을 우회합니다.
Strict=False: 일부 키가 맞지 않을 수 있으므로 strict 모드를 해제합니다.

이렇게 수정하면 다음과 같은 매핑이 이루어집니다:

segformer.encoder.patch_embeddings.0.proj.weight → original_model.segformer.encoder.patch_embeddings.0.proj.weight
decode_head.linear_c.0.proj.weight → original_model.decode_head.linear_c.0.proj.weight


학습할 때와 추론할 때 모델 래핑 방식이 다른 이유는 몇 가지가 있습니다:
1. 학습 시 직접 모델 저장
학습 코드에서는 DirectSegFormer 클래스를 정의하고, 이 클래스 내부의 original_model이 실제 SegformerForSemanticSegmentation입니다.
하지만 모델을 저장할 때 전체 DirectSegFormer 객체를 저장하지 않고, original_model의 state_dict만 저장했을 가능성이 높습니다:
python# 학습 코드에서 이렇게 저장했을 것으로 추정:
torch.save(model.original_model.state_dict(), "best_seg_model.pth")
# 또는
torch.save(model.state_dict(), "best_seg_model.pth")  # DirectSegFormer 전체
2. 키 이름 불일치의 원인들
A) 학습 시 original_model만 저장한 경우
python# 저장: model.original_model.state_dict()
# 키 이름: "segformer.encoder.xxx", "decode_head.xxx"

# 로드: DirectSegFormer().load_state_dict()
# 기대하는 키: "original_model.segformer.encoder.xxx"
B) 다른 라이브러리나 프레임워크 사용
학습 시 MMSegmentation, Lightning, 또는 다른 래핑 프레임워크를 사용했다면 모델 구조가 달라질 수 있습니다.
C) 모델 정의 차이
학습할 때 사용한 모델 클래스와 추론할 때 사용하는 모델 클래스가 미묘하게 다를 수 있습니다.
3. 해결 방법들
가장 정확한 해결책을 찾기 위해 체크포인트를 분석해보세요:
pythonimport torch

# 체크포인트 구조 확인
checkpoint = torch.load("ckpts/best_seg_model.pth", map_location="cpu", weights_only=False)
print("체크포인트 타입:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\n상위 키들:")
    for key in list(checkpoint.keys())[:10]:
        print(f"  {key}")

    # 모델 구조가 중첩되어 있는지 확인
    if 'model' in checkpoint:
        print("\n'model' 키 내부:")
        for key in list(checkpoint['model'].keys())[:5]:
            print(f"  {key}")
이 분석 결과에 따라 정확한 로딩 방법을 결정할 수 있습니다. 학습 코드를 보면 더 정확하게 알 수 있겠지만, 일반적으로 이런 불일치는 모델 저장/로드 방식의 차이에서 발생합니다.

"""
