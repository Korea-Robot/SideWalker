#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 설정 변수 ---
# 1. Realsense 카메라 이미지 토픽 이름 (이 부분만 수정하시면 됩니다)
REALSENSE_TOPIC = "/camera/camera/color/image_raw" 

# /camera/camera/color/image_raw

# 2. 학습된 모델 가중치 파일 경로
MODEL_PATH = "ckpts/best_seg_model.pth"

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
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):   
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

def create_color_palette():
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

class SemanticInferenceNode:
    def __init__(self):
        """ROS 노드 및 모델 초기화"""
        rospy.init_node('realsense_surface_inference_node', anonymous=True)
        rospy.loginfo("Realsense Surface Inference 노드 시작")

        self.bridge = CvBridge()
        self.color_palette = create_color_palette()

        # 모델 로드
        self.model = self.load_model()
        rospy.loginfo(f"'{MODEL_PATH}'에서 모델 로드 완료. {DEVICE}에서 추론을 실행합니다.")

        # 이미지 전처리기 (학습 스크립트와 동일한 정규화 사용)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # ROS Subscriber 생성
        self.image_sub = rospy.Subscriber(REALSENSE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)

    def load_model(self):
        """DirectSegFormer 모델을 로드하고 state_dict를 적용"""
        model = DirectSegFormer(num_classes=NUM_LABELS)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # 추론 모드로 설정
        return model

    def preprocess_image(self, cv_image):
        """OpenCV 이미지를 모델 입력 텐서로 변환"""
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 텐서로 변환 및 정규화
        input_tensor = self.transform(rgb_image)
        
        # 배치 차원 추가 및 디바이스로 전송
        return input_tensor.unsqueeze(0).to(DEVICE)

    def image_callback(self, msg):
        """이미지 토픽을 수신하고 추론 및 시각화를 수행하는 콜백 함수"""
        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_h, original_w, _ = cv_image.shape
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 1. 이미지 전처리
        input_tensor = self.preprocess_image(cv_image)

        # 2. 추론 수행 (Gradient 계산 비활성화)
        with torch.no_grad():
            # DirectSegFormer는 logits를 직접 반환
            logits = self.model(input_tensor)

        # 3. 결과 후처리
        # 로짓을 원본 이미지 크기로 업샘플링
        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        # 가장 확률이 높은 클래스로 예측 마스크 생성
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()
        
        # NumPy 배열로 변환
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # 4. 시각화
        # 세그멘테이션 마스크에 색상 입히기
        segmentation_image = self.color_palette[pred_mask_np]

        # 원본 이미지와 마스크를 오버레이
        overlay_image = cv2.addWeighted(cv_image, 0.6, segmentation_image, 0.4, 0)

        # 5. 결과 화면에 표시
        cv2.imshow("Segment-only Result", segmentation_image)
        cv2.imshow("Overlay Result", overlay_image)
        
        # 윈도우가 업데이트되도록 1ms 대기
        cv2.waitKey(1)

def main():
    try:
        SemanticInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("노드를 종료하고 모든 창을 닫습니다.")

if __name__ == '__main__':
    main()
