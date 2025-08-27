#!/usr/bin/env python3
"""
RealSense 카메라를 이용한 실시간 세그멘테이션 추론 및 시각화
OpenCV로 직접 시각화
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import logging
import time
from datetime import datetime
import threading
import queue

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealSenseSegmentation:
    def __init__(self, args):
        self.args = args
        
        # 클래스 정보 설정
        self.setup_classes()
        
        # 모델 로드
        self.load_model()
        
        # 색상 맵 생성
        self.colormap = self.create_colormap(self.num_labels)
        
        # ROS2 관련 설정
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=5)
        self.latest_image = None
        
        # 성능 측정 변수
        self.frame_count = 0
        self.total_inference_time = 0
        self.start_time = time.time()
        
        # 시각화 설정
        self.setup_visualization()
        
        logger.info("RealSense Segmentation initialized successfully!")
    
    def setup_classes(self):
        """클래스 정보 설정"""
        self.class_to_idx = {
            'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
            'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
            'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
            'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
            'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
            'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
            'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
        }
        
        self.id2label = {int(idx): label for label, idx in self.class_to_idx.items()}
        self.label2id = self.class_to_idx
        self.num_labels = len(self.id2label)
        
        logger.info(f"Number of classes: {self.num_labels}")
    
    def load_model(self):
        """학습된 모델 로드"""
        try:
            # 디바이스 설정
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # 모델 생성
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.args.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )
            
            # 학습된 가중치 로드
            if os.path.exists(self.args.model_path):
                checkpoint = torch.load(self.args.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model weights loaded from: {self.args.model_path}")
            else:
                logger.warning(f"Model file not found: {self.args.model_path}")
                logger.info("Using pretrained model without fine-tuned weights")
            
            self.model.to(self.device)
            self.model.eval()
            
            # 이미지 전처리기 설정
            self.processor = SegformerImageProcessor.from_pretrained(self.args.model_name)
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def create_colormap(self, num_classes):
        """클래스별 고유 색상 생성 (BGR 순서로)"""
        colors = []
        for i in range(num_classes):
            # HSV 색공간에서 균등하게 분포된 색상 생성
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            
            # HSV를 RGB로 변환
            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c
            
            if hue < 1/6:
                r, g, b = c, x, 0
            elif hue < 2/6:
                r, g, b = x, c, 0
            elif hue < 3/6:
                r, g, b = 0, c, x
            elif hue < 4/6:
                r, g, b = 0, x, c
            elif hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # BGR 순서로 저장 (OpenCV용)
            colors.append([int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)])
        
        # 배경은 검은색으로
        colors[0] = [0, 0, 0]
        
        return np.array(colors, dtype=np.uint8)
    
    def setup_visualization(self):
        """OpenCV 시각화 창 설정"""
        if self.args.show_window:
            # 창 생성 및 크기 설정
            cv2.namedWindow('Segmentation Results', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Segmentation Results', 1200, 400)
            
            # 클래스 정보 창
            if self.args.show_class_info:
                cv2.namedWindow('Class Information', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Class Information', 300, 600)
                self.create_class_info_image()
    
    def create_class_info_image(self):
        """클래스 정보를 보여주는 이미지 생성"""
        height = 30 * self.num_labels + 50
        width = 300
        info_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 제목
        cv2.putText(info_img, 'Classes', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 각 클래스별 정보
        for i, (class_id, class_name) in enumerate(self.id2label.items()):
            y = 60 + i * 25
            
            # 색상 박스
            color = self.colormap[class_id].tolist()
            cv2.rectangle(info_img, (10, y-10), (30, y+5), color, -1)
            
            # 클래스 이름
            cv2.putText(info_img, f"{class_id}: {class_name}", (35, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.class_info_image = info_img
    
    def preprocess_image(self, cv_image):
        """이미지 전처리"""
        # RGB로 변환
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # SegformerImageProcessor를 사용한 전처리
        inputs = self.processor(rgb_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values, rgb_image
    
    def predict(self, pixel_values, original_size):
        """모델 추론"""
        with torch.no_grad():
            start_time = time.time()
            
            # 추론 실행
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # 원본 크기로 업샘플링
            upsampled_logits = F.interpolate(
                logits,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            
            # 예측 결과
            predictions = torch.argmax(upsampled_logits, dim=1)
            
            # 확률 맵 (소프트맥스 적용)
            probabilities = F.softmax(upsampled_logits, dim=1)
            
            inference_time = time.time() - start_time
            
            return predictions[0].cpu().numpy(), probabilities[0].cpu().numpy(), inference_time
    
    def create_segmentation_visualization(self, original_image, segmentation_mask):
        """세그멘테이션 결과를 컬러로 변환"""
        # 세그멘테이션 마스크를 컬러로 변환
        height, width = segmentation_mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(self.num_labels):
            mask = (segmentation_mask == class_id)
            colored_mask[mask] = self.colormap[class_id]
        
        return colored_mask
    
    def create_overlay(self, original_image, segmentation_mask, alpha=0.6):
        """원본 이미지와 세그멘테이션 결과를 오버레이"""
        # 세그멘테이션 마스크를 컬러로 변환
        colored_mask = self.create_segmentation_visualization(original_image, segmentation_mask)
        
        # 배경 영역 (클래스 0) 제외
        background_mask = (segmentation_mask == 0)
        colored_mask[background_mask] = [0, 0, 0]
        
        # 오버레이 생성
        overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
        
        return overlay, colored_mask
    
    def create_combined_visualization(self, original_image, segmentation_mask, inference_time, fps):
        """3개 이미지를 하나로 합친 시각화 생성"""
        # 오버레이 및 컬러 마스크 생성
        overlay, colored_mask = self.create_overlay(original_image, segmentation_mask)
        
        # 이미지 크기 조정 (너무 크면 축소)
        h, w = original_image.shape[:2]
        if w > 400:
            scale = 400 / w
            new_w, new_h = int(w * scale), int(h * scale)
            original_image = cv2.resize(original_image, (new_w, new_h))
            colored_mask = cv2.resize(colored_mask, (new_w, new_h))
            overlay = cv2.resize(overlay, (new_w, new_h))
        
        # 3개 이미지를 가로로 결합
        combined = np.hstack([original_image, colored_mask, overlay])
        
        # 텍스트 정보 추가
        combined = self.add_info_text(combined, inference_time, fps)
        
        # 각 이미지에 제목 추가
        cv2.putText(combined, 'Original', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Segmentation', (original_image.shape[1] + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Overlay', (original_image.shape[1] * 2 + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def add_info_text(self, image, inference_time, fps):
        """이미지에 정보 텍스트 추가"""
        # 하단에 정보 추가를 위한 공간 생성
        h, w = image.shape[:2]
        info_height = 80
        info_img = np.zeros((info_height, w, 3), dtype=np.uint8)
        
        # 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(info_img, f"Inference Time: {inference_time*1000:.1f}ms", 
                   (10, 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"FPS: {fps:.1f}", 
                   (10, 50), font, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"Device: {self.device}", 
                   (250, 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"Frame: {self.frame_count}", 
                   (250, 50), font, 0.6, (0, 255, 0), 2)
        
        # 이미지와 정보를 세로로 결합
        result = np.vstack([image, info_img])
        
        return result
    
    def save_results(self, original_image, segmentation_result, overlay):
        """결과 저장 (옵션)"""
        if self.args.save_results and self.frame_count % self.args.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            cv2.imwrite(f"{self.args.output_dir}/original_{timestamp}.jpg", original_image)
            cv2.imwrite(f"{self.args.output_dir}/segmentation_{timestamp}.jpg", segmentation_result)
            cv2.imwrite(f"{self.args.output_dir}/overlay_{timestamp}.jpg", overlay)
            
            logger.info(f"Results saved: {timestamp}")
    
    def calculate_fps(self, inference_time):
        """FPS 계산"""
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
        else:
            fps = 0
        
        # 주기적으로 통계 출력
        if self.frame_count % 30 == 0:
            avg_inference_time = self.total_inference_time / self.frame_count
            logger.info(f"Frame {self.frame_count} - FPS: {fps:.1f}, Avg Inference: {avg_inference_time*1000:.1f}ms")
        
        return fps
    
    def process_image(self, cv_image):
        """이미지 처리 및 시각화"""
        try:
            # 이미지 전처리
            pixel_values, rgb_image = self.preprocess_image(cv_image)
            
            # 추론
            segmentation_mask, probabilities, inference_time = self.predict(
                pixel_values, 
                (cv_image.shape[0], cv_image.shape[1])
            )
            
            # FPS 계산
            fps = self.calculate_fps(inference_time)
            
            # 시각화 생성
            if self.args.show_window:
                combined_viz = self.create_combined_visualization(
                    cv_image, segmentation_mask, inference_time, fps
                )
                
                # 결과 표시
                cv2.imshow('Segmentation Results', combined_viz)
                
                # 클래스 정보 표시
                if self.args.show_class_info:
                    cv2.imshow('Class Information', self.class_info_image)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("ESC pressed. Shutting down...")
                    return False
                elif key == ord('s'):  # S 키로 스크린샷 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", combined_viz)
                    logger.info(f"Screenshot saved: screenshot_{timestamp}.jpg")
            
            # 결과 저장 (옵션)
            if self.args.save_results:
                _, colored_mask = self.create_overlay(cv_image, segmentation_mask)
                overlay, _ = self.create_overlay(cv_image, segmentation_mask)
                self.save_results(cv_image, colored_mask, overlay)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return True

# ROS2 노드 클래스 (이미지 수신만 담당)
class ImageSubscriber(Node):
    def __init__(self, topic, callback):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.callback = callback
        
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )
        
        logger.info(f"Subscribed to: {topic}")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.callback(cv_image)
        except Exception as e:
            logger.error(f"Error in image callback: {e}")

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='RealSense Segmentation Inference')
    
    # 모델 관련
    parser.add_argument('--model_name', type=str, 
                    #    default="nvidia/mit-b0",
                       default="best_seg_model.safetensors ",
                       help='사용할 모델 이름')
    parser.add_argument('--model_path', type=str,
                       default="ckpts/best_model.pth",
                       help='학습된 모델 가중치 경로')
    
    # ROS2 관련
    parser.add_argument('--realsense_topic', type=str,
                       default="/camera/camera/color/image_raw",
                       help='RealSense 이미지 토픽')
    
    # 시각화 관련
    parser.add_argument('--show_window', action='store_true', default=True,
                       help='시각화 창 표시 여부')
    parser.add_argument('--show_class_info', action='store_true',
                       help='클래스 정보 창 표시')
    parser.add_argument('--save_results', action='store_true',
                       help='결과 이미지 저장 여부')
    parser.add_argument('--output_dir', type=str, default="inference_results",
                       help='결과 저장 디렉토리')
    parser.add_argument('--save_interval', type=int, default=30,
                       help='결과 저장 간격 (프레임 단위)')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_args()
    
    # RealSense Segmentation 객체 생성
    segmentation = RealSenseSegmentation(args)
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        # ROS2 노드 생성
        image_subscriber = ImageSubscriber(args.realsense_topic, segmentation.process_image)
        
        logger.info("Starting segmentation inference...")
        logger.info("Press ESC in OpenCV window to quit")
        logger.info("Press 's' in OpenCV window to save screenshot")
        
        # ROS2 스핀 (별도 스레드에서)
        import threading
        ros_thread = threading.Thread(target=lambda: rclpy.spin(image_subscriber), daemon=True)
        ros_thread.start()
        
        # 메인 루프 (OpenCV 창 유지)
        while True:
            if not segmentation.args.show_window:
                time.sleep(0.1)
            else:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    break
                
        logger.info("Shutting down...")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # 정리
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
