#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms

# TF2 관련 라이브러리
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs # TF 변환에 필요

# --- 사용자 설정 변수 ---
# 1. 모델 및 토픽 설정
REALSENSE_COLOR_TOPIC = "/camera/camera/color/image_raw"
REALSENSE_DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
REALSENSE_INFO_TOPIC = "/camera/camera/color/camera_info" # RGB 카메라의 파라미터 사용
MODEL_PATH = "best_seg_model.pth" # 여기에 실제 모델 경로를 입력하세요.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 로봇 좌표계(TF) 설정
ROBOT_BASE_FRAME = "base_link" # 로봇의 기준 좌표계
CAMERA_FRAME = "camera_link"  # 카메라 좌표계 (실제 TF에 맞게 수정)

# 3. Costmap 설정
COSTMAP_RESOLUTION = 0.05  # Costmap 격자 하나의 크기 (미터 단위, 5cm)
COSTMAP_WIDTH_M = 10.0     # Costmap 너비 (미터)
COSTMAP_HEIGHT_M = 10.0    # Costmap 높이 (미터)
OBSTACLE_COST = 100        # 장애물 비용 (0-100 사이 값)

# 4. 장애물로 간주할 클래스 ID 설정
# CLASS_TO_IDX를 참고하여 실제 장애물에 해당하는 ID를 리스트로 만듭니다.
OBSTACLE_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
]

# --- 학습 스크립트에서 가져온 모델 클래스 ---
class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=30):
        super().__init__()
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    def forward(self, x):   
        return self.original_model(pixel_values=x).logits

# --- 메인 노드 클래스 ---
class SegformerCostmapNode(Node):
    def __init__(self):
        super().__init__('segformer_costmap_node')

        # 모델 로드
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ROS 관련 초기화
        self.bridge = CvBridge()
        self.camera_intrinsics = None

        # TF 리스너 초기화
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher (Costmap 발행)
        self.costmap_pub = self.create_publisher(OccupancyGrid, 'costmap', 10)

        # Subscriber (카메라 정보는 한 번만 받음)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, REALSENSE_INFO_TOPIC, self.cam_info_callback, 10)

        # 여러 토픽을 동기화하여 수신
        self.color_sub = message_filters.Subscriber(self, Image, REALSENSE_COLOR_TOPIC)
        self.depth_sub = message_filters.Subscriber(self, Image, REALSENSE_DEPTH_TOPIC)

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1)
        self.time_synchronizer.registerCallback(self.perception_callback)

        self.get_logger().info(f"'{self.get_name()}'가 시작되었습니다. 모델: {MODEL_PATH}")

    def load_model(self):
        model = DirectSegFormer(num_classes=30)
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            new_state_dict = {('original_model.' + k): v for k, v in checkpoint.items()}
            model.load_state_dict(new_state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            self.get_logger().info("모델 로드 성공")
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")
            model = None
        return model

    def cam_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.cam_info_sub.destroy() # 한 번만 받고 구독 해제
            self.get_logger().info("카메라 내부 파라미터 수신 완료.")

    def perception_callback(self, color_msg, depth_msg):
        if self.model is None or self.camera_intrinsics is None:
            self.get_logger().warn("모델 또는 카메라 파라미터가 준비되지 않았습니다.", throttle_duration_sec=5)
            return

        try:
            # 1. 추론 수행
            img_bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # 16비트 정수형
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(img_rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_tensor)
            
            upsampled_logits = nn.functional.interpolate(
                logits, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # 2. Costmap 생성
            self.create_and_publish_costmap(pred_mask, depth_image, color_msg.header)

        except Exception as e:
            self.get_logger().error(f"처리 중 에러 발생: {e}")

    def create_and_publish_costmap(self, mask, depth, header):
        # OccupancyGrid 메시지 초기화
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = ROBOT_BASE_FRAME
        grid_msg.info.resolution = COSTMAP_RESOLUTION
        grid_msg.info.width = int(COSTMAP_WIDTH_M / COSTMAP_RESOLUTION)
        grid_msg.info.height = int(COSTMAP_HEIGHT_M / COSTMAP_RESOLUTION)
        # Costmap 원점을 로봇의 뒤쪽/왼쪽으로 설정 (로봇이 중앙에 위치)
        grid_msg.info.origin.position.x = -COSTMAP_WIDTH_M / 2
        grid_msg.info.origin.position.y = -COSTMAP_HEIGHT_M / 2
        
        # 데이터 배열 초기화 (-1: 알 수 없음, 0: 비어있음, 100: 점유)
        grid_msg.data = [-1] * (grid_msg.info.width * grid_msg.info.height)
        
        # 카메라 파라미터
        K = self.camera_intrinsics.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        # TF 변환 정보 가져오기
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, header.frame_id, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'TF 변환을 가져올 수 없습니다: {e}')
            return

        # 장애물 픽셀만 3D로 투영 (처리 속도를 위해 다운샘플링)
        obstacle_pixels = np.argwhere(np.isin(mask, OBSTACLE_CLASS_IDS))
        for v, u in obstacle_pixels[::10]: # 10픽셀마다 하나씩 샘플링
            d = depth[v, u] / 1000.0 # 밀리미터를 미터로 변환
            if d == 0: continue

            # 3D 투영 (카메라 좌표계)
            x_cam = (u - cx) * d / fx
            y_cam = (v - cy) * d / fy
            z_cam = d

            # TF 변환 (카메라 -> 로봇 베이스)
            point_in_camera = tf2_geometry_msgs.PointStamped()
            point_in_camera.header = header
            point_in_camera.point.x = z_cam # RealSense는 Z축이 앞으로 나감
            point_in_camera.point.y = -x_cam
            point_in_camera.point.z = -y_cam
            
            point_in_base = tf2_ros.TransformStamped()
            point_in_base = tf2_geometry_msgs.do_transform_point(point_in_camera, transform)

            # 로봇 좌표를 Costmap 격자 인덱스로 변환
            grid_x = int((point_in_base.point.x - grid_msg.info.origin.position.x) / COSTMAP_RESOLUTION)
            grid_y = int((point_in_base.point.y - grid_msg.info.origin.position.y) / COSTMAP_RESOLUTION)

            if 0 <= grid_x < grid_msg.info.width and 0 <= grid_y < grid_msg.info.height:
                index = grid_y * grid_msg.info.width + grid_x
                grid_msg.data[index] = OBSTACLE_COST
        
        self.costmap_pub.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SegformerCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
