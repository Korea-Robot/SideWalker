#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import traceback
import threading
import time



# 2. 학습된 모델 가중치 파일 경로
MODEL_PATH = "best_model2.pth"
# MODEL_PATH = "best_seg_model.pth" # best accuracy feel
# MODEL_PATH = "seg_model_epoch_100.pth"



# --- 학습 스크립트에서 가져온 클래스 정보 ---
CLASS_TO_IDX = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
    'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
    'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
    'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
    'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
    'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}


NUM_LABELS = len(CLASS_TO_IDX)

# --- 학습 스크립트에서 가져온 모델 클래스 정의 ---
class DirectSegFormer(nn.Module):
    """학습 시 사용된 모델과 동일한 구조의 클래스"""
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=30):
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

class SegformerControlNode(Node):
    def __init__(self):
        super().__init__('segformer_control_node_kr')

        # === 모델 및 제어 설정 ===
        self.MODEL_PATH = "best_seg_model2.pth"  # 사용할 Segformer 모델 경로
        self.NUM_LABELS = 30
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.FIXED_LINEAR_V = 0.4
        self.NUM_CANDIDATE_ACTIONS = 17

        ## 추가됨: 장애물 비용 계산을 위한 설정 ##
        self.NUM_STEERING_BINS = 9  # 이미지를 몇 개의 세로 영역으로 나눌지 결정
        self.OBSTACLE_WEIGHT_POWER = 2.0 # 이미지 하단 장애물에 얼마나 더 큰 가중치를 줄지 결정

        ## 수정됨: 행동 결정 가중치 (Segformer에 맞게 조절) ##
        self.SAFETY_WEIGHT = 1.5  # 안전 점수(장애물 회피)에 대한 가중치
        self.GOAL_WEIGHT = 0.5     # 목표 점수(목표 방향 근접도)에 대한 가중치
        self.GOAL_ANGLE_SMOOTHNESS = 2.0


        
        # === ROS2 설정 ===
        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # === 상태 및 웨이포인트 변수 ===
        self.current_pose = None
        self.current_segmentation_mask = None # 수정됨: RGB 텐서 대신 분할 마스크를 저장
        self.waypoint_index = 0
        self.goal_threshold = 0.5
        self.waypoints = [(0.0, 0.0),(2.5, 0.0), (2.5, 2.6), (2.5, 0.0), (0.0, 0.0)]

        # === 모델 및 후보 마스크 설정 ===
        self.setup_segmentation_model()
        self.angular_velocities = np.linspace(-1.0, 1.0, self.NUM_CANDIDATE_ACTIONS)

        # === 시각화 설정 ===
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_lock = threading.Lock()
        self.visualization_image = None # 원본 + 오버레이 이미지를 저장
        self.latest_final_scores = np.zeros(self.NUM_CANDIDATE_ACTIONS)
        self.latest_obstacle_costs = np.zeros(self.NUM_STEERING_BINS)
        self.best_action_idx = 0
        self.latest_goal_angle = 0.0
        self.running = True
        self.vis_thread.start()
        
        self.get_logger().info(f"✅ Segformer 제어 노드가 {self.DEVICE}에서 시작되었습니다.")

    def setup_segmentation_model(self):
        """Segformer 모델 로드 및 이미지 전처리 파이프라인 설정"""
        self.seg_model = DirectSegFormer(num_classes=self.NUM_LABELS)
        
        try:
            # 체크포인트 로드 (weights_only=False로 시도)
            checkpoint = torch.load(MODEL_PATH, map_location=self.DEVICE, weights_only=False)

            # 일부 모델은 state_dict가 딕셔너리 안에 있을 수 있음
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                # model. 키 접두사 제거 (PyTorch Lightning 등에서 저장 시 추가될 수 있음)
                elif key.startswith('model.'):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            self.seg_model.load_state_dict(new_state_dict, strict=False)
            self.get_logger().info("Segformer 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"Segformer 모델 로드 실패: {e}")

        self.seg_model.to(self.DEVICE)
        self.seg_model.eval()

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.color_palette = self.create_color_palette()

    def create_color_palette(self):
        """클래스별 시각화 색상 팔레트 생성"""
        cmap = plt.cm.get_cmap('jet', self.NUM_LABELS)
        palette = np.zeros((self.NUM_LABELS, 3), dtype=np.uint8)
        for i in range(self.NUM_LABELS):
            if i == 0: palette[i] = [0, 0, 0]; continue
            rgba = cmap(i)
            palette[i] = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        return palette

    def image_callback(self, msg: Image):
        """이미지를 받아 Segformer 추론을 수행하고 결과를 저장"""
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_h, original_w, _ = img_bgr.shape
            
            # 1) 이미지 전처리
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self.seg_transform(img_rgb).unsqueeze(0).to(self.DEVICE)

            # 2) 추론 수행
            with torch.no_grad():
                logits = self.seg_model(input_tensor)

            # 3) 결과 후처리
            upsampled_logits = F.interpolate(logits, size=(original_h, original_w), mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # 4) 시각화 이미지 생성 및 결과 저장
            segmentation_image = self.color_palette[pred_mask]
            overlay = cv2.addWeighted(img_bgr, 0.6, segmentation_image, 0.4, 0)
            
            with self.vis_lock:
                self.current_segmentation_mask = pred_mask
                self.visualization_image = np.hstack((img_bgr, overlay))

        except Exception as e:
            self.get_logger().error(f"이미지 콜백 오류: {e}\n{traceback.format_exc()}")
    
    # transform quaterion => euler angular z 
    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    # control callback 
    def control_callback(self):
        """Segformer 마스크와 목표 지점을 함께 사용하여 제어 명령 생성"""
        if self.current_segmentation_mask is None or self.current_pose is None:
            return

        # --- 웨이포인트 및 로컬 목표 계산 (이전과 동일) ---
        if self.waypoint_index >= len(self.waypoints): self.publish_stop_command(); return
        target_wp = self.waypoints[self.waypoint_index]
        current_x, current_y, current_yaw = self.current_pose
        distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
        if distance_to_goal < self.goal_threshold:
            self.waypoint_index += 1; self.get_logger().info(f"✅ 웨이포인트 {self.waypoint_index-1} 도착!");
            if self.waypoint_index >= len(self.waypoints): self.get_logger().info("✅ 모든 웨이포인트 도착."); self.publish_stop_command(); return
        
        target_wp = self.waypoints[self.waypoint_index]
        dx_global = target_wp[0] - current_x; dy_global = target_wp[1] - current_y
        local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
        local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
        target_angle = math.atan2(local_y, local_x)
        
        # --- Segformer 마스크 기반 제어 로직 ---
        try:
            with self.vis_lock:
                mask = self.current_segmentation_mask.copy()

            ## NEW: 마스크로부터 안전 점수 계산 ##
            safety_scores = self.calculate_safety_scores_from_mask(mask)
            
            ## NEW: 목표 점수 계산 (이전과 동일) ##
            angular_vels_tensor = torch.from_numpy(self.angular_velocities).float().to(self.DEVICE)
            angle_diffs = torch.abs(angular_vels_tensor - target_angle)
            goal_scores = torch.exp(-self.GOAL_ANGLE_SMOOTHNESS * angle_diffs)
            
            ## NEW: 최종 점수 계산 ##
            final_scores = (self.SAFETY_WEIGHT * safety_scores) + (self.GOAL_WEIGHT * goal_scores)
            
            best_action_idx = torch.argmax(final_scores).item()
            chosen_angular_z = self.angular_velocities[best_action_idx]
            chosen_linear_x = self.FIXED_LINEAR_V / (1.0 + 0.8 * abs(chosen_angular_z))

            with self.vis_lock:
                self.latest_final_scores = final_scores.cpu().numpy()
                self.best_action_idx = best_action_idx
                self.latest_goal_angle = target_angle

            # --- 명령 전송 ---
            twist = Twist(); twist.linear.x = float(chosen_linear_x); twist.angular.z = float(chosen_angular_z)
            self.cmd_pub.publish(twist)
            
            log_msg = (f"WP[{self.waypoint_index}] | 최적 Idx: {best_action_idx}, "
                       f"총점: {final_scores[best_action_idx]:.2f} "
                       f"(안전: {safety_scores[best_action_idx]:.2f}, 목표: {goal_scores[best_action_idx]:.2f}) "
                       f"-> v:{chosen_linear_x:.2f}, w:{chosen_angular_z:.2f}")
            self.get_logger().info(log_msg)

        except Exception as e:
            self.get_logger().error(f"제어 루프 오류: {e}\n{traceback.format_exc()}")

    def calculate_safety_scores_from_mask(self, mask):
        """분할 마스크를 분석하여 각 방향(Bin)의 안전 점수를 계산합니다."""
        h, w = mask.shape
        # 배경(0)이 아닌 모든 픽셀을 장애물(1)로 처리
        obstacle_mask = (mask > 0).astype(np.float32)

        # 이미지 하단에 높은 가중치를 부여하는 행렬 생성
        # (h, 1) 형태로 만들어 브로드캐스팅이 가능하게 함
        row_weights = np.linspace(0.1, 1.0, h) ** self.OBSTACLE_WEIGHT_POWER
        weighted_obstacle_mask = obstacle_mask * row_weights[:, np.newaxis]

        bin_width = w // self.NUM_STEERING_BINS
        obstacle_costs = np.zeros(self.NUM_STEERING_BINS)

        for i in range(self.NUM_STEERING_BINS):
            start_col = i * bin_width
            end_col = (i + 1) * bin_width
            bin_mask = weighted_obstacle_mask[:, start_col:end_col]
            obstacle_costs[i] = np.sum(bin_mask)
        
        # 시각화를 위해 비용 저장
        with self.vis_lock:
            self.latest_obstacle_costs = obstacle_costs.copy()

        # 비용을 0~1 범위로 정규화 (비용이 클수록 위험)
        if np.max(obstacle_costs) > 0:
            normalized_costs = obstacle_costs / (np.max(obstacle_costs) + 1e-6)
        else:
            normalized_costs = obstacle_costs

        # 안전 점수 계산 (비용이 높을수록 안전 점수는 낮아짐)
        safety_scores_by_bin = 1.0 - normalized_costs

        # 9개의 Bin 점수를 17개의 후보 행동 점수로 매핑(선형 보간)
        bin_indices = np.linspace(0, self.NUM_STEERING_BINS - 1, self.NUM_STEERING_BINS)
        action_indices = np.linspace(0, self.NUM_STEERING_BINS - 1, self.NUM_CANDIDATE_ACTIONS)
        safety_scores_by_action = np.interp(action_indices, bin_indices, safety_scores_by_bin)
        
        return torch.from_numpy(safety_scores_by_action).float().to(self.DEVICE)

    def publish_stop_command(self):
        twist = Twist(); twist.linear.x = 0.0; twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _visualization_thread(self):
        """시각화 스레드: 추론 결과, 장애물 비용, 목표 방향 등을 표시"""
        self.get_logger().info("CV2 시각화 스레드를 시작합니다.")
        while self.running and rclpy.ok():
            with self.vis_lock:
                if self.visualization_image is None: time.sleep(0.1); continue
                vis_img = self.visualization_image.copy()
                costs = self.latest_obstacle_costs.copy()
                goal_angle = self.latest_goal_angle
                best_idx = self.best_action_idx

            # 장애물 비용 및 목표 방향 시각화 추가
            vis_img = self.draw_control_info(vis_img, costs, goal_angle, best_idx)

            cv2.imshow("Segformer Control Vision", vis_img)
            if cv2.waitKey(1) == 27: self.running = False; break
        cv2.destroyAllWindows()

    def draw_control_info(self, image, costs, goal_angle, best_idx):
        """이미지 위에 제어 관련 정보를 시각화합니다."""
        # image는 원본 + 오버레이가 합쳐진 넓은 이미지 (w*2, h)
        h, w, _ = image.shape
        w = w // 2 # 실제 이미지 너비
        
        # 1. 장애물 비용을 바 그래프로 표시
        bar_max_h = 50
        bin_w = w // self.NUM_STEERING_BINS
        if np.max(costs) > 0:
            norm_costs = (costs / np.max(costs)) * bar_max_h
        else:
            norm_costs = costs
            
        for i, cost in enumerate(norm_costs):
            start_x = w + i * bin_w # 오버레이 이미지 영역에 그리기
            end_x = w + (i + 1) * bin_w
            # 비용이 높을수록 빨간색에 가깝게
            color = (0, int(128 * (1-cost/bar_max_h)), int(255 * cost/bar_max_h))
            cv2.rectangle(image, (start_x, h - int(cost)), (end_x, h), color, -1)

        # 2. 목표 방향 화살표
        arrow_len = 80
        arrow_end_x = int(w + w/2 - arrow_len * math.sin(goal_angle))
        arrow_end_y = int(h - 20 - arrow_len * math.cos(goal_angle))
        cv2.arrowedLine(image, (w + w // 2, h - 20), (arrow_end_x, arrow_end_y), (0, 255, 255), 3)

        # 3. 선택된 경로 방향 표시
        chosen_angle_rad = np.deg2rad((best_idx / (self.NUM_CANDIDATE_ACTIONS-1) - 0.5) * -90) # 각속도를 대략적인 각도로 변환
        chosen_end_x = int(w + w/2 - arrow_len * math.sin(chosen_angle_rad))
        chosen_end_y = int(h - 20 - arrow_len * math.cos(chosen_angle_rad))
        cv2.arrowedLine(image, (w + w // 2, h - 20), (chosen_end_x, chosen_end_y), (255, 0, 255), 2, tipLength=0.2) # 자홍색
        
        return image

    def destroy_node(self):
        self.get_logger().info("노드를 종료합니다...")
        self.running = False
        self.publish_stop_command()
        self.vis_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SegformerControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
