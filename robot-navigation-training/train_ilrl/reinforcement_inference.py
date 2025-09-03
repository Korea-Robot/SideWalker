#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import torchvision.transforms as transforms
import math
from collections import deque

from reinforcement_learning_trainer import ReinforcementLearningModel

class ReinforcementInferenceNode(Node):
    def __init__(self, model_path, waypoints):
        super().__init__('reinforcement_inference_node')
        
        # 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = ReinforcementLearningModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Action space 정의
        self.linear_actions = [0.0, 0.2, 0.5]  # 정지, 느림, 빠름
        self.angular_actions = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]  # 좌회전 ~ 우회전
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 10
        )
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/command_odom', 
            self.odom_callback, 10
        )
        
        # 변수 초기화
        self.waypoints = waypoints
        self.current_pose = None
        self.waypoint_index = 0
        self.goal_threshold = 0.6
        
        # 시퀀스 버퍼 (5프레임)
        self.sequence_length = 5
        self.depth_buffer = deque(maxlen=self.sequence_length)
        self.rgb_buffer = deque(maxlen=self.sequence_length)
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((360, 640)),
            transforms.ToTensor()
        ])
        
        # 제어 타이머
        self.control_timer = self.create_timer(0.1, self.control_callback)
        
        self.get_logger().info("Reinforcement learning inference node started")
    
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose = [x, y, yaw]
    
    def depth_callback(self, msg):
        try:
            # Depth 이미지 처리
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0
            depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32)
            depth_cv[depth_cv > max_depth_value] = 0
            
            # 텐서로 변환 (grayscale)
            depth_tensor = self.transform(depth_cv)
            if depth_tensor.shape[0] == 3:  # RGB로 변환된 경우 첫 번째 채널만 사용
                depth_tensor = depth_tensor[0:1]
            
            self.depth_buffer.append(depth_tensor)
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")
    
    def rgb_callback(self, msg):
        try:
            # RGB 이미지 처리
            rgb_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
            
            # 텐서로 변환
            rgb_tensor = self.transform(rgb_cv)
            self.rgb_buffer.append(rgb_tensor)
            
        except Exception as e:
            self.get_logger().error(f"RGB processing error: {e}")
    
    def control_callback(self):
        if len(self.depth_buffer) < self.sequence_length or len(self.rgb_buffer) < self.sequence_length:
            return
        
        if self.current_pose is None:
            return
        
        try:
            # 웨이포인트 완료 확인
            if self.waypoint_index >= len(self.waypoints):
                twist = Twist()
                self.cmd_pub.publish(twist)
                return
            
            # 현재 목표 웨이포인트
            target_wp = self.waypoints[self.waypoint_index]
            current_x, current_y, current_yaw = self.current_pose
            
            # 목표 도달 확인
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                return
            
            # 시퀀스 데이터 준비
            depth_sequence = torch.stack(list(self.depth_buffer)).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
            rgb_sequence = torch.stack(list(self.rgb_buffer)).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
            
            # 모델 추론
            with torch.no_grad():
                linear_logits, angular_logits, value = self.model(depth_sequence, rgb_sequence)
                
                # Action 확률 계산
                linear_probs = F.softmax(linear_logits, dim=-1)
                angular_probs = F.softmax(angular_logits, dim=-1)
                
                # Action 샘플링 (확률적) 또는 최대값 선택 (결정적)
                # 추론 시에는 보통 결정적으로 선택
                linear_action_idx = torch.argmax(linear_probs, dim=-1).item()
                angular_action_idx = torch.argmax(angular_probs, dim=-1).item()
                
                # 실제 action 값으로 변환
                linear_x = self.linear_actions[linear_action_idx]
                angular_z = self.angular_actions[angular_action_idx]
                
                # 목표 방향 고려한 추가 조정
                dx_global = target_wp[0] - current_x
                dy_global = target_wp[1] - current_y
                local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
                local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
                
                # 목표가 뒤쪽에 있으면 회전 우선
                if local_x < 0:
                    linear_x = 0.0
                    if local_y > 0:
                        angular_z = max(angular_z, 0.2)  # 좌회전 강화
                    else:
                        angular_z = min(angular_z, -0.2)  # 우회전 강화
            
            # 제어 명령 발행
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)
            
            self.get_logger().info(
                f"WP[{self.waypoint_index}] -> Local({local_x:.2f},{local_y:.2f}) | "
                f"Actions: L={linear_action_idx}({linear_x:.2f}), A={angular_action_idx}({angular_z:.2f}) | "
                f"Value: {value.item():.3f}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Control error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained RL model')
    parser.add_argument('--waypoints', nargs='+', type=float, 
                       default=[0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                       help='Waypoints as flat list: x1 y1 x2 y2 ...')
    args = parser.parse_args()
    
    # 웨이포인트 파싱
    waypoints = []
    for i in range(0, len(args.waypoints), 2):
        waypoints.append((args.waypoints[i], args.waypoints[i+1]))
    
    rclpy.init()
    node = ReinforcementInferenceNode(args.model_path, waypoints)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
