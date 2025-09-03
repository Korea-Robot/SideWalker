#!/usr/bin/env python3

import torch
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

from imitation_learning_trainer import ImitationLearningModel

class ImitationInferenceNode(Node):
    def __init__(self, model_path, waypoints):
        super().__init__('imitation_inference_node')
        
        # 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = ImitationLearningModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 10
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
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((360, 640)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        
        # 제어 타이머
        self.control_timer = self.create_timer(0.1, self.control_callback)
        
        self.get_logger().info("Imitation learning inference node started")
    
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
            
            # 텐서로 변환
            self.current_depth_tensor = self.transform(depth_cv).unsqueeze(0).to(self.device)
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")
    
    def control_callback(self):
        if not hasattr(self, 'current_depth_tensor') or self.current_pose is None:
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
            
            # Local goal 계산
            dx_global = target_wp[0] - current_x
            dy_global = target_wp[1] - current_y
            
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                waypoint_pred, collision_prob = self.model(self.current_depth_tensor, local_goal_tensor)
                
                # 첫 번째 waypoint를 사용하여 제어 명령 생성
                first_waypoint = waypoint_pred[0, 0].cpu().numpy()  # [x, y, z]
                
                # 간단한 제어 로직
                target_x, target_y = first_waypoint[0], first_waypoint[1]
                
                # 선속도 계산
                linear_x = np.clip(target_x * 2.0, 0.0, 0.5)
                
                # 각속도 계산
                angular_z = np.clip(target_y * 3.0, -1.0, 1.0)
                
                # 충돌 확률이 높으면 정지
                collision_val = collision_prob.cpu().item()
                if collision_val > 0.8:
                    linear_x = 0.0
                    angular_z = 0.0
                    self.get_logger().warn(f"High collision probability: {collision_val:.3f}")
                
                # 회전이 클 때는 속도 줄이기
                if abs(angular_z) > 0.3:
                    linear_x *= 0.5
            
            # 제어 명령 발행
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)
            
            self.get_logger().info(
                f"WP[{self.waypoint_index}] -> Local({local_x:.2f},{local_y:.2f}) | "
                f"CMD: v={linear_x:.2f}, w={angular_z:.2f} | Collision: {collision_val:.3f}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Control error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--waypoints', nargs='+', type=float, 
                       default=[0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
                       help='Waypoints as flat list: x1 y1 x2 y2 ...')
    args = parser.parse_args()
    
    # 웨이포인트 파싱
    waypoints = []
    for i in range(0, len(args.waypoints), 2):
        waypoints.append((args.waypoints[i], args.waypoints[i+1]))
    
    rclpy.init()
    node = ImitationInferenceNode(args.model_path, waypoints)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
