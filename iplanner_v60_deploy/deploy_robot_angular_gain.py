#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import json

# PlannerNet 관련 import (기존 코드에서)
from planner_net import PlannerNet
from traj_cost import TrajCost

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control')
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        
        # 제어 타이머 (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)
        
        # PlannerNet 초기화
        self.setup_planner()
        
        # 상태 변수
        self.current_depth = None
        self.goal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # 2m 전방 목표
        self.current_cmd = Twist()
        
        # angular gain
        self.angular_gain = 2
        
        self.get_logger().info("RealSense PlannerNet Control Node Started")

    def setup_planner(self):
        """PlannerNet 모델 설정"""
        # 설정 로드
        config_path = os.path.join(os.path.dirname(os.getcwd()), 'config', 'training_config.json')
        with open(config_path) as f:
            config = json.load(f)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['dataConfig']['crop-size'])),
            transforms.Grayscale(num_output_channels=3), # 3채널로 만듬.드
            transforms.ToTensor()
        ])
        
        # 궤적 비용 계산기 (간단한 더미로 대체 가능)
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        
        self.get_logger().info("PlannerNet model loaded successfully")

    def depth_callback(self, msg):
        """깊이 이미지 콜백"""
        try:
            # 깊이 이미지 변환
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 최대 깊이 제한 (설정에 따라)
            depth = np.clip(depth, 0, 5000)  # 5m 제한
            
            # 정규화 (0-255)
            depth_normalized = (depth / 5000.0 * 255).astype(np.uint8)
            
            # 텐서로 변환
            depth_tensor = self.transform(depth_normalized).unsqueeze(0)
            
            if torch.cuda.is_available():
                depth_tensor = depth_tensor.cuda()
            
            self.current_depth = depth_tensor
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def waypoints_to_cmd_vel(self, waypoints, dt=0.1):
        """궤적 포인트를 속도 명령으로 변환"""
        if waypoints.shape[1] < 2:
            return torch.zeros(1, 1, 2, device=waypoints.device)
        
        # 위치 차이 계산
        dx = waypoints[:, 1:, 0] - waypoints[:, :-1, 0]
        dy = waypoints[:, 1:, 1] - waypoints[:, :-1, 1]
        
        # 선속도 계산
        linear_x = torch.sqrt(dx**2 + dy**2) / dt
        
        # 각속도 계산
        heading_angles = torch.atan2(dy, dx)
        angular_z = torch.zeros_like(linear_x)
        
        if waypoints.shape[1] > 2:
            angle_diff = heading_angles[:, 1:] - heading_angles[:, :-1]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angular_z[:, 1:] = angle_diff / dt
            
            # 아니면 first angular value 
            # angular_z[:, 0] = angular_z[:, 1] 
            angular_z[:, 0] = heading_angles[:, 0] / (dt*self.angular_gain)
        
        return torch.stack([linear_x, angular_z], dim=-1)

    def control_callback(self):
        """제어 콜백 - 실제 추론 및 제어 수행"""
        if self.current_depth is None:
            return
        
        try:
        # if 1:
            with torch.no_grad():
                # 더미 오도메트리 (실제로는 로봇에서 받아야 함)
                odom = torch.zeros(1, 3, dtype=torch.float32)
                goal = self.goal.unsqueeze(0)
                
                if torch.cuda.is_available():
                    odom = odom.cuda()
                    goal = goal.cuda()
                
                # PlannerNet 추론
                preds, fear = self.net(self.current_depth, goal)
                
                # 궤적 생성
                waypoints = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
                
                    
                # 속도 명령 생성
                cmd_vels = self.waypoints_to_cmd_vel(waypoints)
                
                # 속도 제한
                linear_x = torch.clamp(cmd_vels[0, 0, 0], -1.0, 1.0).item()
                angular_z = torch.clamp(cmd_vels[0, 0, 1], -1.0, 1.0).item()
                
                fear = fear.cpu().item()
                
                if fear < 0.3 and fear >= 0.1:
                    linear_x = 0
                    angular_z = 0

                if fear < 0.1:
                    linear_x = -0.2
                    angular_z = 0

                # breakpoint()
                
                
                
            # """
                # Twist 메시지 생성
                twist = Twist()
                twist.linear.x = float(linear_x)
                twist.angular.z = float(angular_z)
                
                # 퍼블리시
                self.cmd_pub.publish(twist)
                
                # 로그
                self.get_logger().info(f"Cmd: linear_x={linear_x:.3f}, angular_z={angular_z:.3f}, fear={fear:.3f}")
                
                
                
                
        except Exception as e:
            self.get_logger().error(f"Control error: {e}")
            # 안전을 위해 정지
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
        # """
        
        
def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealSensePlannerControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 정지 명령 송신
        if 'node' in locals():
            stop_twist = Twist()
            node.cmd_pub.publish(stop_twist)
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
