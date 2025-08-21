#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry # [수정 1] Odometry 추가
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import json
import threading
import time
import math # [수정 1] math 추가
import traceback # [수정 1] traceback 추가

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)

        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None  # [x, y, yaw]
        self.waypoints = [(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        self.waypoint_index = 0
        self.goal_threshold = 0.4  # [수정 2] 변수명 오타 수정

        # Odom subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/command_odom',
            self.odom_callback,
            10,
        )

        # Control Timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # PlannerNet Initialization
        self.setup_planner()

        # State Variables
        self.current_depth_tensor = None
        self.angular_gain = 2.0

        # Camera Intrinsics for Visualization
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 384.0, 384.0, 320.0, 240.0

        # Visualization Thread Setup
        self.visualization_image = None
        self.running = True
        self.data_lock = threading.Lock()
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()

        self.get_logger().info("✅ RealSense PlannerNet Control with Visualization has started.")

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation_q)
        self.current_pose = [x, y, yaw]
    
    # ... (시각화 스레드, 플래너 설정, 뎁스 콜백 등 다른 함수들은 그대로 유지) ...
    # ... (waypoints_to_cmd_vel, draw_path_and_direction 함수도 그대로 유지) ...
    def _visualization_thread(self):
        self.get_logger().info("Starting visualization thread.")
        while self.running and rclpy.ok():
            with self.data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
            if display_image is not None:
                cv2.imshow("PlannerNet Navigation", display_image)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("Visualization thread stopped.")

    def setup_planner(self):
        config_path = os.path.join(os.path.dirname(os.getcwd()), 'config', 'training_config.json')
        with open(config_path) as f: config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        if torch.cuda.is_available(): self.net = self.net.cuda()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['dataConfig']['crop-size'])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded successfully on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_cv = np.clip(depth_cv, 0, 5000)
            depth_normalized = (depth_cv / 5000.0 * 255).astype(np.uint8)
            self.current_depth_tensor = self.transform(depth_normalized).unsqueeze(0).to(self.device)
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            with self.data_lock:
                self.visualization_image = depth_display
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def waypoints_to_cmd_vel(self, waypoints, dt=0.1):
        if waypoints.shape[1] < 2: return torch.zeros(1, 1, 2, device=waypoints.device)
        dx = waypoints[:, 1:, 0] - waypoints[:, :-1, 0]
        dy = waypoints[:, 1:, 1] - waypoints[:, :-1, 1]
        linear_x = torch.sqrt(dx**2 + dy**2) / dt
        heading_angles = torch.atan2(dy, dx)
        angular_z = torch.zeros_like(linear_x)
        if waypoints.shape[1] > 2:
            angle_diff = heading_angles[:, 1:] - heading_angles[:, :-1]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angular_z[:, 1:] = angle_diff / dt
        angular_z[:, 0] = heading_angles[:, 0] / (dt * self.angular_gain)
        return torch.stack([linear_x, angular_z], dim=-1)

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z):
        if image is None: return None
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape
        for i, point in enumerate(waypoints):
            wp_x, wp_y = point[0], point[1]
            Z_cam, X_cam = wp_x, -wp_y
            if Z_cam > 0.1:
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy)
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)
        turn_text, arrow_color, arrow_end = "Straight", (255, 255, 0), (w // 2, h - 50)
        if angular_z > 0.15: turn_text, arrow_color, arrow_end = "Turn Left", (0, 255, 255), (w // 2 - 50, h - 50)
        elif angular_z < -0.15: turn_text, arrow_color, arrow_end = "Turn Right", (255, 0, 255), (w // 2 + 50, h - 50)
        cv2.putText(image, turn_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)
        return image

    def control_callback(self):
        # [수정 5] odom 데이터가 들어오기 전에 실행되는 것을 방지
        if self.current_depth_tensor is None or self.current_pose is None:
            return

        try:
            # [수정 3] 웨이포인트 주행 로직 전체 추가
            # 1. 모든 웨이포인트에 도달했는지 확인
            if self.waypoint_index >= len(self.waypoints):
                self.get_logger().info("All waypoints reached. Stopping.")
                self.cmd_pub.publish(Twist()) # 정지 명령
                return

            # 2. 현재 목표 웨이포인트와 로봇의 현재 위치 가져오기
            target_wp = self.waypoints[self.waypoint_index]
            current_x, current_y, current_yaw = self.current_pose

            # 3. 목표까지의 거리 계산 및 다음 웨이포인트로 전환
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                return # 다음 제어 주기에서 새 목표로 다시 계산

            # 4. 전역 목표(Global Goal)를 로봇 기준 지역 목표(Local Goal)로 변환
            dx_global = target_wp[0] - current_x
            dy_global = target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            # AI 모델에 입력할 지역 목표 텐서 생성
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 변환된 local_goal_tensor를 AI 모델에 입력
                preds, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                
                waypoints = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
                cmd_vels = self.waypoints_to_cmd_vel(waypoints)
                
                linear_x = torch.clamp(cmd_vels[0, 0, 0], -1.0, 0.5).item()
                angular_z = torch.clamp(cmd_vels[0, 0, 1], -1.0, 0.8).item()
                
                fear_val = fear.cpu().item()
                
                # [수정 4] 'Fear' 안전 로직 부등호 및 반응 수정
                if fear_val < 0.3: # 위험도가 0.3보다 크면 (매우 가까우면) 후진
                    linear_x  = 0.0
                    angular_z = 0.0
                elif fear_val < 0.1: # 위험도가 0.1보다 크면 속도 점진적 감소
                    # 원래 속도에 (1.0 - fear_val) 비율을 곱해 위험할수록 느려지게 만듦
                    linear_x = -0.15
                    angular_z = 0.0
                    # linear_x *= max(0, 1.0 - fear_val)


            # 시각화 및 명령 발행
            with self.data_lock:
                if self.visualization_image is not None:
                    img_to_draw = self.visualization_image.copy()
                    final_img = self.draw_path_and_direction(img_to_draw, waypoints, angular_z)
                    if final_img is not None:
                        self.visualization_image = final_img
            
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)
            
            self.get_logger().info(f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | Cmd:v={linear_x:.2f},w={angular_z:.2f} | Fear:{fear_val:.2f}")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
            self.cmd_pub.publish(Twist())

    # ... (destroy_node, main 함수는 그대로 유지) ...
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        self.cmd_pub.publish(Twist())
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealSensePlannerControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
