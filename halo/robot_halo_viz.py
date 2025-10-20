#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import threading
import time
import math
import traceback
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# HALO Model imports
from reward_estimation_model import HALORewardModel

class RealSenseHALOControl(Node):
    def __init__(self):
        super().__init__('realsense_halo_control')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)

        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None
        self.waypoints = [(0.0, 0.0), (2.7, 0.0), (2.54, 2.6), (2.7, 0.0), (0.0, 0.0)]
        self.waypoint_index = 0
        self.goal_threshold = 0.5

        self.control_timer = self.create_timer(0.1, self.control_callback)
        
        # HALO Model Setup
        self.setup_halo_model()

        self.current_rgb_tensor = None
        self.current_depth_image = None

        # Visualization
        self.visualization_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()
        
        # Matplotlib 플롯을 위한 데이터 저장
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_rewards = np.array([])
        self.latest_angular_velocities = np.array([])
        self.latest_selected_action = 0

        self.get_logger().info("✅ RealSense HALO Control Node Started")

    def setup_halo_model(self):
        """HALO Reward Model 초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model path (학습된 모델 경로로 변경)
        MODEL_PATH = './gamma1.0_lr_1e-05/best_model.pth'
        
        # Model 로드
        self.model = HALORewardModel(freeze_dino=True).to(self.device)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
        
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        
        # RGB 전처리 설정 (224x224 입력)
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 후보 액션(궤적) 생성
        self.NUM_CANDIDATE_ACTIONS = 17
        self.FIXED_LINEAR_V = 0.6
        self.IMG_SIZE_MASK = 32
        self.candidate_masks, self.angular_velocities = self.generate_candidate_masks()
        
        self.get_logger().info(f"HALO Model loaded on {self.device}")
        self.get_logger().info(f"Generated {self.NUM_CANDIDATE_ACTIONS} candidate actions")

    def draw_line_bresenham(self, x0, y0, x1, y1):
        """Bresenham 선 알고리즘"""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        rr, cc = [], []
        
        while True:
            rr.append(x0)
            cc.append(y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
                
        return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)

    def generate_trajectory_mask_from_df(self, df, img_size):
        """속도 명령으로부터 궤적 마스크 생성"""
        if len(df) < 2:
            return np.zeros((img_size, img_size), dtype=np.uint8)

        # Odometry 계산
        delta_t = df['timestamp'].diff().fillna(0) / 1000.0
        x, y, theta = 0.0, 0.0, 0.0
        odom_list = [[x, y, theta]]
        for i in range(1, len(df)):
            v = df['manual_linear_x'].iloc[i]
            w = df['manual_angular_z'].iloc[i]
            dt = delta_t.iloc[i]
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            odom_list.append([x, y, theta])
        odom_segment = np.array(odom_list)

        # Egocentric 좌표계 변환
        x0, y0, theta0 = odom_segment[0]
        coords_translated = odom_segment[:, :2] - np.array([x0, y0])
        c, s = np.cos(-theta0), np.sin(-theta0)
        rotation_matrix = np.array([[c, -s], [s, c]])
        ego_coords = (rotation_matrix @ coords_translated.T).T

        # 마스크 생성
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        max_range = 1.0
        
        u = np.clip(((ego_coords[:, 0] / max_range) * (img_size - 1)).astype(int), 0, img_size - 1)
        v = np.clip((ego_coords[:, 1] / (max_range / 1.3) * (img_size - 1) / 2 + (img_size / 2)).astype(int), 0, img_size - 1)

        for i in range(len(u) - 1):
            rr, cc = self.draw_line_bresenham(u[i], v[i], u[i+1], v[i+1])
            mask[rr, cc] = 1
        
        mask[0, img_size // 2] = 1
        return mask

    def generate_candidate_masks(self):
        """후보 궤적 마스크 생성"""
        angular_velocities = np.linspace(-1.0, 1.0, self.NUM_CANDIDATE_ACTIONS)
        candidate_masks = []
        
        for w in angular_velocities:
            duration = 2.0
            hz = 10
            num_points = int(duration * hz)
            timestamps = np.arange(num_points) * (1000 / hz)
            linear_v = self.FIXED_LINEAR_V / (1 + 0.5 * abs(w))
            
            dummy_df = pd.DataFrame({
                'timestamp': timestamps,
                'manual_linear_x': [linear_v] * num_points,
                'manual_angular_z': [w] * num_points,
            })
            
            mask_np = self.generate_trajectory_mask_from_df(dummy_df, img_size=self.IMG_SIZE_MASK)
            candidate_masks.append(torch.from_numpy(mask_np).float())
            
        return torch.stack(candidate_masks).unsqueeze(1).to(self.device), angular_velocities

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        with self.plot_data_lock:
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y])

    def rgb_callback(self, msg):
        """RGB 이미지 콜백"""
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
            
            # 모델 입력용 텐서 생성
            self.current_rgb_tensor = self.rgb_transform(rgb_cv).unsqueeze(0).to(self.device)
            
            # 시각화용 이미지 저장
            with self.plot_data_lock:
                self.visualization_image = rgb_cv.copy()
                
        except Exception as e:
            self.get_logger().error(f"RGB processing error: {e}")

    def depth_callback(self, msg):
        """뎁스 이미지 콜백 (시각화용)"""
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.current_depth_image = depth_cv
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def control_callback(self):
        """메인 제어 루프"""
        if self.current_rgb_tensor is None or self.current_pose is None:
            return

        try:
            # 모든 웨이포인트 완료 확인
            if self.waypoint_index >= len(self.waypoints):
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

            # 현재 웨이포인트까지의 거리 계산
            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            
            # 웨이포인트 도착 확인
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    twist = Twist()
                    self.cmd_pub.publish(twist)
                    return

            # === HALO Model 추론 ===
            with torch.no_grad():
                # RGB 이미지를 후보 액션 수만큼 복제
                rgb_expanded = self.current_rgb_tensor.repeat(self.NUM_CANDIDATE_ACTIONS, 1, 1, 1)
                
                # 모델 추론: 보상과 깊이 예측
                predicted_rewards, predicted_depths = self.model(rgb_expanded, self.candidate_masks)
                
                # 결과 가져오기
                rewards = predicted_rewards.squeeze().cpu().numpy()
                
                # 최대 보상을 가진 액션 선택
                best_action_idx = np.argmax(rewards)
                selected_angular_z = self.angular_velocities[best_action_idx]
                
                # 데이터 저장 (시각화용)
                with self.plot_data_lock:
                    self.latest_rewards = rewards
                    self.latest_angular_velocities = self.angular_velocities
                    self.latest_selected_action = best_action_idx

            # === 제어 명령 생성 ===
            # 회전이 크면 속도 감소
            if abs(selected_angular_z) > 0.3:
                linear_x = 0.2
            elif abs(selected_angular_z) > 0.15:
                linear_x = 0.3
            else:
                linear_x = 0.4

            # 보상이 너무 낮으면 정지 (충돌 위험)
            if rewards[best_action_idx] < -0.5:
                linear_x = 0.0
                selected_angular_z = 0.0
                self.get_logger().warn("⚠️ Low reward detected - STOPPING")

            # Twist 메시지 발행
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(selected_angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}] | "
                f"Linear: {linear_x:.2f} m/s | "
                f"Angular: {selected_angular_z:.2f} rad/s | "
                f"Reward: {rewards[best_action_idx]:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")

    def _visualization_thread(self):
        """시각화 스레드"""
        self.get_logger().info("Starting visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
                rewards = self.latest_rewards.copy() if self.latest_rewards.size > 0 else None
                angular_vels = self.latest_angular_velocities.copy() if self.latest_angular_velocities.size > 0 else None
                selected_idx = self.latest_selected_action
            
            if display_image is not None:
                # 보상 정보 오버레이
                if rewards is not None and angular_vels is not None:
                    img_with_info = self.draw_reward_info(display_image, rewards, angular_vels, selected_idx)
                else:
                    img_with_info = display_image
                
                cv2.imshow("HALO Vision", img_with_info)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        self.get_logger().info("Visualization thread stopped.")

    def draw_reward_info(self, image, rewards, angular_vels, selected_idx):
        """이미지에 보상 정보 오버레이"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # 선택된 액션 표시
        selected_angular = angular_vels[selected_idx]
        reward_val = rewards[selected_idx]
        
        # 방향 텍스트
        if abs(selected_angular) < 0.1:
            direction_text = "STRAIGHT"
            color = (0, 255, 0)
        elif selected_angular > 0:
            direction_text = "LEFT"
            color = (255, 255, 0)
        else:
            direction_text = "RIGHT"
            color = (255, 0, 255)
        
        # 정보 표시
        cv2.putText(img, f"Action: {direction_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"Angular: {selected_angular:.2f} rad/s", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Reward: {reward_val:.3f}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 간단한 보상 바 그래프 (하단)
        bar_height = 50
        bar_y_start = h - bar_height - 10
        bar_width = w // len(rewards)
        
        for i, r in enumerate(rewards):
            # 보상 정규화 (0~1)
            normalized_reward = np.clip((r + 1) / 2, 0, 1)
            bar_color = (0, int(255 * normalized_reward), int(255 * (1 - normalized_reward)))
            
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            y1 = bar_y_start + int(bar_height * (1 - normalized_reward))
            y2 = bar_y_start + bar_height
            
            cv2.rectangle(img, (x1, y1), (x2, y2), bar_color, -1)
            
            # 선택된 액션 강조
            if i == selected_idx:
                cv2.rectangle(img, (x1, bar_y_start), (x2, bar_y_start + bar_height), (0, 255, 255), 3)
        
        return img

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        super().destroy_node()


def update_plot(frame, node, ax, traj_line, current_point, heading_line, reached_wps_plot, pending_wps_plot):
    """Matplotlib 애니메이션 업데이트"""
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj:
        return []

    reached_wps = all_wps[:wp_idx]
    pending_wps = all_wps[wp_idx:]
    
    if reached_wps.size > 0:
        reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else:
        reached_wps_plot.set_data([], [])
    
    if pending_wps.size > 0:
        pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else:
        pending_wps_plot.set_data([], [])

    traj_arr = np.array(traj)
    traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0])

    current_x, current_y, current_yaw = pose
    current_point.set_data([-current_y], [current_x])
    
    heading_len = 0.5
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

    return [traj_line, current_point, heading_line, reached_wps_plot, pending_wps_plot]


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseHALOControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Real-time Trajectory with HALO Model')
    ax.set_xlabel('-Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.5
    x_max, y_max = wps_array.max(axis=0) + 1.5
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(-y_max, -y_min)
    
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    ax.legend()
    
    ani = FuncAnimation(fig, update_plot, 
                        fargs=(node, ax, traj_line, current_point, heading_line, reached_wps_plot, pending_wps_plot),
                        interval=100, blit=True)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == '__main__':
    main()

