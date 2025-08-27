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
import json
import threading
import time
import math
import traceback

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/command_odom', self.odom_callback, 10)

        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None  # [x, y, yaw]
        self.waypoints = [(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        self.waypoint_index = 0
        self.goal_threshold = 0.4

        # Control Timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # PlannerNet Initialization
        self.setup_planner()

        # State Variables
        self.current_depth_tensor = None
        self.angular_gain = 2.0

        # === Matplotlib 플롯을 위한 데이터 저장 변수 ===
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []      # 로봇의 전체 궤적 [[x1, y1], [x2, y2], ...]
        self.latest_preds = np.array([])       # 모델 예측 경로점 (로컬)
        self.latest_waypoints = np.array([])   # 보간된 경로 (로컬)
        self.latest_local_goal = np.array([])  # 현재 목표 (로컬)

        self.get_logger().info("✅ RealSense PlannerNet Control with Integrated Plotting has started.")

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation_q)
        
        with self.plot_data_lock:
            self.current_pose = [x, y, yaw]
            self.trajectory_data.append([x, y])

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

    def control_callback(self):
        if self.current_depth_tensor is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                return

            target_wp = self.waypoints[self.waypoint_index]
            
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                # 마지막 웨이포인트에 도달하면 루프를 계속 돌지 않도록 함
                if self.waypoint_index >= len(self.waypoints):
                    return

            # 다음 목표를 다시 설정
            target_wp = self.waypoints[self.waypoint_index]
            dx_global = target_wp[0] - current_x
            dy_global = target_wp[1] - current_y
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1)
                
                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                
                fear_val = fear.cpu().item()
            
            # === Twist 발행 부분 주석 처리 ===
            # twist = Twist()
            # ...
            # self.cmd_pub.publish(twist)
            # ===============================
            
            self.get_logger().info(f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | Visualizing preds and path. Fear:{fear_val:.2f}")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_cv = (np.clip(depth_cv, 0, 10000)/1000.0).astype(np.float32)
            # depth_normalized = (depth_cv / 5000.0 * 255).astype(np.uint8)
            depth_cv[depth_cv>10] = 0

            # breakpoint()
            self.current_depth_tensor = self.transform(depth_cv).unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        super().destroy_node()


# Matplotlib 애니메이션을 위한 업데이트 함수
def update_plot(frame, node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    with node.plot_data_lock:
        # 스레드 간 안전한 데이터 복사
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj:
        return []

    # 1. 전역 웨이포인트 상태 업데이트 (도착/미도착)
    reached_wps = all_wps[:wp_idx]
    pending_wps = all_wps[wp_idx:]
    
    if reached_wps.size > 0:
        # X축과 Y축 교체: set_data(y, x)
        reached_wps_plot.set_data(reached_wps[:, 1], reached_wps[:, 0])
    else:
        reached_wps_plot.set_data([], [])

    if pending_wps.size > 0:
        # X축과 Y축 교체: set_data(y, x)
        pending_wps_plot.set_data(pending_wps[:, 1], pending_wps[:, 0])
    else:
        pending_wps_plot.set_data([], [])

    # 2. 로봇 궤적 업데이트
    traj_arr = np.array(traj)
    # X축과 Y축 교체: set_data(y, x)
    traj_line.set_data(traj_arr[:, 1], traj_arr[:, 0])

    current_x, current_y, current_yaw = pose
    
    # 3. 현재 위치 및 헤딩 업데이트
    # X축과 Y축 교체: set_data(y, x)
    current_point.set_data([current_y], [current_x])
    heading_len = 0.5
    # 헤딩 벡터도 y, x 순서로 계산
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([current_y, heading_end_y], [current_x, heading_end_x])

    # 4. Local -> Global 좌표 변환 및 플롯 업데이트
    if preds_local.size > 0 and waypoints_local.size > 0 and goal_local.size > 0:
        rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                               [math.sin(current_yaw),  math.cos(current_yaw)]])
        
        waypoints_global = (rot_matrix @ waypoints_local[:, :2].T).T + np.array([current_x, current_y])
        preds_global = (rot_matrix @ preds_local[:, :2].T).T + np.array([current_x, current_y])
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])

        # X축과 Y축 교체: set_data(y, x)
        waypoints_line.set_data(waypoints_global[:, 1], waypoints_global[:, 0])
        preds_points.set_data(preds_global[:, 1], preds_global[:, 0])
        goal_point.set_data([goal_global[1]], [goal_global[0]])

    return [traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot]

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    # X축과 Y축 교체: 라벨 변경
    ax.set_title('Real-time Trajectory and PlannerNet Prediction')
    ax.set_xlabel('Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # 전체 웨이포인트를 기반으로 뷰 범위 설정
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.0
    x_max, y_max = wps_array.max(axis=0) + 1.0
    # X축과 Y축 교체: xlim, ylim 설정 변경
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(y_min, y_max)
    
    # 플롯 객체 생성
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    preds_points, = ax.plot([], [], 'ro', markersize=5, label='Preds (Model Output)')
    waypoints_line, = ax.plot([], [], 'y.-', lw=1, label='Waypoints (Path)')
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    # 웨이포인트 마커 추가
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    
    ax.legend()
    
    # 애니메이션 생성
    ani = FuncAnimation(fig, update_plot, 
                        fargs=(node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot),
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
