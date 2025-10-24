#!/usr/bin/env python3
"""
RealSense PlannerNet Control Node with BEV Map Avoidance

Subscribes to:
- /camera/camera/depth/image_rect_raw (Depth Image)
- /krm_auto_localization/odom (Odometry)
- /semantic_bev_map (BEV Occupancy Grid)

Publishes:
- /mcu/command/manual_twist (Control Command)

Logic:
1. PlannerNet: Generates a reactive path from depth images.
2. BEV Avoidance: Detects obstacles from the BEV map, inflates them by 
   'robot_radius', and calculates an avoidance vector.
3. Blending: Blends the commands from (1) and (2) based on proximity
   to BEV obstacles.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
import threading
import time
import math
import traceback
import sensor_msgs_py.point_cloud2 as pc2 # BEV맵 용

# Matplotlib 추가
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PlannerNet and TrajCost imports
from planner_net import PlannerNet
from traj_cost import TrajCost

print('0925 test start!!!!!!!')
print('0925 test start!!!!!!!')
print('0925 test start!!!!!!!')
print('0925 test start!!!!!!!')
print('0925 test start!!!!!!!')


class RealSensePlannerControl(Node):
    def __init__(self):
        super().__init__('realsense_planner_control_viz')

        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # === BEV 맵 구독자 추가 ===
        self.bev_sub = self.create_subscription(
            PointCloud2,
            '/semantic_bev_map',  # BEV 맵 토픽
            self.bev_callback,
            10
        )

        # === BEV 맵 파라미터 (semantic_bev_node와 일치시킬 것) ===
        # 중요: 이 값들은 BEV 맵 생성 노드의 파라미터와
        # 반드시 일치해야 합니다!
        self.bev_resolution = 0.1  # 맵 해상도 (m/cell)
        self.bev_size_x = 30.0    # 맵 전체 너비 (m)
        self.bev_size_y = 30.0    # 맵 전체 높이 (m)
        self.bev_origin_x = -self.bev_size_x / 2.0  # 맵 원점 X
        self.bev_origin_y = -self.bev_size_y / 2.0  # 맵 원점 Y
        self.bev_cells_x = int(self.bev_size_x / self.bev_resolution)
        self.bev_cells_y = int(self.bev_size_y / self.bev_resolution)

        # === BEV 맵 데이터 저장을 위한 변수 ===
        self.bev_lock = threading.Lock()  # 스레드간 데이터 접근 보호
        self.occupied_cells = set()       # 점유된 셀의 (c, r) 인덱스를 저장 (팽창됨)
        
        # === BEV 회피 및 마진 파라미터 ===
        self.robot_radius = 0.35  # 로봇 반경 + 안전 마진 (미터)
        self.robot_radius_cells = int(self.robot_radius / self.bev_resolution) # 팽창 반경 (셀 단위)
        
        # 이 거리(미터) 내로 BEV 장애물이 감지되면 회피 시작
        self.bev_avoidance_distance = 2.5
        # BEV 장애물 감지 시 최대 회피 각속도 (rad/s)
        self.bev_avoidance_gain = 0.8
        # BEV 장애물 감지 시 기본 감속 속도
        self.bev_avoidance_slowdown_vel = 0.15


        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None
        # ... (웨이포인트 리스트는 사용자가 제공한 원본을 유지) ...
        d1 = (0.0,0.0) # (-0.138,-0.227) 
        d2 = (2.7,0 ) # (2.516,-0.336) 
        d3 = (2.433,2.274)
        d4 = (-0.223,2.4)
        d5 = (-2.55,5.0)
        d6 = d4
        d7 = d1
        d8 = d2
        d9 = d3 
        d10 =d2
        d11= d1
        d12= d4
        d13= d5
        d14= d4
        d15= d1 
        self.waypoints = [d1,d2,d3,d1,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]
        
        self.waypoint_index = 0
        self.goal_threshold = 0.7

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_planner()

        self.current_depth_tensor = None
        self.angular_gain = 2.0

        self.depth_cv = None

        # === CV2 시각화를 위한 변수들 ===
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 384.0, 384.0, 320.0, 240.0
        self.visualization_image = None
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.start()

        # === Matplotlib 플롯을 위한 데이터 저장 변수 ===
        self.plot_data_lock = threading.Lock()
        self.trajectory_data = []
        self.latest_preds = np.array([])
        self.latest_waypoints = np.array([])
        self.latest_local_goal = np.array([])

        self.get_logger().info("✅ RealSense PlannerNet Control with BEV Avoidance has started.")

        ## Controller Params
        # === 멋진 제어기를 위한 파라미터들 ===
        self.max_linear_velocity = 0.5   # 로봇의 최대 직진 속도 (m/s)
        self.min_linear_velocity = 0.15  # 로봇의 최소 직진 속도 (m/s)
        self.max_angular_velocity = 1.0    # 로봇의 최대 회전 속도 (rad/s)
        
        # Pure Pursuit Controller 파라미터 (현재는 PlannerNet 제어기가 우선)
        self.look_ahead_dist_base = 0.95
        self.look_ahead_dist_k = 0.3
        self.turn_damping_factor = 2.5


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

    def setup_planner(self):
        # model load and data preprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        if torch.cuda.is_available(): self.net = self.net.cuda()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(([360, 640])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded successfully on {self.device}")

    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0  # meter unit
            depth_cv = (np.clip(depth_cv, 0, max_depth_value * 1000) / 1000.0).astype(np.float32)
            depth_cv[depth_cv > max_depth_value] = 0

            depth_cv = depth_cv / max_depth_value
            depth_normalized = (depth_cv * 255).astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # AI 모델 입력용 텐서 생성
            depth_tensor = torch.from_numpy(depth_cv).unsqueeze(0)
            depth_tensor = depth_tensor.repeat(3, 1, 1)
            depth_tensor = TF.resize(depth_tensor, [360, 640])
            depth_tensor = depth_tensor.unsqueeze(0)
            self.current_depth_tensor = depth_tensor.to(self.device)

            # 시각화 스레드에서 사용할 기본 이미지 저장
            with self.plot_data_lock:
                self.visualization_image = depth_display
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    def bev_callback(self, msg: PointCloud2):
        """
        BEV 맵을 수신하고, 'robot_radius'만큼 팽창(inflation)시켜
        점유된 셀 인덱스(set)로 변환합니다.
        """
        raw_occupied_cells = set()
        
        for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
            global_x, global_y = point[0], point[1]
            
            grid_c = int((global_x - self.bev_origin_x) / self.bev_resolution)
            grid_r = int((global_y - self.bev_origin_y) / self.bev_resolution)
            
            if 0 <= grid_c < self.bev_cells_x and 0 <= grid_r < self.bev_cells_y:
                raw_occupied_cells.add((grid_c, grid_r))
        
        # --- 장애물 팽창(Inflation/Dilation) 로직 ---
        inflated_cells = set()
        radius = self.robot_radius_cells
        
        if not raw_occupied_cells:
            with self.bev_lock:
                self.occupied_cells = set()
            return

        # 미리 계산된 팽창 마스크 (원 모양)
        dilation_mask = []
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if dc*dc + dr*dr <= radius*radius:
                    dilation_mask.append((dc, dr))

        for c, r in raw_occupied_cells:
            for dc, dr in dilation_mask:
                inflated_cells.add((c + dc, r + dr)) # 맵 경계 체크는 get_bev_avoidance_cmd에서 수행
        
        with self.bev_lock:
            self.occupied_cells = inflated_cells

    def get_bev_avoidance_cmd(self, current_pose) -> tuple[float, float, float]:
        """
        현재 로봇 위치 기준, 전방의 팽창된 BEV 맵을 스캔하여
        회피 명령(가중치, 선속도, 각속도)을 계산합니다.

        Returns:
            tuple[float, float, float]: (weight, linear_x, angular_z)
            weight (0.0 ~ 1.0): 회피 명령의 가중치. 1.0이면 완전 회피.
            linear_x: 회피 시 권장 선속도
            angular_z: 회피 시 권장 각속도 (장애물을 밀어내는 방향)
        """
        current_x, current_y, current_yaw = current_pose
        
        # 로봇의 글로벌 그리드 좌표 (중심 기준)
        robot_c_float = (current_x - self.bev_origin_x) / self.bev_resolution
        robot_r_float = (current_y - self.bev_origin_y) / self.bev_resolution

        # 글로벌 -> 로컬 변환을 위한 코사인/사인 (반대 방향)
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)

        min_dist = self.bev_avoidance_distance
        nearest_obstacle_local_y = 0.0 # 가장 가까운 장애물의 로컬 y좌표

        with self.bev_lock:
            if not self.occupied_cells:
                return 0.0, 0.0, 0.0 # (weight, linear, angular)

            for c, r in self.occupied_cells:
                # 1. 글로벌 그리드 차이 계산 (셀 중심 기준)
                global_dx = (c + 0.5) * self.bev_resolution + self.bev_origin_x - current_x
                global_dy = (r + 0.5) * self.bev_resolution + self.bev_origin_y - current_y
                
                # 2. 로컬 좌표계로 변환 (로봇 기준)
                local_x = global_dx * cos_yaw + global_dy * sin_yaw
                local_y = -global_dx * sin_yaw + global_dy * cos_yaw
                
                # 3. 로봇의 전방, 회피 거리 내에 있는 장애물만 고려
                if 0.0 < local_x < self.bev_avoidance_distance and \
                   abs(local_y) < (self.bev_avoidance_distance / 2.0):
                    
                    dist = math.sqrt(local_x*local_x + local_y*local_y)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obstacle_local_y = local_y

        # --- 회피 명령 계산 ---
        if min_dist < self.bev_avoidance_distance:
            # 1. 가중치 (Weight): 가까울수록 1.0에 가까워짐
            weight = 1.0 - (min_dist / self.bev_avoidance_distance)
            weight = np.clip(weight * 1.5, 0.0, 1.0) # 더 민감하게 반응
            
            # 2. 각속도 (Angular):
            # 장애물이 왼쪽에(local_y > 0) 있으면, 오른쪽(음수)으로 회전
            # 장애물이 오른쪽에(local_y < 0) 있으면, 왼쪽(양수)으로 회전
            # (math.copysign: 부호만 복사)
            if abs(nearest_obstacle_local_y) > 0.01: # 0으로 나누는 것 방지
                bev_angular_z = -math.copysign(self.bev_avoidance_gain, nearest_obstacle_local_y)
            else:
                bev_angular_z = 0.0 # 정면에 있으면 직진
            
            # 3. 선속도 (Linear): 가까울수록 감속
            bev_linear_x = self.bev_avoidance_slowdown_vel
            
            # (가중치, 회피 선속도, 회피 각속도) 반환
            return weight, bev_linear_x, bev_angular_z
            
        return 0.0, 0.0, 0.0 # 회피 필요 없음

    def control_callback(self):
        if self.current_depth_tensor is None or self.current_pose is None:
            return

        try:
            if self.waypoint_index >= len(self.waypoints):
                # all waypoints complete
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose
                pose_copy = self.current_pose[:] # 회피 및 플로팅용 복사본

            target_wp = self.waypoints[self.waypoint_index]
            distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
            
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info(f"✅ Waypoint {self.waypoint_index} reached!")
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    return
                # 다음 웨이포인트로 갱신
                target_wp = self.waypoints[self.waypoint_index]


            # global difference vector : target - current
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y

            # local goal position : Rotation transform
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)

            # --- 1. PlannerNet 명령 계산 ---
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)
                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1)
                cmd_vels = preds_tensor[:, :, :2]
                fear_val = fear.cpu().item()

                k = 2
                h = 3
                # PlannerNet이 제안하는 각속도
                pn_angular_z = torch.clamp(cmd_vels[0, k:k + h, 1], -1.0, 1.0).mean().cpu().item()
                pn_angular_z = self._discretize_value(pn_angular_z, 0.2)
                
                # PlannerNet이 제안하는 선속도
                pn_linear_x = 0.4
                if pn_angular_z >= 0.4:
                    pn_linear_x = 0.0

            # --- 2. BEV 회피 명령 계산 ---
            # (가중치, 회피 선속도, 회피 각속도)
            bev_weight, bev_linear_x, bev_angular_z = self.get_bev_avoidance_cmd(pose_copy)

            # --- 3. 명령 혼합 (Blending) ---
            # (1-가중치) * PlannerNet + (가중치) * BEV회피
            final_linear_x = (1.0 - bev_weight) * pn_linear_x + bev_weight * bev_linear_x
            final_angular_z = (1.0 - bev_weight) * pn_angular_z + bev_weight * bev_angular_z
            
            bev_active_flag = bev_weight > 0.0 # 시각화용 플래그

            # --- 4. 최종 안전 정지 (Fear) ---
            # if fear_val > 0.7:
            #     self.get_logger().warn(f"!!! PlannerNet FEAR STOP !!! Fear: {fear_val:.2f}")
            #     final_linear_x = 0.0
            #     final_angular_z = 0.0
            
            # --- (주석 처리된 ROI 기반 충돌 회피 로직은 삭제 또는 유지) ---
            # ...

            with self.plot_data_lock:
                self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                self.latest_local_goal = np.array([local_x, local_y])
                
                # CV2 이미지에 그리기
                if self.visualization_image is not None:
                    img_to_draw = self.visualization_image.copy()
                    # (향후 draw_path_and_direction에 bev_active_flag를 넘겨서
                    #  화면에 "BEV AVOIDING" 경고를 띄우도록 수정 가능)
                    final_img = self.draw_path_and_direction(img_to_draw, waypoints_tensor, final_angular_z, fear_val)
                    self.visualization_image = final_img


            # Twist message generate & Publish
            twist = Twist()
            twist.linear.x = float(final_linear_x)
            twist.angular.z = float(final_angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(
                f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | "
                f"CMD: l_x={final_linear_x:.2f} a_z={final_angular_z:.2f} | "
                f"Fear:{fear_val:.2f} BEV_Avoid(w={bev_weight:.2f})"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")

    def _discretize_value(self, value, step):
        return round(value / step) * step

    def _visualization_thread(self):
        """CV2 뎁스 영상과 AI 판단 정보를 보여주는 스레드"""
        self.get_logger().info("Starting CV2 visualization thread.")
        while self.running and rclpy.ok():
            with self.plot_data_lock:
                display_image = self.visualization_image.copy() if self.visualization_image is not None else None
            
            if display_image is not None:
                cv2.imshow("PlannerNet Vision", display_image)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()
        self.get_logger().info("CV2 visualization thread stopped.")

    def draw_path_and_direction(self, image, waypoints_tensor, angular_z, fear_val):
        """뎁스 이미지 위에 경로, 방향, 위험 경고를 그리는 함수"""
        if image is None: return None
        
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        h, w, _ = image.shape

        # 1. 예측 경로 (Waypoints) 그리기
        for point in waypoints:
            wp_x, wp_y = point[0], point[1]
            Z_cam, X_cam = wp_x, -wp_y
            
            if Z_cam > 0.1:
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy)
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10))
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1)

        # 2. 주행 방향 그리기
        arrow_color = (255, 255, 0) # 노란색 (Straight)
        turn_text = "Straight"
        arrow_end = (w // 2, h - 50)
        
        if angular_z > 0.15:
            turn_text, arrow_color, arrow_end = "Turn Left", (0, 255, 255), (w // 2 - 50, h - 50)
        elif angular_z < -0.15:
            turn_text, arrow_color, arrow_end = "Turn Right", (255, 0, 255), (w // 2 + 50, h - 50)
        
        cv2.putText(image, turn_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (w // 2, h - 20), arrow_end, arrow_color, 3)

        # 3. 위험 경고 그리기
        if fear_val > 0.6:
            cv2.putText(image, "!! DANGER - STOP !!", (w // 2 - 200, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif fear_val > 0.4:
            cv2.putText(image, "CAUTION - SLOWING", (w // 2 - 190, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

        return image

    # (주석처리) - Pure Pursuit 방식의 waypoints_to_cmd_vel
    # 현재는 PlannerNet의 예측값을 직접 사용 + BEV 혼합 방식을 사용 중입니다.
    # def waypoints_to_cmd_vel(self, waypoints_tensor):
    #     ... (이전 코드 내용) ...
    #     return linear_x, angular_z

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        
        # 정지 명령 발행
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        
        super().destroy_node()

# Matplotlib 애니메이션 함수
def update_plot(frame, node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj or pose is None:
        return []

    reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:]
    if reached_wps.size > 0: reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else: reached_wps_plot.set_data([], [])
    if pending_wps.size > 0: pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else: pending_wps_plot.set_data([], [])

    traj_arr = np.array(traj)
    traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0])

    current_x, current_y, current_yaw = pose
    
    current_point.set_data([-current_y], [current_x])
    heading_len = 0.5
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

    if preds_local.size > 0 and waypoints_local.size > 0 and goal_local.size > 0:
        rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                              [math.sin(current_yaw),  math.cos(current_yaw)]])
        waypoints_global = (rot_matrix @ waypoints_local[:, :2].T).T + np.array([current_x, current_y])
        preds_global = (rot_matrix @ preds_local[:, :2].T).T + np.array([current_x, current_y])
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
        preds_points.set_data(-preds_global[:, 1], preds_global[:, 0])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])

    return [traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot]

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePlannerControl()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title('Real-time Trajectory and PlannerNet Prediction')
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
    preds_points, = ax.plot([], [], 'ro', markersize=5, label='Preds (Model Output)')
    waypoints_line, = ax.plot([], [], 'y.-', lw=1, label='Waypoints (Path)')
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    ax.legend()
    
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
