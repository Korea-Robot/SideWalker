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
import torchvision.transforms.functional as TF 
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
        # self.odom_sub = self.create_subscription(Odometry, '/command_odom', self.odom_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/rko_lio/odometry', self.odom_callback, 10)
        
        # Odometry 및 웨이포인트 관련 변수
        self.current_pose = None
        # self.waypoints = [(0.0, 0.0),(5.0, 0.0), (5.0, 10.0), (20.0, 10.0), (20.0, 50.0),(-20.0, 50.0),(-20.0,10.0),(5.0,10.0)]
        # self.waypoints = [(0.0, 0.0),(0.0, 3.0), (0.0, 0.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0)] # self rotation 3



        d1 = (0.0,0.0) # (-0.138,-0.227) 
        d2 = (2.7,0 ) # (2.516,-0.336) 
        d3 = (2.433,2.274)
        d4 = d2 #(-2.55,5.0)
        d5 = d1 #(-0.223,2.4)
        d6 = d2
        d7 = d3
        d8 = d2
        d9 = d1 
        d10 =d2
        d11= d3
        d12= d2
        d13= d1 

        self.waypoints = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13]

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
        
        # self.waypoints = [(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0),(3.0, 0.0), (3.0, 3.0), (0.0, 3.0),(0.0, 0.0)] # self rotation 3
        
        self.waypoint_index = 0 # len(self.waypoints)
        self.goal_threshold = 0.6

        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.setup_planner()

        self.current_depth_tensor = None
        self.angular_gain = 2.0

        self.depth_cv = None

        # === CV2 시각화를 위한 변수들 ===
        # Intel RealSense D435의 일반적인 내장 파라미터 (640x480 기준)
        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 384.0, 384.0, 320.0, 240.0 # why we need?
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

        self.get_logger().info("✅ RealSense PlannerNet Control with Integrated Plotting has started.")


        ## Controller Params
        
        # === 멋진 제어기를 위한 파라미터들 ===
        self.max_linear_velocity = 0.5  # 로봇의 최대 직진 속도 (m/s)
        self.min_linear_velocity = 0.15  # 로봇의 최소 직진 속도 (m/s)
        self.max_angular_velocity = 1.0   # 로봇의 최대 회전 속도 (rad/s)
        
        # Pure Pursuit Controller 파라미터
        self.look_ahead_dist_base = 0.95  # 기본 전방 주시 거리 (m)
        self.look_ahead_dist_k = 0.3     # 속도에 비례한 전방 주시 거리 계수
        self.turn_damping_factor = 2.5   # 회전 시 감속을 위한 계수 (클수록 급격히 감속)
        


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
        # config_path = os.path.join(os.path.dirname(os.getcwd()), 'config', 'training_config.json')
        # with open(config_path) as f: 
        #     config = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "./models/plannernet.pt"
        self.net, _ = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        if torch.cuda.is_available(): self.net = self.net.cuda()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((config['dataConfig']['crop-size'])),
            transforms.Resize(([360, 640])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.traj_cost = TrajCost(0 if not torch.cuda.is_available() else 0)
        self.get_logger().info(f"PlannerNet model loaded successfully on {self.device}")


    def depth_callback(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            max_depth_value = 10.0 # meter  unit 
            depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32) # mm => meter range change 
            depth_cv[depth_cv>max_depth_value] = 0 # over max depth value is zero value

            depth_cv = depth_cv / max_depth_value
            # 뎁스 이미지를 컬러맵으로 변환하여 시각화용 이미지 생성
            depth_normalized = (depth_cv* 255).astype(np.uint8)
            # depth_normalized = (depth_cv/max_depth_value* 255).astype(np.uint8)
            # depth_normalized = (depth_cv / max_depth_value * 255).astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # AI 모델 입력용 텐서 생성


            depth_tensor = torch.from_numpy(depth_cv).unsqueeze(0)
            depth_tensor = depth_tensor.repeat(3,1,1)
            depth_tensor = TF.resize(depth_tensor,[360,640])
            depth_tensor = depth_tensor.unsqueeze(0)
            self.current_depth_tensor = depth_tensor.to(self.device)

            # self.current_depth_tensor = self.transform(depth_cv).unsqueeze(0).to(self.device)
    
            
            # 시각화 스레드에서 사용할 기본 이미지 저장
            with self.plot_data_lock:
                self.visualization_image = depth_display
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

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

            target_wp = self.waypoints[self.waypoint_index]
            with self.plot_data_lock:
                current_x, current_y, current_yaw = self.current_pose

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

            # global coordinate 
            target_wp = self.waypoints[self.waypoint_index]

            # global difference vector : target - current 
            dx_global, dy_global = target_wp[0] - current_x, target_wp[1] - current_y
            
            # local goal position : Rotation transform 
            local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
            local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
            local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            # tensor([[5., 0., 0.]], device='cuda:0')
            
            with torch.no_grad():
                preds_tensor, fear = self.net(self.current_depth_tensor, local_goal_tensor)

                # only use preds_tensor. 

                waypoints_tensor = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds_tensor, step=0.1) # meter unit.

                # (Pdb) preds_tensor (x,y,z) : right-handed coordinate 
                # shape (1,5,3)
                # tensor([[[ 7.3114e-01, -8.5702e-01, -5.5543e-04],
                #         [ 1.6043e+00, -1.3735e+00, -5.4416e-04],
                #         [ 2.6050e+00, -1.4175e+00,  2.5468e-03],
                #         [ 3.8897e+00, -7.5083e-01,  7.0542e-03],
                #         [ 4.9303e+00, -2.1319e-02,  4.4444e-03]]], device='cuda:0')
                

                # cmd_vels = self.waypoints_to_cmd_vel(waypoints_tensor)
                # shape = (1,5,2)
                # tensor([[
                # [11.2652, -4.3225],
                # [10.1447,  3.3032],
                # [10.0171,  4.9026],
                # [14.4730,  5.2260],
                # [12.7088,  1.3271]]], device='cuda:0')
                
                # direct cmd vel like 
                cmd_vels = preds_tensor[:,:,:2]

                fear_val = fear.cpu().item()

                # select k preds  and k+H preds mean 
                k =2
                h = 3
                ############################################################## use pred waypoints directly control
                angular_z = torch.clamp(cmd_vels[0, k:k+h, 1], -1.0, 1.0).mean().cpu().item()

                angular_z = self._discretize_value(angular_z,0.2)
                # angular_z = 1.5*angular_z

                # main controller 

                linear_x= 0.4
                if angular_z >=0.4:
                    linear_x = 0 


                
                stop_distance =  0.01 #meter # 0.77

                width,height = 640,360

                roi_x_start = int(width * 0.4)
                roi_x_end   = int(width * 0.6)
                roi_y_start = int(height* 0.4)
                roi_y_end   = int(height* 0.6)

                front_roi = self.current_depth_tensor[:,0,roi_y_start:roi_y_end,roi_x_start:roi_x_end]
                # self.get_logger().info(f"depth :{front_roi:.2f}")

                # breakpoint()
                # # breakpoint()


                # collision probability 
                # if fear_val > 0.7:
                #     linear_x=0.0
                #     angular_z = 0.0
                
                # # non-collision case => change to  categorical distribution 
                # else:
                #     if abs(angular_z)> 0.15:
                #         linear_x = 0.0
                #     else:
                #         linear_x = np.clip(linear_x,0.2,0.5)

                # linear_x,angular_z  = self.waypoints_to_cmd_vel(waypoints_tensor)

                if torch.mean(front_roi).item() < stop_distance:
                    linear_x   = 0.0
                    angular_z  = 0.0


                with self.plot_data_lock:
                    self.latest_preds = preds_tensor.squeeze().cpu().numpy()
                    self.latest_waypoints = waypoints_tensor.squeeze().cpu().numpy()
                    self.latest_local_goal = np.array([local_x, local_y])
                    
                    # CV2 이미지에 그리기
                    if self.visualization_image is not None:
                        img_to_draw = self.visualization_image.copy()
                        # 강화된 그리기 함수 호출
                        final_img = self.draw_path_and_direction(img_to_draw, waypoints_tensor, angular_z, fear_val)
                        self.visualization_image = final_img


            # Twist message generate & Publish
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z= float(angular_z)
            self.cmd_pub.publish(twist)

            self.get_logger().info(f"WP[{self.waypoint_index}]->({local_x:.1f},{local_y:.1f}) | CMD : linear_x ={linear_x:.2f} angular_z = {angular_z:.2f} Fear:{fear_val:.2f}")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
    
    def _discretize_value(self,value,step):
        return round(value/step)*step
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
                # 이미지가 아직 준비되지 않았으면 잠시 대기
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
            # 로봇 기준 좌표 (X: 전방, Y: 좌측)
            wp_x, wp_y = point[0], point[1]
            
            # 카메라 좌표계로 변환 (Z: 전방, X: 우측)
            Z_cam, X_cam = wp_x, -wp_y
            
            # 2D 이미지에 투영 (Pinhole Camera Model)
            if Z_cam > 0.1: # 10cm 이상 앞에 있는 점만 그림
                u = int(self.cam_fx * (X_cam / Z_cam) + self.cam_cx)
                v = int(self.cam_fy * (-0.1 / Z_cam) + self.cam_cy) # 로봇 높이(약 10cm) 고려
                
                if 0 <= u < w and 0 <= v < h:
                    radius = int(np.clip(8 / Z_cam, 2, 10)) # 멀수록 작게 그림
                    cv2.circle(image, (u, v), radius, (0, 255, 0), -1) # 초록색 점

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

    # don't use!! this!
    # def waypoints_to_cmd_vel(self, waypoints, dt=0.1): # 기존과 동일
    #     if waypoints.shape[1] < 2: return torch.zeros(1, 1, 2, device=waypoints.device)
    #     dx = waypoints[:, 1:, 0] - waypoints[:, :-1, 0]
    #     dy = waypoints[:, 1:, 1] - waypoints[:, :-1, 1]
    #     linear_x = torch.sqrt(dx**2 + dy**2) / dt
    #     heading_angles = torch.atan2(dy, dx)
    #     angular_z = torch.zeros_like(linear_x)
    #     if waypoints.shape[1] > 2:
    #         angle_diff = heading_angles[:, 1:] - heading_angles[:, :-1]
    #         angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    #         angular_z[:, 1:] = angle_diff / dt
    #     angular_z[:, 0] = heading_angles[:, 0] / (dt * self.angular_gain)
    #     # return torch.stack([linear_x, angular_z], dim=-1)
    #     return linear_x,angular_z


    def waypoints_to_cmd_vel(self, waypoints_tensor):
        """
        주어진 지역 경로(waypoints)를 바탕으로 Pure Pursuit 알고리즘을 사용해
        최적의 선속도(linear_x)와 각속도(angular_z)를 계산합니다.

        Args:
            waypoints_tensor (torch.Tensor): 모델이 예측한 로봇 기준 지역 경로. (N, 3) 형태.

        Returns:
            (float, float): 계산된 linear_x, angular_z
        """
        waypoints = waypoints_tensor.squeeze().cpu().numpy()
        
        # 1. 속도에 따라 동적으로 전방 주시 거리(Look-Ahead Distance) 결정
        #    저속에서는 더 정교하게, 고속에서는 더 부드럽게 주행하도록 합니다.
        #    (현재 속도를 모르므로, 우선 기본값으로 계산하고 나중에 선속도에 따라 조절)
        look_ahead_dist = 1 # self.look_ahead_dist_base 

        # 2. 경로 상에서 '전방 주시 점' 찾기
        #    로봇(원점)에서 look_ahead_dist 거리만큼 떨어진 경로 위의 점을 찾습니다.
        target_point = None
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            # 두 웨이포인트 사이의 거리가 0에 가까우면 무시
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue

            # 원점에서 가장 가까운 경로상의 점을 찾아 거리 계산
            dist_to_path = np.linalg.norm(np.cross(p2-p1, p1)) / np.linalg.norm(p2-p1)
            
            # 원과 선분의 교점을 찾는 방식으로 target_point를 근사합니다.
            # 경로가 로봇과 멀어지기 시작하는 지점 중 가장 가까운 점을 목표로 설정
            if p2[0] > 0 and np.linalg.norm(p2) > look_ahead_dist:
                 target_point = p2
                 break
        
        # 만약 적절한 target_point를 찾지 못했다면, 가장 마지막 웨이포인트를 목표로 삼습니다.
        if target_point is None:
            target_point = waypoints[-1]
            if np.linalg.norm(target_point) < 0.1: # 목표가 너무 가까우면 정지
                 return 0.0, 0.0

        # 3. 최적 각속도(angular_z) 계산
        #    로봇이 target_point를 향해 회전하기 위한 곡률(curvature)을 계산합니다.
        #    alpha: 로봇의 현재 방향과 target_point 사이의 각도
        alpha = math.atan2(target_point[1], target_point[0])
        
        # 곡률 = 2 * sin(alpha) / Ld. 여기서 Ld는 실제 target_point까지의 거리
        distance_to_target = np.linalg.norm(target_point)
        curvature = (2.0 * math.sin(alpha)) / distance_to_target
        angular_z = curvature * self.max_linear_velocity # 임시 선속도로 각속도 계산
        
        # 4. 최적 선속도(linear_x) 계산
        #    회전이 클수록(곡률이 클수록) 선속도를 줄여 안정적인 코너링을 수행합니다.
        linear_x = self.max_linear_velocity / (1 + self.turn_damping_factor * abs(curvature))
        linear_x = np.clip(linear_x, self.min_linear_velocity, self.max_linear_velocity)

        # 5. 최종 제어값 정규화 (클리핑)
        angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 재계산: 최종 선속도에 맞춰 각속도를 다시 한번 스케일링
        angular_z = curvature * linear_x
        angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)

        return linear_x, angular_z
    

    
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.running = False
        self.vis_thread.join()
        super().destroy_node()

# Matplotlib 애니메이션 함수 (이전과 동일)
def update_plot(frame, node, ax, traj_line, preds_points, waypoints_line, current_point, heading_line, goal_point, reached_wps_plot, pending_wps_plot):
    # ... (내용은 이전과 동일하므로 생략) ...
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        preds_local = node.latest_preds.copy()
        waypoints_local = node.latest_waypoints.copy()
        goal_local = node.latest_local_goal.copy()
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj:
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

    # Matplotlib 설정 (이전과 동일)
    fig, ax = plt.subplots(figsize=(10, 10),constrained_layout=True)
    # ... (내용은 이전과 동일하므로 생략) ...
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
