#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
import threading
import time

# Matplotlib 시각화
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 새로 추가된 임포트 ---
import torch
from dynamics_predictor_nn import DynamicsPredictor # 1번 파일 임포트
# --- ---

class CalibratingDynamicsLearner(Node):
    def __init__(self):
        super().__init__('calibrating_dynamics_learner')

        # --- 파라미터 ---
        self.declare_parameter('control_dt', 0.1)
        self.dt = self.get_parameter('control_dt').get_parameter_value().double_value
        
        # --- (핵심) NN 기반 Dynamics Predictor 초기화 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = DynamicsPredictor(self.device, self.dt, learning_rate=1e-3)
        self.get_logger().info(f"✅ DynamicsPredictor (NN) initialized on {self.device}.")
        
        # --- (삭제) v_scale, w_scale 및 파라미터 콜백 제거 ---

        # --- 테스트할 제어 시퀀스 (동일) ---
        self.control_sequence = [
            (0.5, 0.0, 3.0),   (0.0, 0.0, 1.0),
            (0.0, 0.8, 6.0),   (0.0, 0.0, 1.0),
            (0.0, -0.4, 12.0), (0.0, 0.0, 1.0),
            (0.5, 0.0, 3.0),   (0.0, 0.0, 1.0),
        ]





        # --- (핵심) 보정 계수 파라미터 선언 ---
        # (v, w, duration_sec)
        # v: linear.x, w: angular.z
        
        # 로봇의 최대 속도/각속도를 고려하여 값을 설정하세요.
        max_v = 0.8  # 예: 로봇의 최대 선속도
        max_w = 1.0  # 예: 로봇의 최대 각속도
        
        self.control_sequence = [
            # 1. 기본 축 테스트 (직진, 정지, 제자리 회전)
            (0.5 * max_v, 0.0, 5.0),         # 1. 중간 속도 직진
            (0.0, 0.0, 2.0),                 # 2. 정지
            (0.0, 0.8 * max_w, 5.0),         # 3. 중간 속도 좌측 제자리 회전
            (0.0, 0.0, 2.0),                 # 4. 정지
            (0.0, -0.8 * max_w, 5.0),        # 5. 중간 속도 우측 제자리 회전
            (0.0, 0.0, 2.0),                 # 6. 정지
            (max_v, 0.0, 5.0),               # 7. 최대 속도 직진
            (0.0, 0.0, 2.0),                 # 8. 정지

            # 2. (중요) 결합된 모션 - 곡선 주행
            # (v > 0, w > 0) -> 전진하며 좌회전
            (0.5 * max_v, 0.5 * max_w, 5.0), # 9. (중간v, 중간w) 좌회전 커브
            (0.0, 0.0, 2.0),                 # 10. 정지
            (max_v, 0.3 * max_w, 5.0),       # 11. (최대v, 약한w) 고속 좌회전 커브
            (0.0, 0.0, 2.0),                 # 12. 정지
            (0.3 * max_v, max_w, 4.0),       # 13. (약한v, 최대w) 저속 급 좌회전
            (0.0, 0.0, 2.0),                 # 14. 정지

            # 3. (중요) 결합된 모션 - 반대쪽 곡선 주행
            # (v > 0, w < 0) -> 전진하며 우회전
            (0.5 * max_v, -0.5 * max_w, 5.0),# 15. (중간v, 중간w) 우회전 커브
            (0.0, 0.0, 2.0),                 # 16. 정지
            (max_v, -0.3 * max_w, 5.0),      # 17. (최대v, 약한w) 고속 우회전 커브
            (0.0, 0.0, 2.0),                 # 18. 정지
            (0.3 * max_v, -max_w, 4.0),      # 19. (약한v, 최대w) 저속 급 우회전
            (0.0, 0.0, 2.0),                 # 20. 정지

            # 4. (선택) 후진 테스트 (로봇이 후진을 지원하고, 후진 시 coeff가 다르다면)
            # (-0.3 * max_v, 0.0, 3.0),        # 21. 후진
            # (0.0, 0.0, 2.0),                 # 22. 정지
            # (-0.3 * max_v, 0.4 * max_w, 4.0),# 23. 후진하며 커브 (v<0, w>0)
            # (0.0, 0.0, 2.0),                 # 24. 정지
            # (-0.3 * max_v, -0.4 * max_w, 4.0),# 25. 후진하며 커브 (v<0, w<0)
            # (0.0, 0.0, 2.0),                 # 26. 정지
        ]



        self.sequence_index = 0
        self.time_in_step = 0.0
        
        # --- ROS 2 Setup (동일) ---
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 (동일) ---
        self.actual_pose = None      # [x, y, yaw] (Odom 기준)
        self.last_cmd_vel = (0.0, 0.0) # (v, w)
        self.last_odom_pose_for_pred = None 
        self.initial_pose_set = False

        # --- 시각화 데이터 (동일) ---
        self.plot_lock = threading.Lock()
        self.actual_trajectory = []
        self.predicted_trajectory = [] 
        self.error_history_x = []
        self.error_history_y = []
        self.error_history_yaw = []
        
        # (추가) 학습 손실(Loss) 기록
        self.loss_history = [] 

        # 메인 제어/시뮬레이션 타이머
        self.control_timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"✅ Calibrating Dynamics Learner (NN) has started.")
        self.get_logger().info("---")
        self.get_logger().info("Waiting for initial odometry...")


    # --- (삭제) parameter_callback 제거 ---
    # --- (삭제) unicycle_model 제거 (Predictor가 대체) ---
    
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        """Odom 수신, (핵심) 학습 수행, 예측 수행, 오차 계산"""
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        current_actual_pose = [current_x, current_y, current_yaw]
        
        with self.plot_lock:
            self.actual_pose = current_actual_pose
            self.actual_trajectory.append([current_x, current_y])

            if not self.initial_pose_set:
                self.initial_pose_set = True
                self.get_logger().info(f"  Initial pose set: [{current_x:.2f}, {current_y:.2f}]")
                self.predicted_trajectory.append([current_x, current_y])
                return

            # --- (핵심) 학습 및 예측 ---
            if self.last_odom_pose_for_pred is not None:
                v, w = self.last_cmd_vel
                start_pose_list = self.last_odom_pose_for_pred
                
                # --- 1. 데이터를 PyTorch 텐서로 변환 (배치 크기 K=1) ---
                start_state_tensor = torch.tensor([start_pose_list], dtype=torch.float32, device=self.device)
                control_tensor = torch.tensor([[v, w]], dtype=torch.float32, device=self.device)
                actual_next_state_tensor = torch.tensor([current_actual_pose], dtype=torch.float32, device=self.device)
                
                # --- 2. 학습 (Learning Step) ---
                # (v, w)가 0이 아닐 때만 학습 (정지 상태 데이터는 제외)
                if v != 0.0 or w != 0.0:
                    loss = self.predictor.learn(start_state_tensor, control_tensor, actual_next_state_tensor)
                    self.loss_history.append(loss)
                
                # --- 3. 예측 (Prediction for Plotting) ---
                # 방금 학습된 모델로 예측 수행
                with torch.no_grad():
                    predicted_pose_tensor = self.predictor.motion_model(
                        start_state_tensor, control_tensor
                    )
                    predicted_pose = predicted_pose_tensor.cpu().numpy()[0]
                
                # 예측 궤적에 추가
                self.predicted_trajectory.append(predicted_pose[:2])
                
                # --- 4. 오차 계산 (Actual - Predicted) (시각화용) ---
                err_x = current_actual_pose[0] - predicted_pose[0]
                err_y = current_actual_pose[1] - predicted_pose[1]
                err_yaw = current_actual_pose[2] - predicted_pose[2]
                err_yaw = math.atan2(math.sin(err_yaw), math.cos(err_yaw))
                
                self.error_history_x.append(err_x)
                self.error_history_y.append(err_y)
                self.error_history_yaw.append(err_yaw)
                
                # 로그 (Loss 값도 함께 출력)
                if len(self.error_history_yaw) % 20 == 0 and (v != 0 or w != 0):
                    avg_loss = np.mean(self.loss_history[-20:])
                    avg_yaw_err = np.mean(self.error_history_yaw[-20:])
                    self.get_logger().info(f"Avg Loss: {avg_loss:.6f} | Avg Yaw Err: {avg_yaw_err:+.4f}")

    def timer_callback(self):
        """제어 명령 발행 및 '예측 기준 시점' 저장 (기존과 거의 동일)"""
        
        if not self.initial_pose_set:
            return

        if self.sequence_index >= len(self.control_sequence):
            if not self.control_timer.is_canceled():
                self.get_logger().info("✅ Control sequence finished.")
                self.stop_robot()
                self.control_timer.cancel()
                
                # (추가) 학습 완료 후 모델 저장
                try:
                    torch.save(self.predictor.calibration_net.state_dict(), 'calibration_net_final.pth')
                    self.get_logger().info("✅ NN Model weights saved to 'calibration_net_final.pth'")
                except Exception as e:
                    self.get_logger().error(f"Failed to save model: {e}")
            return

        # 1. 제어 명령 가져오기 및 발행
        v, w, duration = self.control_sequence[self.sequence_index]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        
        # 2. 다음 Odom 콜백에서 사용할 정보 저장
        with self.plot_lock:
            self.last_cmd_vel = (v, w)
            self.last_odom_pose_for_pred = self.actual_pose 

        # 3. 시퀀스 시간 업데이트 (동일)
        self.time_in_step += self.dt
        if self.time_in_step >= duration:
            self.sequence_index += 1
            self.time_in_step = 0.0
            if self.sequence_index < len(self.control_sequence):
                self.get_logger().info(f"  Moving to step {self.sequence_index}...")

    def stop_robot(self):
        # (동일)
        twist = Twist()
        self.cmd_pub.publish(twist)
        time.sleep(0.1)
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        # (동일)
        self.get_logger().info("Shutting down Calibrating Dynamics Learner...")
        self.stop_robot()
        super().destroy_node()

# ==============================================================================
# --- Matplotlib 시각화 함수 (오차 플롯 + Loss 플롯) ---
# ==============================================================================

def update_plot(frame, node: CalibratingDynamicsLearner,  
              ax_traj, actual_line, predicted_line, actual_point, predicted_point,
              ax_err, yaw_error_line,
              ax_loss, loss_line): # Loss 플롯 아티스트 추가
    
    with node.plot_lock:
        actual_traj = np.array(list(node.actual_trajectory))
        predicted_traj = np.array(list(node.predicted_trajectory))
        yaw_errors = list(node.error_history_yaw)
        losses = list(node.loss_history)

    artists = []

    # --- 1. 궤적 플롯 (ax_traj) ---
    if actual_traj.size > 0 and predicted_traj.size > 0:
        actual_line.set_data(actual_traj[:, 0], actual_traj[:, 1])
        predicted_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])
        
        actual_point.set_data(actual_traj[-1, 0], actual_traj[-1, 1])
        predicted_point.set_data(predicted_traj[-1, 0], predicted_traj[-1, 1])
        
        all_points_plot = np.vstack((actual_traj, predicted_traj))
        x_min, y_min = np.min(all_points_plot, axis=0) - 1.0
        x_max, y_max = np.max(all_points_plot, axis=0) + 1.0
        ax_traj.set_xlim(x_min, x_max)
        ax_traj.set_ylim(y_min, y_max)
        
        artists.extend([actual_line, predicted_line, actual_point, predicted_point])

    # --- 2. Yaw 오차 플롯 (ax_err) ---
    if yaw_errors:
        steps = range(len(yaw_errors))
        yaw_error_line.set_data(steps, yaw_errors)
        ax_err.set_xlim(0, len(yaw_errors) + 10)
        
        min_err = np.min(yaw_errors)
        max_err = np.max(yaw_errors)
        padding = max(0.05, (max_err - min_err) * 0.1)
        ax_err.set_ylim(min_err - padding, max_err + padding)
        
        artists.append(yaw_error_line)

    # --- 3. Loss 플롯 (ax_loss) ---
    if losses:
        steps = range(len(losses))
        loss_line.set_data(steps, losses)
        ax_loss.set_xlim(0, len(losses) + 10)
        
        min_loss = 0.0
        max_loss = np.max(losses)
        padding = max(0.001, (max_loss - min_loss) * 0.1)
        ax_loss.set_ylim(min_loss - padding, max_loss + padding)
        
        artists.append(loss_line)

    return artists


def main(args=None):
    rclpy.init(args=args)
    node = CalibratingDynamicsLearner() 

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정 (3개의 플롯: 궤적, 오차, 손실)
    fig, (ax_traj, ax_err, ax_loss) = plt.subplots(
        3, 1, figsize=(10, 16), 
        gridspec_kw={'height_ratios': [3, 1, 1]}, # 궤적 플롯을 가장 크게
        constrained_layout=True
    )
    
    # --- 궤적 플롯 (ax_traj) 설정 ---
    ax_traj.set_title('Dynamics Model Calibration (Live Learning)', fontsize=14)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.grid(True)
    ax_traj.set_aspect('equal', adjustable='box')
    
    actual_line, = ax_traj.plot([], [], 'b-', lw=2, label='Actual Trajectory (Odom)')
    predicted_line, = ax_traj.plot([], [], 'r--', lw=2, label='Predicted Trajectory (NN Model)')
    actual_point, = ax_traj.plot([], [], 'bo', markersize=8, label='Actual Position')
    predicted_point, = ax_traj.plot([], [], 'ro', markersize=8, label='Predicted (1-Step Ahead)')
    ax_traj.legend(loc='upper right', fontsize=10)
    
    # --- 오차 플롯 (ax_err) 설정 ---
    ax_err.set_title('Real-time Yaw Error (Actual - Predicted)', fontsize=12)
    ax_err.set_xlabel('Time (steps)')
    ax_err.set_ylabel('Yaw Error (rad)')
    ax_err.grid(True)
    ax_err.axhline(0, color='black', linestyle='--', lw=1)
    yaw_error_line, = ax_err.plot([], [], 'g-', lw=1, label='Yaw Error')
    ax_err.legend(loc='upper right', fontsize=9)
    
    # --- Loss 플롯 (ax_loss) 설정 ---
    ax_loss.set_title('Real-time Training Loss', fontsize=12)
    ax_loss.set_xlabel('Training Steps')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.grid(True)
    loss_line, = ax_loss.plot([], [], 'm-', lw=1, label='Total Loss')
    ax_loss.legend(loc='upper right', fontsize=9)
    ax_loss.set_yscale('log') # Loss는 log 스케일로 보는 것이 유용할 수 있습니다.

    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax_traj, actual_line, predicted_line, actual_point, predicted_point,
               ax_err, yaw_error_line,
               ax_loss, loss_line), # fargs에 loss 아티스트 추가
        interval=100, blit=True
    )

    try:
        plt.show() 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()
        print("Dynamics learner shutdown complete.")


if __name__ == '__main__':
    main()
