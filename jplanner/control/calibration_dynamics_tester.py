#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
import threading
import time

# Matplotlib 시각화
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CalibratingDynamicsTester(Node):
    def __init__(self):
        super().__init__('calibrating_dynamics_tester')

        # --- 파라미터 ---
        self.declare_parameter('control_dt', 0.1)
        self.dt = self.get_parameter('control_dt').get_parameter_value().double_value
        
        # --- (핵심) 보정 계수 파라미터 선언 ---
        self.declare_parameter('v_scale', 1.0) # 선형 속도 보정 계수
        self.declare_parameter('w_scale', 1.0) # 각속도 보정 계수
        
        self.v_scale = self.get_parameter('v_scale').get_parameter_value().double_value
        self.w_scale = self.get_parameter('w_scale').get_parameter_value().double_value
        
        # 파라미터가 실시간으로 변경될 때 호출될 콜백 등록
        self.add_on_set_parameters_callback(self.parameter_callback)

        # --- 테스트할 제어 시퀀스 ---
        self.control_sequence = [
            (0.5, 0.0, 3.0),   (0.0, 0.0, 1.0),
            (0.0, 0.8, 6.0),   (0.0, 0.0, 1.0),
            (0.0, -0.4,12.0),   (0.0, 0.0,1.0),
            (0.5, 0.0, 3.0),   (0.0, 0.0, 1.0),
            
        ]
        self.sequence_index = 0
        self.time_in_step = 0.0
        
        # --- ROS 2 Setup ---
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 ---
        self.actual_pose = None      # [x, y, yaw] (Odom 기준)
        self.last_cmd_vel = (0.0, 0.0) # (v, w)
        # 예측의 기준이 될 '직전' odom 포즈
        self.last_odom_pose_for_pred = None 
        self.initial_pose_set = False

        # --- 시각화 데이터 ---
        self.plot_lock = threading.Lock()
        self.actual_trajectory = []
        self.predicted_trajectory = [] # '한 스텝 예측' 값들을 저장
        
        # 오차(Actual - Predicted) 기록
        self.error_history_x = []
        self.error_history_y = []
        self.error_history_yaw = []

        # 메인 제어/시뮬레이션 타이머
        self.control_timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"✅ Calibrating Dynamics Tester (Live Tuning) has started.")
        self.get_logger().info(f"   v_scale: {self.v_scale:.3f}, w_scale: {self.w_scale:.3f}")
        self.get_logger().info("   Waiting for initial odometry...")
        self.get_logger().info("---")
        self.get_logger().info("HOW TO TUNE:")
        self.get_logger().info("In a new terminal, run:")
        self.get_logger().info("ros2 param set /calibrating_dynamics_tester w_scale 1.05")
        self.get_logger().info("---")

    def parameter_callback(self, params):
        """ROS 2 파라미터가 변경될 때마다 호출됨"""
        success = True
        for param in params:
            if param.name == 'v_scale':
                self.v_scale = param.value
                self.get_logger().info(f"Parameter 'v_scale' updated to: {self.v_scale:.3f}")
            elif param.name == 'w_scale':
                self.w_scale = param.value
                self.get_logger().info(f"Parameter 'w_scale' updated to: {self.w_scale:.3f}")
            else:
                success = False
        return SetParametersResult(successful=success)

    def unicycle_model(self, state, v, w, dt, v_scale, w_scale):
        """
        보정 계수가 적용된 유니사이클 동역학 모델.
        """
        x, y, yaw = state
        
        # 보정 계수 적용
        v_eff = v * v_scale
        w_eff = w * w_scale
        
        x_next = x + v_eff * math.cos(yaw) * dt
        y_next = y + v_eff * math.sin(yaw) * dt
        yaw_next = yaw + w_eff * dt
        
        yaw_next = math.atan2(math.sin(yaw_next), math.cos(yaw_next))
        return [x_next, y_next, yaw_next]

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        """Odom 수신, 예측 수행, 오차 계산"""
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        current_actual_pose = [current_x, current_y, current_yaw]
        
        with self.plot_lock:
            self.actual_pose = current_actual_pose
            self.actual_trajectory.append([current_x, current_y])

            if not self.initial_pose_set:
                self.initial_pose_set = True
                self.get_logger().info(f"   Initial pose set: [{current_x:.2f}, {current_y:.2f}]")
                # 예측 궤적의 시작점도 현재 위치로 초기화
                self.predicted_trajectory.append([current_x, current_y])
                return

            # --- 예측 및 오차 계산 ---
            # prediction & error calibration
            # 'last_odom_pose_for_pred' (timer_callback에서 설정)가 있어야 예측 가능
            if self.last_odom_pose_for_pred is not None:
                v, w = self.last_cmd_vel
                start_pose = self.last_odom_pose_for_pred
                
                # 'start_pose'에서 (v, w) 명령을 줬을 때,
                # 'current_actual_pose'가 되어야 하는데, 모델은 뭐라고 예측하는가?
                predicted_pose = self.unicycle_model(
                    start_pose, v, w, self.dt, 
                    self.v_scale, self.w_scale # (핵심) 실시간 보정값 사용
                )
                
                # 예측 궤적에 추가
                self.predicted_trajectory.append(predicted_pose[:2])
                
                # 오차 계산 (Actual - Predicted)
                err_x = current_actual_pose[0] - predicted_pose[0]
                err_y = current_actual_pose[1] - predicted_pose[1]
                # Yaw 오차 (Wrapping 처리)
                err_yaw = current_actual_pose[2] - predicted_pose[2]
                err_yaw = math.atan2(math.sin(err_yaw), math.cos(err_yaw))
                
                self.error_history_x.append(err_x)
                self.error_history_y.append(err_y)
                self.error_history_yaw.append(err_yaw)
                
                # 오차 로그 (너무 자주 찍히지 않게 조절)
                if len(self.error_history_yaw) % 20 == 0 and (v != 0 or w != 0):
                    avg_yaw_err = np.mean(self.error_history_yaw[-20:])
                    self.get_logger().info(f"Avg Yaw Err (Act-Pred): {avg_yaw_err:+.4f} (w_scale: {self.w_scale:.3f})")

    def timer_callback(self):
        """제어 명령 발행 및 '예측 기준 시점' 저장"""
        
        if not self.initial_pose_set:
            return

        if self.sequence_index >= len(self.control_sequence):
            if not self.control_timer.is_canceled():
                self.get_logger().info("✅ Control sequence finished.")
                self.stop_robot()
                self.control_timer.cancel()
            return

        # 1. 제어 명령 가져오기 및 발행
        v, w, duration = self.control_sequence[self.sequence_index]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        
        # 2. (핵심) 다음 Odom 콜백에서 사용할 정보 저장
        with self.plot_lock:
            # 방금 보낸 명령(v, w)을 저장
            self.last_cmd_vel = (v, w)
            # 이 명령이 적용되기 '직전'의 실제 포즈를 '예측 기준점'으로 저장
            self.last_odom_pose_for_pred = self.actual_pose 

        # 3. 시퀀스 시간 업데이트
        self.time_in_step += self.dt
        if self.time_in_step >= duration:
            self.sequence_index += 1
            self.time_in_step = 0.0
            if self.sequence_index < len(self.control_sequence):
                self.get_logger().info(f"   Moving to step {self.sequence_index}...")

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        time.sleep(0.1)
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.get_logger().info("Shutting down Calibrating Dynamics Tester...")
        self.stop_robot()
        super().destroy_node()

# ==============================================================================
# --- Matplotlib 시각화 함수 (오차 플롯 추가) ---
# ==============================================================================

def update_plot(frame, node: CalibratingDynamicsTester, 
                ax_traj, actual_line, predicted_line, actual_point, predicted_point,
                ax_err, yaw_error_line):
    
    with node.plot_lock:
        actual_traj = np.array(list(node.actual_trajectory))
        predicted_traj = np.array(list(node.predicted_trajectory))
        yaw_errors = list(node.error_history_yaw)

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
        
        # Y축 범위 동적 조절
        min_err = np.min(yaw_errors)
        max_err = np.max(yaw_errors)
        padding = max(0.05, (max_err - min_err) * 0.1)
        ax_err.set_ylim(min_err - padding, max_err + padding)
        
        artists.append(yaw_error_line)

    return artists


def main(args=None):
    rclpy.init(args=args)
    node = CalibratingDynamicsTester() 

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정 (2개의 플롯: 궤적, 오차)
    fig, (ax_traj, ax_err) = plt.subplots(
        2, 1, figsize=(10, 13), 
        gridspec_kw={'height_ratios': [3, 1]}, # 궤적 플롯을 더 크게
        constrained_layout=True
    )
    
    # --- 궤적 플롯 (ax_traj) 설정 ---
    ax_traj.set_title('Dynamics Model Calibration (Live Tuning)', fontsize=14)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.grid(True)
    ax_traj.set_aspect('equal', adjustable='box')
    
    actual_line, = ax_traj.plot([], [], 'b-', lw=2, label='Actual Trajectory (Odom)')
    predicted_line, = ax_traj.plot([], [], 'r--', lw=2, label='Predicted Trajectory (Model)')
    actual_point, = ax_traj.plot([], [], 'bo', markersize=8, label='Actual Position')
    predicted_point, = ax_traj.plot([], [], 'ro', markersize=8, label='Predicted (1-Step Ahead)')
    ax_traj.legend(loc='upper right', fontsize=10)
    
    # --- 오차 플롯 (ax_err) 설정 ---
    ax_err.set_title('Real-time Yaw Error (Actual - Predicted)', fontsize=12)
    ax_err.set_xlabel('Time (steps)')
    ax_err.set_ylabel('Yaw Error (rad)')
    ax_err.grid(True)
    ax_err.axhline(0, color='black', linestyle='--', lw=1) # 0 기준선
    yaw_error_line, = ax_err.plot([], [], 'g-', lw=1, label='Yaw Error')
    ax_err.legend(loc='upper right', fontsize=9)

    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax_traj, actual_line, predicted_line, actual_point, predicted_point,
               ax_err, yaw_error_line),
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
        print("Dynamics tester shutdown complete.")


if __name__ == '__main__':
    main()
