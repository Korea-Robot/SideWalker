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

class CorrectingDynamicsTester(Node):
    def __init__(self):
        super().__init__('correcting_dynamics_tester')

        # --- 파라미터 ---
        self.declare_parameter('control_dt', 0.1) # 제어 및 시뮬레이션 주기 (초)
        self.dt = self.get_parameter('control_dt').get_parameter_value().double_value

        # --- 테스트할 제어 시퀀스 ---
        # (linear_vel, angular_vel, duration_sec)
        self.control_sequence = [
            (0.5, 0.0, 3.0),   # 3초간 직진 (0.5 m/s)
            (0.0, 0.0, 1.0),   # 1초간 정지
            (0.0, 0.8, 3.0),   # 3초간 좌회전 (0.8 rad/s)
            (0.0, 0.0, 1.0),   # 1초간 정지
            (0.3, 0.4, 5.0),   # 5초간 곡선 주행
            (0.0, 0.0, 2.0)    # 2초간 정지
        ]
        self.sequence_index = 0
        self.time_in_step = 0.0
        
        # --- ROS 2 Setup ---
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/krm_auto_localization/odom', self.odom_callback, 10)

        # --- 상태 변수 ---
        self.actual_pose = None      # [x, y, yaw] (Odom 기준)
        # self.predicted_pose는 더 이상 누적 상태로 사용하지 않음
        self.initial_pose_set = False

        # --- 시각화 데이터 ---
        self.plot_lock = threading.Lock()
        self.actual_trajectory = []
        self.predicted_trajectory = [] # '한 스텝 예측' 값들을 저장할 리스트

        # 메인 제어/시뮬레이션 타이머
        self.control_timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"✅ Correcting Dynamics Tester (One-Step) has started.")
        self.get_logger().info("   Waiting for initial odometry...")


    def unicycle_model(self, state, v, w, dt):
        """
        간단한 유니사이클 동역학 모델.
        :param state: [x, y, yaw] 현재 상태
        :param v: linear velocity
        :param w: angular velocity
        :param dt: time step
        :return: [x_next, y_next, yaw_next] 다음 상태
        """
        x, y, yaw = state
        
        x_next = x + v * math.cos(yaw) * dt
        y_next = y + v * math.sin(yaw) * dt
        yaw_next = yaw + w * dt
        
        # Yaw 값을 -pi ~ +pi 범위로 정규화
        yaw_next = math.atan2(math.sin(yaw_next), math.cos(yaw_next))
        
        return [x_next, y_next, yaw_next]

    def quaternion_to_yaw(self, q):
        """Odometry 메시지의 Quaternion을 Yaw 각도로 변환"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        """실제 로봇의 위치(Odom)를 수신"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        with self.plot_lock:
            self.actual_pose = [x, y, yaw]
            self.actual_trajectory.append([x, y])

            # 시뮬레이션 시작 위치 설정 (가장 처음 Odom 메시지 기준)
            if not self.initial_pose_set:
                self.initial_pose_set = True
                # 예측 궤적의 시작점도 현재 위치로 초기화
                self.predicted_trajectory.append([x, y])
                self.get_logger().info(f"   Initial pose set: [{x:.2f}, {y:.2f}, {yaw:.2f}]")

    def timer_callback(self):
        """제어 명령 발행 및 '보정된' 동역학 모델 시뮬레이션을 위한 메인 루프"""
        
        # 아직 Odom 수신 전(실제 위치를 모름)이면 대기
        if not self.initial_pose_set or self.actual_pose is None:
            return

        # 제어 시퀀스가 종료되었는지 확인
        if self.sequence_index >= len(self.control_sequence):
            if self.control_timer.is_canceled(): return
            
            self.get_logger().info("✅ Control sequence finished.")
            self.stop_robot()
            self.control_timer.cancel() # 타이머 중지
            return

        # 1. 현재 시퀀스의 제어 명령 가져오기
        v, w, duration = self.control_sequence[self.sequence_index]

        # 2. 로봇에게 제어 명령 발행
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        # 3. (핵심 변경) 동역학 모델로 '다음 1스텝' 예측
        with self.plot_lock:
            # 예측의 시작점을 '누적된 예측값'이 아닌 '현재 실제 Odom 위치'로
            # '보정(Correction)'합니다.
            one_step_prediction = self.unicycle_model(self.actual_pose, v, w, self.dt)
            
            # 이 '한 스텝 예측' 값을 궤적 리스트에 추가합니다.
            self.predicted_trajectory.append(one_step_prediction[:2])

        # 4. 시퀀스 시간 업데이트
        self.time_in_step += self.dt
        if self.time_in_step >= duration:
            self.sequence_index += 1
            self.time_in_step = 0.0
            
            if self.sequence_index < len(self.control_sequence):
                next_v, next_w, _ = self.control_sequence[self.sequence_index]
                self.get_logger().info(
                    f"   Moving to step {self.sequence_index}: v={next_v}, w={next_w}")
            else:
                self.get_logger().info("   Final step reached.")

    def stop_robot(self):
        """로봇 정지 명령 발행"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        time.sleep(0.1) # 발행 보장을 위해 잠시 대기
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.get_logger().info("Shutting down Correcting Dynamics Tester...")
        self.stop_robot()
        super().destroy_node()

# ==============================================================================
# --- Matplotlib 시각화 함수 ---
# (이 부분은 이전과 동일하게 사용하셔도 됩니다)
# ==============================================================================

def update_plot(frame, node: CorrectingDynamicsTester, ax, 
                actual_line, predicted_line, 
                actual_point, predicted_point):
    
    with node.plot_lock:
        actual_traj = np.array(list(node.actual_trajectory))
        predicted_traj = np.array(list(node.predicted_trajectory))

    # 두 궤적 중 하나라도 비어있으면 그리지 않음
    if actual_traj.size == 0 or predicted_traj.size == 0:
        return [actual_line, predicted_line, actual_point, predicted_point]

    # --- (x, y) 좌표계로 플로팅 ---
    actual_line.set_data(actual_traj[:, 0], actual_traj[:, 1])
    predicted_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])
    
    # 마지막 점 표시 (현재 위치 vs 1스텝 예측 위치)
    actual_point.set_data(actual_traj[-1, 0], actual_traj[-1, 1])
    predicted_point.set_data(predicted_traj[-1, 0], predicted_traj[-1, 1])
    
    all_points_plot = np.vstack((actual_traj, predicted_traj))

    # 축 범위 동적 조절
    x_min, y_min = np.min(all_points_plot, axis=0) - 1.0
    x_max, y_max = np.max(all_points_plot, axis=0) + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return [actual_line, predicted_line, actual_point, predicted_point]


def main(args=None):
    rclpy.init(args=args)
    # 클래스 이름을 수정된 'CorrectingDynamicsTester'로 변경
    node = CorrectingDynamicsTester() 

    # rclpy.spin을 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정 (메인 스레드에서 실행)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title('Correcting Dynamics Tester (One-Step Error)', fontsize=14)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    actual_line, = ax.plot([], [], 'b-', lw=2, label='Actual Trajectory (Odom)')
    predicted_line, = ax.plot([], [], 'r--', lw=2, label='Predicted Trajectory (Model, Corrected)')
    actual_point, = ax.plot([], [], 'bo', markersize=8, label='Actual Position')
    predicted_point, = ax.plot([], [], 'ro', markersize=8, label='Predicted (1-Step Ahead)')
    
    ax.legend(loc='upper right', fontsize=10)
    
    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax, actual_line, predicted_line, actual_point, predicted_point),
        interval=100, blit=True
    )

    try:
        plt.show() # Matplotlib 창을 닫을 때까지 메인 스레드 블로킹
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 노드 및 rclpy 정리
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()
        print("Dynamics tester shutdown complete.")


if __name__ == '__main__':
    main()
