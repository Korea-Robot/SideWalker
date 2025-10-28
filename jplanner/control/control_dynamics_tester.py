#!/usr/bin/env python3
"""
auto dynamics model tuner

"""

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

class DynamicsTester(Node):
    def __init__(self):
        super().__init__('dynamics_tester')

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
        self.predicted_pose = None   # [x, y, yaw] (Model 기준)
        self.initial_pose_set = False

        # --- 시각화 데이터 ---
        self.plot_lock = threading.Lock()
        self.actual_trajectory = []
        self.predicted_trajectory = []

        # 메인 제어/시뮬레이션 타이머
        self.control_timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"✅ Dynamics Tester (Open-Loop) Hias started.")
        self.get_logger().info(f"   Test Sequence: {len(self.control_sequence)} steps.")
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
        current_pose = [x, y, yaw]

        with self.plot_lock:
            self.actual_pose = current_pose
            self.actual_trajectory.append([x, y])

            # 시뮬레이션 시작 위치 설정 (가장 처음 Odom 메시지 기준)
            if not self.initial_pose_set:
                self.predicted_pose = current_pose
                self.predicted_trajectory.append([x, y])
                self.initial_pose_set = True
                self.get_logger().info(f"   Initial pose set: [{x:.2f}, {y:.2f}, {yaw:.2f}]")

    def timer_callback(self):
        """제어 명령 발행 및 동역학 모델 시뮬레이션을 위한 메인 루프"""
        
        # 아직 Odom 수신 전이면 대기
        if not self.initial_pose_set:
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

        # 3. 동역학 모델로 다음 상태 예측
        with self.plot_lock:
            # self.predicted_pose가 None이 아님을 보장 (initial_pose_set 플래그 덕분)
            self.predicted_pose = self.unicycle_model(self.predicted_pose, v, w, self.dt)
            self.predicted_trajectory.append(self.predicted_pose[:2])

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
        self.get_logger().info("Shutting down Dynamics Tester...")
        self.stop_robot()
        super().destroy_node()

# ==============================================================================
# --- Matplotlib 시각화 함수 ---
# ==============================================================================

def update_plot(frame, node: DynamicsTester, ax, 
                actual_line, predicted_line, 
                actual_point, predicted_point):
    
    with node.plot_lock:
        actual_traj = np.array(list(node.actual_trajectory))
        predicted_traj = np.array(list(node.predicted_trajectory))

    if actual_traj.size == 0 or predicted_traj.size == 0:
        return [actual_line, predicted_line, actual_point, predicted_point]

    # --- (x, y) 좌표계로 플로팅 ---
    # AStarPlanner와 동일하게 (-y, x)로 플로팅하려면 주석을 교체하세요.
    
    # 1. 표준 (x, y) 플로팅
    actual_line.set_data(actual_traj[:, 0], actual_traj[:, 1])
    predicted_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])
    actual_point.set_data(actual_traj[-1, 0], actual_traj[-1, 1])
    predicted_point.set_data(predicted_traj[-1, 0], predicted_traj[-1, 1])
    all_points_plot = np.vstack((actual_traj, predicted_traj))
    
    # 2. A* 플래너 호환 (-y, x) 플로팅
    # actual_line.set_data(-actual_traj[:, 1], actual_traj[:, 0])
    # predicted_line.set_data(-predicted_traj[:, 1], predicted_traj[:, 0])
    # actual_point.set_data(-actual_traj[-1, 1], actual_traj[-1, 0])
    # predicted_point.set_data(-predicted_traj[-1, 1], predicted_traj[-1, 0])
    # all_points_plot = np.vstack((
    #     np.c_[-actual_traj[:, 1], actual_traj[:, 0]], 
    #     np.c_[-predicted_traj[:, 1], predicted_traj[:, 0]]
    # ))

    # 축 범위 동적 조절
    x_min, y_min = np.min(all_points_plot, axis=0) - 1.0
    x_max, y_max = np.max(all_points_plot, axis=0) + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return [actual_line, predicted_line, actual_point, predicted_point]


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsTester()

    # rclpy.spin을 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Matplotlib 설정 (메인 스레드에서 실행)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title('Dynamics Model Validation (Open-Loop Test)', fontsize=14)
    
    # --- 플로팅 좌표계에 맞춰 라벨 설정 ---
    ax.set_xlabel('X Position (m)') # 표준 (x, y)
    ax.set_ylabel('Y Position (m)') # 표준 (x, y)
    # ax.set_xlabel('-Y Position (m)') # A* 호환 (-y, x)
    # ax.set_ylabel('X Position (m)')  # A* 호환 (-y, x)
    
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    actual_line, = ax.plot([], [], 'b-', lw=2, label='Actual Trajectory (Odom)')
    predicted_line, = ax.plot([], [], 'r--', lw=2, label='Predicted Trajectory (Model)')
    actual_point, = ax.plot([], [], 'bo', markersize=8, label='Actual Position')
    predicted_point, = ax.plot([], [], 'ro', markersize=8, label='Predicted Position')
    
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
