#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
import matplotlib.pyplot as plt

class SquareTwistPublisher(Node):
    def __init__(self):
        super().__init__('square_twist_publisher')
        # 퍼블리셔 생성: /mcu/command/manual_twist 토픽, 큐 사이즈=10
        self.publisher_ = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        
        # 타이머 생성 (0.1초마다 콜백 실행)
        timer_period = 0.1  # 100ms
        self.timer_ = self.create_timer(timer_period, self.timer_callback)
        
        # 상태 변수: 0~7까지 진행 후 종료
        # 상태 0,2,4,6: 전진 (각 3미터) / 상태 1,3,5,7: 좌측 90도 회전
        self.state = 0
        self.state_start_time = self.get_clock().now()

        # 로봇의 가상 위치 및 방향 초기화
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # 라디안 단위

        # 경로 저장 리스트 (실시간 시각화용)
        self.x_data = [self.x]
        self.y_data = [self.y]

        # Matplotlib 설정 (interactive mode 활성화)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.x_data, self.y_data, 'b-', label='Trajectory')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Robot Trajectory (Square)')
        self.ax.legend()
        # 예상 이동 범위에 따른 축 범위 설정
        self.ax.set_xlim(-1, 4)
        self.ax.set_ylim(-1, 4)
        
        # yaw scaling factor (회전 보정 값)
        self.yaw_scale = 1.4

    def timer_callback(self):
        msg = Twist()
        current_time = self.get_clock().now()
        dt = 0.1  # 타이머 주기 (초)

        # 전진 및 회전에 필요한 시간 계산
        forward_duration = 15.0  # 3미터 전진: 3 / 0.2 = 15초
        # 회전 시, 실제 angular 명령은 0.5 * yaw_scale로 주므로 지속 시간도 조정
        turn_duration = (math.pi / 2) / (0.5 * self.yaw_scale)
        
        # 상태 머신에 따른 제어
        if self.state in [0, 2, 4, 6]:
            # 전진 상태: 0.2 m/s, 회전 없음
            msg.linear.x = 0.2
            msg.angular.z = 0.0
            if (current_time - self.state_start_time).nanoseconds * 1e-9 >= forward_duration:
                self.state += 1
                self.state_start_time = current_time

        elif self.state in [1, 3, 5, 7]:
            # 회전 상태: 제자리 회전, yaw scaling factor 적용
            msg.linear.x = 0.0
            msg.angular.z = 0.5 * self.yaw_scale
            if (current_time - self.state_start_time).nanoseconds * 1e-9 >= turn_duration:
                self.state += 1
                self.state_start_time = current_time

        else:
            # 모든 상태 완료 후 정지
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher_.publish(msg)
            self.destroy_timer(self.timer_)
            self.get_logger().info('Square trajectory complete. Robot stopped.')
            plt.ioff()
            plt.show()
            return

        # Twist 메시지 발행
        self.publisher_.publish(msg)

        # 간단한 운동 모델 (Euler integration): 현재 속도 명령으로 위치 업데이트
        v = msg.linear.x
        w = msg.angular.z
        self.theta += w * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        # 경로 데이터 업데이트 및 플롯 갱신
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = SquareTwistPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
