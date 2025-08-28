#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ManualTwistPublisher(Node):
    def __init__(self):
        super().__init__('manual_twist_publisher')
        # 퍼블리셔 생성: /mcu/command/manual_twist 토픽, 큐 사이즈=10
        self.publisher_ = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)

        # 타이머 생성 (0.1초마다 콜백 실행)
        timer_period = 0.1  # 100ms
        self.timer_ = self.create_timer(timer_period, self.timer_callback)

        # 동작을 단계별로 제어하기 위해 상태를 저장
        self.state = 0
        self.state_start_time = self.get_clock().now()

    def timer_callback(self):
        """주기적으로 Twist 메시지를 발행하여 로봇을 움직임."""
        msg = Twist()
        current_time = self.get_clock().now()

        # 현재 state(단계)에 따라 다른 속도 명령을 보냄
        if self.state == 0:
            # 0단계: 직진 (예: 3초 동안)
            msg.linear.x = 0.2   # 전진 속도
            msg.angular.z = 0.0  # 회전 없음

            # 3초 후 다음 단계로
            if (current_time - self.state_start_time).nanoseconds * 1e-9 >= 3.0:
                self.state = 1
                self.state_start_time = current_time

        elif self.state == 1:
            # 1단계: 회전 (예: 2초 동안)
            msg.linear.x = 0.0
            msg.angular.z = 0.5  # 제자리 회전

            # 2초 후 다음 단계로
            if (current_time - self.state_start_time).nanoseconds * 1e-9 >= 2.0:
                self.state = 2
                self.state_start_time = current_time

        elif self.state == 2:
            # 2단계: 정지
            msg.linear.x = 0.0
            msg.angular.z = 0.0

            # 여기서는 멈춘 뒤 더 이상 할 일이 없으니 Timer를 멈춰버림
            # (원한다면 다른 동작을 이어서 할 수도 있음)
            self.destroy_timer(self.timer_)
            self.get_logger().info('모든 동작 완료. 정지 상태 유지.')
        
        # 메시지 퍼블리시
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = ManualTwistPublisher()
    try:
        rclpy.spin(node)  # 노드가 종료될 때까지 스핀
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

