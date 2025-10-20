#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SquareTrajectoryFollower(Node):
    def __init__(self):
        super().__init__('square_trajectory_follower')
        # 퍼블리셔: /mcu/command/manual_twist 토픽에 Twist 메시지 발행
        self.publisher_ = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)

        # 타이머 생성 (0.1초마다 콜백 -> 초당 10Hz)
        self.timer_period = 0.1
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

        # 현재 상태: 'forward' 또는 'turn'
        self.state = 'forward'
        # 몇 번째 변(side)을 이동 중인지 (0~3까지, 총 4변)
        self.side_count = 0
        # 상태 시작 시점 기록 (시간 계산용)
        self.state_start_time = self.get_clock().now()

        # 이동/회전 설정
        self.linear_speed = 0.2  # 전진 속도 (m/s)
        self.angular_speed = 0.5 # 회전 속도 (rad/s)
        
        # 3미터를 0.2 m/s로 이동 => 소요 시간 15초
        self.forward_time = 3.0 / self.linear_speed  # = 15.0초
        # 90도(약 1.57 rad)를 0.5 rad/s로 회전 => 약 3.14초
        self.turn_time = (math.pi / 2) / self.angular_speed  # = ~3.14초

    def timer_callback(self):
        """정해진 시간만큼 직진 -> 회전 -> 다시 직진, 반복해서 4변 이동."""
        cmd = Twist()
        current_time = self.get_clock().now()
        elapsed = (current_time - self.state_start_time).nanoseconds * 1e-9  # (초 단위)

        # 아직 4변을 다 돌지 않았다면
        if self.side_count < 4:
            # 현재 상태 확인
            if self.state == 'forward':
                # 1) 직진
                cmd.linear.x = self.linear_speed
                # 정해진 시간(15초)만큼 전진 후 상태 전환
                if elapsed >= self.forward_time:
                    self.state = 'turn'
                    self.state_start_time = current_time

            elif self.state == 'turn':
                # 2) 회전
                cmd.angular.z = self.angular_speed
                # 회전 시간(약 3.14초) 경과 후 다음 변
                if elapsed >= self.turn_time:
                    self.side_count += 1
                    self.state = 'forward'
                    self.state_start_time = current_time

        else:
            # 네 변 이동을 모두 마친 경우 -> 정지 후 타이머 종료
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)  # 한 번 더 정지 명령 발행
            self.get_logger().info('정사각형 4변 이동 완료, 정지합니다.')
            self.destroy_timer(self.timer_)
            return

        # 계산된 속도 명령 퍼블리시
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajectoryFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

