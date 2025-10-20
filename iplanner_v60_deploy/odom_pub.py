import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
import math
from tf2_ros import TransformBroadcaster

class CmdVelToOdom(Node):
    """
    cmd_vel 토픽을 구독하여 Odometry를 계산하고 /command_odom 토픽으로 발행하는 노드.
    로봇의 움직임은 2D 평면(X, Y, Yaw)으로 제한된다고 가정합니다.
    """
    def __init__(self):
        super().__init__('cmd_vel_to_odom')
        
        self.linear_scale = 1.4
        self.angular_scale = 1.35 # 1.35 # 0.00135


        self.subscription = self.create_subscription(
            Twist,
            '/mcu/command/manual_twist',
            self.cmd_vel_callback,
            10)
        
        self.odom_publisher = self.create_publisher(Odometry, '/command_odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = self.get_clock().now()
        
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.timer = self.create_timer(0.05, self.publish_odom)

    def cmd_vel_callback(self, msg):
        """cmd_vel 메시지를 받으면 현재 속도 변수를 업데이트합니다."""
        self.vx = msg.linear.x * self.linear_scale
        self.vy = msg.linear.y * self.linear_scale
        self.wz = msg.angular.z * self.angular_scale

    def publish_odom(self):
        """타이머에 맞춰 Odometry를 계산하고 발행 및 TF를 브로드캐스팅합니다."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9


        # 1. 위치(x, y) 업데이트를 먼저 수행합니다.
        #    이때 사용하는 self.theta는 이전 스텝의 방향 값입니다.
        # 로봇의 지역 좌표계 속도(vx, vy)를 전역(odom) 좌표계 속도로 변환하여 위치 변화량 계산
        delta_x = (self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)) * dt
        delta_y = (self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)) * dt
        
        self.x += delta_x
        self.y += delta_y

        # 2. 그 다음, 현재 스텝의 방향(theta)을 업데이트합니다.
        delta_theta = self.wz * dt 
        self.theta += delta_theta
        # --- [수정] Odometry 계산 순서 변경 ---
        
        # 각도를 -pi ~ +pi 범위로 정규화
        if self.theta > math.pi:
            self.theta -= 2 * math.pi
        elif self.theta < -math.pi:
            self.theta += 2 * math.pi

        # --- 메시지 생성 및 발행 (이 부분은 동일) ---
        # 1. Odometry 메시지 생성
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        
        q = self.euler_to_quaternion(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = self.vx
        odom_msg.twist.twist.linear.y = self.vy
        odom_msg.twist.twist.angular.z = self.wz

        odom_msg.pose.covariance[0] = -1.0
        odom_msg.twist.covariance[0]= -1.0
        
        self.odom_publisher.publish(odom_msg)

        # 2. TF(TransformStamped) 메시지 생성 및 브로드캐스팅
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        self.last_time = current_time

    def euler_to_quaternion(self, roll, pitch, yaw):
        """오일러 각(roll, pitch, yaw)을 쿼터니언으로 변환합니다."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
