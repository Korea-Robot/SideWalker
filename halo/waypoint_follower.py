# 파일명: waypoint_follower.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import math
import cv2

# tf_transformations 라이브러리는 ROS2에서 기본 제공되지 않을 수 있습니다.
# 설치: pip install tf-transformations
from tf_transformations import euler_from_quaternion

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # ROS2 퍼블리셔, 서브스크라이버, 타이머 설정
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(Odometry, '/command_odom', self.odom_callback, 10)
        
        # 중요: Depth 카메라 토픽을 사용해야 합니다. 
        # RGB 토픽('/camera/camera/rgb/image_rect_raw') 대신 실제 Depth 토픽으로 변경해주세요.
        # 예: '/camera/depth/image_rect_raw'
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        self.control_timer = self.create_timer(0.1, self.control_loop) # 10Hz 제어 루프

        # CV Bridge
        self.bridge = CvBridge()

        # 로봇 상태 및 웨이포인트 변수
        self.current_pose = None  # (x, y, theta)
        self.depth_image = None
        
        # 주어진 웨이포인트 리스트
        self.waypoints = [
            (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0),
            (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0),
            (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)
        ]
        self.waypoint_index = 0
        self.goal_threshold = 0.5  # 목표 도착 판단 거리 (미터)

        # 제어 파라미터
        self.MAX_LINEAR_VEL = 0.5  # 최대 직진 속도
        self.MAX_ANGULAR_VEL = 1.0   # 최대 회전 속도
        self.ANGULAR_GAIN = 2.0      # 회전 속도 제어를 위한 게인 값
        self.OBSTACLE_DISTANCE_THRESHOLD = 1.0  # 장애물 감지 거리 (미터)
        self.ANGULAR_THRESHOLD_FOR_ZERO_LINEAR = 0.5 # 이 각속도를 넘으면 직진 속도를 0으로

        self.get_logger().info("Waypoint Follower 노드가 시작되었습니다.")
        self.get_logger().info(f"총 {len(self.waypoints)}개의 웨이포인트를 추종합니다.")

    def odom_callback(self, msg):
        """Odometry 메시지를 수신하여 현재 위치(x, y)와 방향(theta)을 업데이트합니다."""
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        
        # Quaternion을 Euler 각도로 변환하여 yaw(theta) 값을 얻습니다.
        _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        self.current_pose = (position.x, position.y, yaw)

    def depth_callback(self, msg):
        """Depth Image 메시지를 수신하여 OpenCV 이미지로 변환하고 저장합니다."""
        try:
            # Depth 데이터는 보통 16비트 단일 채널(16UC1) 또는 32비트 부동소수점(32FC1) 입니다.
            # 'passthrough'는 인코딩을 그대로 유지합니다.
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth 이미지 변환 실패: {e}")

    def control_loop(self):
        """메인 제어 로직을 수행하는 함수."""
        # Odometry나 Depth 정보가 아직 수신되지 않았다면 아무것도 하지 않음
        if self.current_pose is None or self.depth_image is None:
            self.get_logger().warn("Odometry 또는 Depth 정보 수신 대기 중...")
            return

        # 모든 웨이포인트를 완료했는지 확인
        if self.waypoint_index >= len(self.waypoints):
            self.get_logger().info("모든 웨이포인트에 도달했습니다. 정지합니다.")
            self.publish_twist(0.0, 0.0)
            self.control_timer.cancel() # 타이머 중지
            return

        # 현재 목표 웨이포인트 설정
        goal_x, goal_y = self.waypoints[self.waypoint_index]
        current_x, current_y, current_theta = self.current_pose
        
        # 목표까지의 거리 계산
        distance_to_goal = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)

        # 목표에 도달했는지 확인
        if distance_to_goal < self.goal_threshold:
            self.get_logger().info(f"웨이포인트 {self.waypoint_index + 1} 도착! 다음으로 이동합니다.")
            self.waypoint_index += 1
            return

        # --- 장애물 회피 로직 ---
        h, w = self.depth_image.shape
        # 전방 중앙 영역(폭의 1/3)을 관심 영역(ROI)으로 설정
        center_roi = self.depth_image[:, w//3 : 2*w//3]
        
        # ROI 내의 유효한(0이 아닌) 거리 값들만 필터링
        valid_depths = center_roi[center_roi > 0]
        
        min_center_depth = np.min(valid_depths) if valid_depths.size > 0 else float('inf')

        if min_center_depth < self.OBSTACLE_DISTANCE_THRESHOLD:
            self.get_logger().warn(f"장애물 감지! 거리: {min_center_depth:.2f}m. 회피 기동 시작.")
            # 왼쪽과 오른쪽 영역의 평균 깊이를 계산하여 더 넓은 공간으로 회전
            left_roi = self.depth_image[:, :w//3]
            right_roi = self.depth_image[:, 2*w//3:]
            
            avg_left_depth = np.mean(left_roi[left_roi > 0]) if np.any(left_roi > 0) else 0
            avg_right_depth = np.mean(right_roi[right_roi > 0]) if np.any(right_roi > 0) else 0

            # 더 넓은 공간 쪽으로 회전 (오른쪽이 넓으면 시계방향, 왼쪽이 넓으면 반시계방향)
            angular_vel = -self.MAX_ANGULAR_VEL if avg_right_depth > avg_left_depth else self.MAX_ANGULAR_VEL
            self.publish_twist(0.0, angular_vel)
            return

        # --- 목표 추종 로직 (장애물이 없을 때) ---
        # 1. 목표 지점을 로봇의 지역 좌표계(local frame)로 변환
        # dx, dy는 월드 좌표계 기준 (Goal - Current)
        dx = goal_x - current_x
        dy = goal_y - current_y
        
        # 로봇의 현재 방향(-theta)만큼 회전 행렬을 곱하여 지역 좌표계로 변환
        # [x'] = [cos(θ)  sin(θ)] [x]
        # [y'] = [-sin(θ) cos(θ)] [y]
        local_goal_x = dx * math.cos(current_theta) + dy * math.sin(current_theta)
        local_goal_y = -dx * math.sin(current_theta) + dy * math.cos(current_theta)

        # 2. 지역 좌표계 기준 목표 방향 계산
        angle_to_goal = math.atan2(local_goal_y, local_goal_x)

        # 3. 제어 입력(속도) 계산
        linear_vel = self.MAX_LINEAR_VEL
        angular_vel = self.ANGULAR_GAIN * angle_to_goal

        # 4. 요청된 속도 규칙 적용
        # 규칙 1: 각속도가 0.5를 초과하면 직진 속도는 0
        if abs(angular_vel) > self.ANGULAR_THRESHOLD_FOR_ZERO_LINEAR:
            linear_vel = 0.0
            
        # 규칙 2: 속도 제한 (clamping)
        angular_vel = np.clip(angular_vel, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        
        # 규칙 3: 각속도 Deadzone (선택적 구현)
        # 만약 0.2 이하의 작은 각속도를 무시하고 싶다면 아래 주석을 해제
        # if -0.2 < angular_vel < 0.2:
        #     angular_vel = 0.0

        self.publish_twist(linear_vel, angular_vel)

    def publish_twist(self, linear_x, angular_z):
        """Twist 메시지를 생성하고 퍼블리시합니다."""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("키보드 인터럽트로 종료합니다.")
    finally:
        # 종료 시 로봇을 정지시킴
        node.publish_twist(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
