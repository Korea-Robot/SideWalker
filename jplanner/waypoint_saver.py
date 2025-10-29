import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty  # 트리거로 사용할 메시지
import yaml  # 파일 저장을 위해
import atexit # 노드 종료 시 파일 저장을 위해

class WaypointSaver(Node):
    def __init__(self):
        super().__init__('waypoint_saver')
        
        # /odom 토픽을 구독 (현재 위치를 알기 위해)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/krm_auto_localization/odom',
            self.odom_callback,
            10)
        
        # /save_waypoint 토픽을 구독 (저장 신호를 받기 위해)
        self.trigger_sub = self.create_subscription(
            Empty,
            '/save_waypoint',
            self.save_callback,
            10)
        
        self.waypoints = []
        self.last_pose = None
        self.save_count = 0
        self.output_file = 'waypoints.yaml' # 저장될 파일 이름

        self.get_logger().info('Waypoint Saver 노드가 시작되었습니다.')
        self.get_logger().info(f"'/save_waypoint' 토픽을 기다리는 중... (파일은 '{self.output_file}'에 저장됨)")

        # 노드가 종료될 때 save_to_file 함수가 실행되도록 등록
        atexit.register(self.save_to_file)

    def odom_callback(self, msg):
        # odom 메시지에서 pose 정보만 저장
        self.last_pose = msg.pose.pose

    def save_callback(self, msg):
        if self.last_pose is None:
            self.get_logger().warn('아직 Odom 데이터를 받지 못했습니다. 저장을 스킵합니다.')
            return

        # 현재 pose를 리스트에 추가
        self.waypoints.append({
            'position': {
                'x': self.last_pose.position.x,
                'y': self.last_pose.position.y,
                'z': self.last_pose.position.z,
            },
            'orientation': {
                'x': self.last_pose.orientation.x,
                'y': self.last_pose.orientation.y,
                'z': self.last_pose.orientation.z,
                'w': self.last_pose.orientation.w,
            }
        })
        self.save_count += 1
        self.get_logger().info(f'✅ 웨이포인트 {self.save_count} 저장 완료! (x: {self.last_pose.position.x:.2f}, y: {self.last_pose.position.y:.2f})')

    def save_to_file(self):
        if not self.waypoints:
            self.get_logger().info('저장된 웨이포인트가 없습니다. 파일을 생성하지 않습니다.')
            return
            
        try:
            with open(self.output_file, 'w') as f:
                yaml.dump(self.waypoints, f, default_flow_style=False)
            self.get_logger().info(f'--- 💾 총 {self.save_count}개의 웨이포인트를 {self.output_file} 파일로 저장했습니다. ---')
        except Exception as e:
            self.get_logger().error(f'파일 저장 중 오류 발생: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
