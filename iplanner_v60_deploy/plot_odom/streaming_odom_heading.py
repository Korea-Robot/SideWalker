import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# ROS2 노드 클래스: 오도메트리 데이터를 수신합니다.
class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_realtime_plotter')
        self.subscription = self.create_subscription(
            Odometry,
            '/command_odom',
            self.odom_callback,
            10)
        self.get_logger().info('Odom data subscriber started.')
        self.trajectory_data = [] # 궤적 데이터를 저장할 리스트
        self.current_pos = [0.0, 0.0] # 현재 위치를 저장할 변수
        self.current_heading = 0.0 # 현재 헤딩(yaw) 각도를 저장할 변수 (라디안)

    def quaternion_to_yaw(self, q):
        """
        쿼터니언을 Yaw(라디안)로 변환하는 헬퍼 함수.
        """
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        
        # Yaw(Z-axis 회전) 계산
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_rad = np.arctan2(t3, t4)
        return yaw_rad

    def odom_callback(self, msg):
        """
        Odometry 메시지를 받을 때마다 호출되어 데이터를 업데이트합니다.
        """
        # x, y 좌표 추출 및 저장
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        self.trajectory_data.append([pos_x, pos_y])
        self.current_pos = [pos_x, pos_y]
        
        # 쿼터니언에서 yaw(헤딩) 각도 계산 및 저장
        orientation_q = msg.pose.pose.orientation
        self.current_heading = self.quaternion_to_yaw(orientation_q)


# Matplotlib 애니메이션을 위한 업데이트 함수
def update_plot(i, node, traj_line, start_point, current_point, heading_line, ax):
    """
    FuncAnimation에 의해 주기적으로 호출되며, 플롯을 업데이트합니다.
    """
    # ROS2 메시지 큐를 한 번 처리하여 최신 데이터를 가져옵니다.
    rclpy.spin_once(node, timeout_sec=0)

    # 데이터가 비어있지 않은지 확인합니다.
    if not node.trajectory_data:
        return

    # 리스트 데이터를 NumPy 배열로 변환합니다.
    data = np.array(node.trajectory_data)

    # 궤적 라인 데이터를 업데이트합니다.
    traj_line.set_data(data[:, 1], data[:, 0])

    # 시작점과 현재점 데이터를 업데이트합니다.
    start_point.set_data(data[0, 1], data[0, 0])
    current_point.set_data(data[-1, 1], data[-1, 0])
    
    # 현재 위치와 헤딩 각도에 따라 헤딩 라인의 끝점을 계산합니다.
    current_x, current_y = node.current_pos
    heading_length = 0.3  # 헤딩 화살표의 길이 (m)
    end_x = current_x + heading_length * np.cos(node.current_heading)
    end_y = current_y + heading_length * np.sin(node.current_heading)

    # 헤딩 라인 데이터를 업데이트합니다.
    heading_line.set_data([current_y, end_y],[current_x, end_x])

    # 플롯의 X, Y 축 범위를 동적으로 조정합니다.
    ax.relim()
    ax.autoscale_view()
    
    # 레전드 업데이트 (선택 사항)
    current_label = f'Current Pos ({current_y:.2f}, {current_x:.2f})'
    current_point.set_label(current_label)
    heading_label = f'Heading ({np.degrees(node.current_heading):.2f}°)'
    heading_line.set_label(heading_label)
    ax.legend()
    
    return traj_line, start_point, current_point, heading_line

def main(args=None):
    # ROS2 초기화
    rclpy.init(args=args)
    # 노드 생성
    odom_node = OdomSubscriber()

    # Matplotlib 초기화
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 궤적, 시작점, 현재 위치, 헤딩을 위한 초기 플롯 객체 생성
    traj_line, = ax.plot([], [], 'b-', linewidth=2, label='Trajectory')
    start_point, = ax.plot([], [], 'ro', markersize=8, label='Start Point')
    current_point, = ax.plot([], [], 'go', markersize=8, label='Current Pos')
    heading_line, = ax.plot([], [], 'r--', linewidth=2, label='Heading')

    # 플롯 설정
    ax.set_title('Real-time Robot Trajectory with Heading')
    ax.set_xlabel('Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.set_aspect('equal', adjustable='box') # X, Y 축 비율을 동일하게 설정
    ax.grid(True)
    ax.legend()

    # FuncAnimation을 사용하여 플롯을 실시간으로 업데이트
    ani = FuncAnimation(fig, update_plot, fargs=(odom_node, traj_line, start_point, current_point, heading_line, ax),
                        interval=50, blit=False, cache_frame_data=False)

    try:
        # GUI 이벤트 루프 시작
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 및 ROS2 리소스 정리
        odom_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
