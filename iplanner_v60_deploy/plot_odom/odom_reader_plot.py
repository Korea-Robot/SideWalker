import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np

class OdomRecorderAndPlotter(Node):
    """
    /command_odom 토픽을 60초 동안 구독하여 궤적을 기록하고,
    종료 후 Matplotlib를 사용해 궤적을 시각화하는 노드.
    """
    def __init__(self):
        super().__init__('odom_recorder_and_plotter')
        
        # 60초 녹화 시간을 설정합니다.
        self.duration_sec = 60.0
        
        # 궤적 데이터를 저장할 리스트를 초기화합니다.
        self.trajectory_data = []
        
        # Odometry 토픽을 구독합니다.
        self.subscription = self.create_subscription(
            Odometry,
            '/command_odom',
            self.odom_callback,
            10)
        
        # 60초 후에 한 번만 실행되는 원샷(one-shot) 타이머를 생성합니다.
        self.timer = self.create_timer(self.duration_sec, self.stop_recording_and_plot)
        
        self.get_logger().info(f'Recording started. Will run for {self.duration_sec} seconds.')

    def odom_callback(self, msg):
        """
        Odometry 메시지를 받을 때마다 호출되어 데이터를 리스트에 추가합니다.
        """
        # x, y 좌표를 추출하여 리스트에 추가합니다.
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        self.trajectory_data.append([pos_x, pos_y])

    def stop_recording_and_plot(self):
        """
        타이머에 의해 호출되며, 기록을 종료하고 궤적을 시각화합니다.
        """
        self.get_logger().info('Recording finished. Plotting trajectory...')
        
        # 더 이상 메시지를 받지 않도록 구독을 파괴합니다.
        self.destroy_subscription(self.subscription)
        
        # 데이터가 비어있지 않은지 확인합니다.
        if not self.trajectory_data:
            self.get_logger().warn("No data recorded. The robot may not have moved or the topic was empty.")
            self.destroy_node()
            return
            
        # 리스트 데이터를 NumPy 배열로 변환합니다.
        data = np.array(self.trajectory_data)
        
        # 궤적을 그립니다.
        self.plot_trajectory(data)
        
        # 노드를 종료하여 spin() 루프를 빠져나옵니다.
        self.destroy_node()

    def plot_trajectory(self, data):
        """
        수집된 궤적 데이터를 Matplotlib로 그리는 함수.
        """
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label='Trajectory', linewidth=2)
        
        # 시작 지점(0,0)을 빨간색 점으로 표시합니다.
        plt.plot(0, 0, 'ro', label='Start Point')
        
        plt.title('Robot Trajectory from /command_odom')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        
        # X, Y 축 비율을 동일하게 설정하여 궤적의 형태가 왜곡되지 않도록 합니다.
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.legend()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = OdomRecorderAndPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
