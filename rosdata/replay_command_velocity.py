import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import time
import argparse
import sys

class CSWTwistReplayer(Node):
    def __init__(self, csv_filepath):
        """
        노드 초기화, 퍼블리셔 생성 및 리플레이 준비
        """
        super().__init__('csv_twist_replayer')
        # Twist 메시지를 발행할 퍼블리셔 생성
        # 원본 스크립트와 동일한 토픽 이름을 사용합니다. 필요시 변경 가능합니다.
        self.publisher_ = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.csv_filepath = csv_filepath
        self.get_logger().info(f'CSV 파일로부터 Twist 데이터를 리플레이합니다: "{csv_filepath}"')
        
        # 리플레이를 별도의 타이머에서 시작하여 ROS 2 노드 초기화가 완료될 시간을 줍니다.
        self.timer = self.create_timer(1.0, self.start_replay)

    def start_replay(self):
        """
        CSV 파일을 읽고 타임스탬프 간격에 맞춰 데이터를 발행하는 메인 로직
        """
        # 타이머는 한 번만 실행하고 즉시 취소하여 반복되지 않도록 합니다.
        self.timer.cancel()

        self.get_logger().info('3초 후에 리플레이를 시작합니다...')
        time.sleep(3) # 시뮬레이션 환경이 준비될 시간을 줍니다.

        try:
            with open(self.csv_filepath, mode='r', newline='') as csv_file:
                # csv.DictReader를 사용하면 헤더 이름으로 열에 접근할 수 있어 편리합니다.
                csv_reader = csv.DictReader(csv_file)
                
                previous_timestamp_ms = None

                for row in csv_reader:
                    # 1. 타임스탬프 처리 및 시간 간격 계산
                    try:
                        current_timestamp_ms = int(row['timestamp'])
                    except (ValueError, KeyError):
                        self.get_logger().warn("타임스탬프 열을 읽을 수 없습니다. 다음 행으로 넘어갑니다.")
                        continue

                    if previous_timestamp_ms is not None:
                        # 밀리초(ms) 단위의 시간 차이를 초(s) 단위로 변환
                        delay_sec = (current_timestamp_ms - previous_timestamp_ms) / 1000.0
                        if delay_sec > 0:
                            time.sleep(delay_sec)
                    
                    previous_timestamp_ms = current_timestamp_ms

                    # 2. Twist 메시지 생성 및 값 할당
                    twist_msg = Twist()

                    # 'nan' 또는 잘못된 값에 대한 예외 처리
                    try:
                        linear_x = 0.0 if row['manual_linear_x'] == 'nan' else float(row['manual_linear_x'])
                        linear_y = 0.0 if row['manual_linear_y'] == 'nan' else float(row['manual_linear_y'])
                        angular_z = 0.0 if row['manual_angular_z'] == 'nan' else float(row['manual_angular_z'])
                        
                        twist_msg.linear.x = linear_x
                        twist_msg.linear.y = linear_y
                        twist_msg.angular.z = angular_z
                    except (ValueError, KeyError) as e:
                        self.get_logger().warn(f"Twist 데이터 파싱 중 오류 발생 ({e}). 0 값으로 발행합니다.")
                        twist_msg.linear.x = 0.0
                        twist_msg.linear.y = 0.0
                        twist_msg.angular.z = 0.0

                    # 3. 메시지 발행
                    self.publisher_.publish(twist_msg)
                    self.get_logger().info(f"발행: [Linear X: {twist_msg.linear.x:.2f}, Angular Z: {twist_msg.angular.z:.2f}]")

        except FileNotFoundError:
            self.get_logger().error(f'에러: CSV 파일을 찾을 수 없습니다 - "{self.csv_filepath}"')
        except Exception as e:
            self.get_logger().error(f'리플레이 중 예외 발생: {e}')

        self.get_logger().info('CSV 리플레이가 완료되었습니다.')
        # 리플레이가 끝나면 노드를 종료합니다.
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # argparse를 사용하여 커맨드 라인에서 CSV 파일 경로를 받습니다.
    parser = argparse.ArgumentParser(description='Replay Twist commands from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the input data.csv file.')
    
    # rclpy.init() 이후에 ROS 관련 인자를 제외하고 파싱합니다.
    parsed_args = parser.parse_args(sys.argv[1:])

    node = CSWTwistReplayer(csv_filepath=parsed_args.csv_file)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
