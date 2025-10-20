import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import time
import argparse
import sys
import numpy as np # 선형 보간을 위해 numpy 라이브러리를 사용합니다.

class AdvancedReplayer(Node):
    def __init__(self, csv_filepath, replay_freq):
        super().__init__('advanced_twist_replayer')
        self.publisher_ = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.csv_filepath = csv_filepath
        self.replay_frequency = replay_freq # 리플레이 주기 (Hz)
        self.get_logger().info(
            f'CSV 파일로부터 Twist 데이터를 {self.replay_frequency}Hz로 보간하여 리플레이합니다: "{csv_filepath}"'
        )
        self.timer = self.create_timer(1.0, self.start_replay)

    def parse_twist_from_row(self, row):
        """CSV의 한 행에서 Twist 값을 파싱하고, 'nan' 등의 예외를 처리합니다."""
        twist = Twist()
        try:
            twist.linear.x = 0.0 if row.get('manual_linear_x') == 'nan' else float(row.get('manual_linear_x', 0.0))
            twist.linear.y = 0.0 if row.get('manual_linear_y') == 'nan' else float(row.get('manual_linear_y', 0.0))
            twist.angular.z = 0.0 if row.get('manual_angular_z') == 'nan' else float(row.get('manual_angular_z', 0.0))
        except (ValueError, TypeError):
            self.get_logger().warn(f"데이터 파싱 오류. 행: {row}. 0 값으로 처리합니다.")
            return Twist() # 0 값으로 초기화된 Twist 반환
        return twist

    def start_replay(self):
        self.timer.cancel()
        self.get_logger().info('3초 후에 리플레이를 시작합니다...')
        time.sleep(3)

        try:
            with open(self.csv_filepath, mode='r', newline='') as csv_file:
                csv_reader = list(csv.DictReader(csv_file)) # 모든 데이터를 메모리에 로드
            
            if len(csv_reader) < 2:
                self.get_logger().error("리플레이를 위한 데이터가 최소 2줄 이상 필요합니다.")
                rclpy.shutdown()
                return

            # 리플레이 루프 시작
            for i in range(len(csv_reader) - 1):
                prev_row = csv_reader[i]
                current_row = csv_reader[i+1]

                # 1. 이전 지점과 현재 지점의 시간 및 Twist 값 추출
                try:
                    prev_time_ms = int(prev_row['timestamp'])
                    current_time_ms = int(current_row['timestamp'])
                except (ValueError, KeyError):
                    self.get_logger().warn("타임스탬프 오류. 해당 구간을 건너뜁니다.")
                    continue
                
                prev_twist = self.parse_twist_from_row(prev_row)
                current_twist = self.parse_twist_from_row(current_row)

                # 2. 두 지점 사이의 시간 간격과 보간할 스텝 수 계산
                time_diff_s = (current_time_ms - prev_time_ms) / 1000.0
                if time_diff_s <= 0: continue # 시간 역행 또는 동일 시간 데이터는 건너뜀

                num_steps = int(time_diff_s * self.replay_frequency)
                if num_steps == 0: num_steps = 1
                
                # 3. 선형 보간(Linear Interpolation)을 사용하여 중간 Twist 값들 생성
                interp_lin_x = np.linspace(prev_twist.linear.x, current_twist.linear.x, num_steps)
                interp_lin_y = np.linspace(prev_twist.linear.y, current_twist.linear.y, num_steps)
                interp_ang_z = np.linspace(prev_twist.angular.z, current_twist.angular.z, num_steps)

                # 4. 보간된 값들을 높은 주기로 발행
                interp_delay_s = time_diff_s / num_steps
                for j in range(num_steps):
                    interp_twist = Twist()
                    interp_twist.linear.x = interp_lin_x[j]
                    interp_twist.linear.y = interp_lin_y[j]
                    interp_twist.angular.z = interp_ang_z[j]
                    
                    self.publisher_.publish(interp_twist)
                    self.get_logger().info(f"보간 발행: [LX: {interp_twist.linear.x:.3f}, AZ: {interp_twist.angular.z:.3f}]", throttle_duration_sec=1.0)
                    time.sleep(interp_delay_s)

        except FileNotFoundError:
            self.get_logger().error(f'에러: CSV 파일을 찾을 수 없습니다 - "{self.csv_filepath}"')
        except Exception as e:
            self.get_logger().error(f'리플레이 중 예외 발생: {e}')

        self.get_logger().info('CSV 리플레이가 완료되었습니다.')
        # 마지막으로 정지 명령 발행
        self.publisher_.publish(Twist())
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Interpolate and replay Twist commands from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the input data.csv file.')
    parser.add_argument('--freq', type=int, default=200, help='Replay frequency in Hz (default: 50)')
    parsed_args = parser.parse_args(sys.argv[1:])

    node = AdvancedReplayer(csv_filepath=parsed_args.csv_file, replay_freq=parsed_args.freq)
    
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
