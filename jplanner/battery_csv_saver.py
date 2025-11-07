





import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState  # ✅ 배터리 메시지만 임포트
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import os
import time
import csv

# ------------------------------------------------
# 1) Global Path/Time Settings
# ✅ 실행 시간을 기준으로 기본 저장 경로 생성
# ------------------------------------------------

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M", time.localtime())
BASE_DIR = f"../data/{SESSION_TIMESTAMP}"
BASE_DIR = f"./"
os.makedirs(BASE_DIR, exist_ok=True)
# (이미지/깊이 등 다른 폴더 생성 로직 제거)

# ------------------------------------------------
# 2) QoS Settings
# ------------------------------------------------
# Quality of service
qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)


class BatteryDataCollector(Node):
    def __init__(self):
        super().__init__('battery_data_collector')

        # ------------------------------------------------
        # ✅ CSV 파일 설정 (배터리 전용)
        # ------------------------------------------------
        self.csv_path = os.path.join(BASE_DIR, f"battery_data.csv")
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # ✅ 수집할 데이터 컬럼 (배터리만)
        self.csv_header = [
            'timestamp',           # 10Hz 타이머 기준 타임스탬프 (ms)
            'battery_voltage',
            'battery_percentage',
            'battery_current',
        ]

        self.csv_writer.writerow(self.csv_header)
        self.csv_file.flush() # 헤더 즉시 쓰기

        # ------------------------------------------------
        # ✅ 10Hz 주기로 업데이트될 데이터
        # ------------------------------------------------
        self.latest_data = {
            # 'battery': {'voltage': ..., 'percentage': ...}
        }
        # (이미지/깊이 등 다른 latest 변수 제거)

        # ------------------------------------------------
        # 3) ✅ ROS2 구독 등록 (배터리만)
        # ------------------------------------------------
        self.create_subscription(BatteryState, "/mcu/state/battery",
                                 self.battery_callback, qos_profile)

        # (다른 모든 구독 제거)

        # ------------------------------------------------
        # 4) 10Hz 타이머 설정
        # ------------------------------------------------
        self.timer_ = self.create_timer(0.1, self.timer_callback) # 10Hz


    # ✅ 노드 종료 시 CSV 파일 닫기
    def destroy_node(self):
        super().destroy_node()
        if self.csv_file:
            self.csv_file.close()
            print(f"데이터 저장 완료: {self.csv_path}")

    # ------------------------------------------------
    # ✅ 유틸리티: 타임스탬프 (밀리초)
    # ------------------------------------------------
    def get_timestamp(self):
        return str(int(time.time() * 1000))

    # ------------------------------------------------
    # 10Hz 타이머 콜백 (CSV 로깅)
    # ------------------------------------------------
    def timer_callback(self):
        # 1) 공통 타임스탬프 생성 (10Hz)
        timestamp = self.get_timestamp()

        # 2) 가장 최근에 수신된 배터리 데이터를 CSV 한 줄로 로깅
        row_dict = {}

        # (1) battery
        battery = self.latest_data.get('battery', {})
        row_dict['battery_voltage'] = battery.get('voltage', 'nan')
        row_dict['battery_percentage'] = battery.get('percentage', 'nan')
        row_dict['battery_current'] = battery.get('current', 'nan')

        # (다른 모든 센서 데이터 로깅 제거)

        # 3) 실제 CSV 쓰기를 위해 올바른 순서의 리스트로 변환
        row = [timestamp]
        for col in self.csv_header[1:]: # 첫 열(timestamp) 제외
            row.append(row_dict.get(col, 'nan'))

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # (이미지/깊이 등 파일 저장 로직 제거)

    # ------------------------------------------------
    # ✅ 콜백: 메시지 수신 시 "latest_data"만 업데이트
    # ------------------------------------------------

    # (1) Battery State
    def battery_callback(self, msg: BatteryState):
        self.latest_data['battery'] = {
            'voltage': msg.voltage,
            'percentage': msg.percentage,
            'current': msg.current,
        }

    # (다른 모든 콜백 제거)


def main():
    # (argparse 로직 제거)
    rclpy.init()

    node = BatteryDataCollector()

    print(f"배터리 데이터 저장을 시작합니다. (10Hz)")
    print(f"저장 위치: {node.csv_path}")
    print("종료하려면 Ctrl+C를 누르세요.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("종료 요청을 받았습니다...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
