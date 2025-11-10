import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

# ==================================================
# [수정됨] 사용자 설정 영역
# ==================================================
BAG_PATH = 'battery_log'                # bag 파일이 저장된 디렉토리 이름
TOPIC_NAME = '/mcu/state/battery'       # 실제 기록한 토픽 이름

# [중요] 메시지 타입 확인 필요! 
# 표준 배터리 메시지라고 가정합니다. 만약 커스텀 메시지라면 이 부분을 수정해야 합니다.
try:
    from sensor_msgs.msg import BatteryState as MsgType
except ImportError:
    print("sensor_msgs를 찾을 수 없습니다. 메시지 타입이 다를 수 있습니다.")
    # 만약 커스텀 메시지라면 아래처럼 바꿉니다:
    # from my_custom_msgs.msg import MyBattery as MsgType
# ==================================================

def read_and_plot_bag():
    storage_options = StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Bag 파일을 열 수 없습니다: {e}")
        return

    storage_filter = StorageFilter(topics=[TOPIC_NAME])
    reader.set_filter(storage_filter)

    timestamps = []
    battery_values = []
    
    print(f"Reading '{TOPIC_NAME}' from '{BAG_PATH}'...")

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        
        if topic == TOPIC_NAME:
            try:
                msg = deserialize_message(data, MsgType)
                timestamps.append(t_ns)
                
                # [수정됨] 메시지 내의 percentage 필드 접근
                # BatteryState 메시지의 경우 percentage는 0.0 ~ 1.0 사이 값이 표준일 수 있으므로
                # 필요하다면 * 100을 해줍니다. (데이터 확인 후 조정하세요)
                battery_values.append(msg.percentage) 
                
            except Exception as e:
                print(f"데이터 역직렬화 실패. 메시지 타입(MsgType)이 맞는지 확인하세요.\n에러: {e}")
                return

    if not timestamps:
        print(f"Error: '{TOPIC_NAME}' 토픽 데이터를 찾을 수 없습니다.")
        return

    # 시간축 변환 (나노초 -> 초)
    start_time = timestamps[0]
    time_sec = [(t - start_time) / 1e9 for t in timestamps]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(time_sec, battery_values, label='Battery Percentage', color='blue', linewidth=2)
    
    plt.title('Battery Usage Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Battery Percentage', fontsize=12) # 단위가 %인지 0~1인지 확인 필요
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    print("Plotting data...")
    plot_path = 'battery_plot.png'
    plt.savefig(plot_path,dpi=150)
    plt.show()

if __name__ == '__main__':
    read_and_plot_bag()
