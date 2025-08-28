import pyrealsense2 as rs

# RealSense 장치 초기화
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No device connected")
    exit(1)

dev = devices[0]
print(f"Device: {dev.get_info(rs.camera_info.name)}")

# IMU 데이터 스트림 확인 및 출력
accelerometer_supported = False
gyroscope_supported = False

for sensor in dev.query_sensors():
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() == rs.stream.accel and not accelerometer_supported:
            print("Accelerometer stream is supported.")
            accelerometer_supported = True
        if profile.stream_type() == rs.stream.gyro and not gyroscope_supported:
            print("Gyroscope stream is supported.")
            gyroscope_supported = True

# IMU 데이터 읽기 및 출력
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            print(f"Accelerometer: x={accel_data.x}, y={accel_data.y}, z={accel_data.z}")
        
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Gyroscope: x={gyro_data.x}, y={gyro_data.y}, z={gyro_data.z}")

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()

