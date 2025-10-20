#!/usr/bin/env python3
import pyrealsense2 as rs
import time
import math
import matplotlib.pyplot as plt

# RealSense 장치 초기화 및 IMU 스트림 지원 확인
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No device connected")
    exit(1)

dev = devices[0]
print(f"Device: {dev.get_info(rs.camera_info.name)}")

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

if not (accelerometer_supported and gyroscope_supported):
    print("Required IMU streams are not supported.")
    exit(1)

# IMU 스트림 활성화를 위한 pipeline 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
pipeline.start(config)

# 초기 상태: 2D 평면에서의 위치, 속도, yaw(회전) 값
x, y = 0.0, 0.0
vx, vy = 0.0, 0.0
yaw = 0.0

trajectory_x = [x]
trajectory_y = [y]

# Matplotlib 실시간 플롯 설정 (interactive mode)
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(trajectory_x, trajectory_y, 'b-', label='Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Real-time Trajectory from Realsense IMU')
ax.legend()
n = 10
ax.set_xlim(-n, n)
ax.set_ylim(-n, n)

prev_time = time.time()

try:
    while True:
        # IMU 데이터 읽기
        frames = pipeline.wait_for_frames()
        now = time.time()
        dt = now - prev_time
        dt = dt/100
        prev_time = now

        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if accel_frame and gyro_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # accelerometer 데이터 (단위: m/s^2)
            ax_body = accel_data.x
            ay_body = accel_data.y
            # gyro z축: yaw rate (단위: rad/s)
            yaw_rate = gyro_data.z

            # yaw(회전각) 업데이트
            yaw += yaw_rate * dt

            # 가정: 장치가 항상 수평이므로, body frame의 x,y를 회전시켜 inertial frame로 변환
            ax_inertial = math.cos(yaw) * ax_body - math.sin(yaw) * ay_body
            ay_inertial = math.sin(yaw) * ax_body + math.cos(yaw) * ay_body

            # 가속도를 적분하여 속도 업데이트
            vx += ax_inertial * dt
            vy += ay_inertial * dt

            # 속도를 적분하여 위치 업데이트
            x += vx * dt
            y += vy * dt

            trajectory_x.append(x)
            trajectory_y.append(y)

            # 디버깅용 출력
            print(f"Accel: ({ax_body:.3f}, {ay_body:.3f}), Gyro z: {yaw_rate:.3f}, Pos: ({x:.3f}, {y:.3f})")

            # 실시간 플롯 업데이트
            line.set_xdata(trajectory_x)
            line.set_ydata(trajectory_y)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    pipeline.stop()
    plt.ioff()
    plt.show()

