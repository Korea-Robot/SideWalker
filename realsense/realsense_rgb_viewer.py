import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()

# 컬러 및 깊이 스트림 설정
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 카메라 스트림 시작
pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()

        # 컬러 이미지와 깊이 이미지 얻기
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 이미지 표시 (컬러 이미지)
        cv2.imshow("Color Image", color_image)

        # 키 입력 대기 (ESC 키로 종료)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break
finally:
    # 파이프라인 정리
    pipeline.stop()

cv2.destroyAllWindows()

