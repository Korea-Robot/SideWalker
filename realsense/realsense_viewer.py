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

        # check type shape
        # cc = color_image
        # dd = depth_image
        #breakpoint()

        # 깊이 이미지를 색상 맵으로 변환
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 컬러 이미지와 깊이 이미지 나란히 표시
        images = np.hstack((color_image, depth_colormap))

        # 이미지를 화면에 표시
        cv2.imshow("Color and Depth Image", images)

        # 키 입력 대기 (ESC 키로 종료)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break
finally:
    # 파이프라인 정리
    pipeline.stop()

cv2.destroyAllWindows()

