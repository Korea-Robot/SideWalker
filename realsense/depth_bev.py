import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from scipy.ndimage import label

#--- 1) RealSense 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pc = rs.pointcloud()
pipeline.start(config)

#--- 2) 파라미터
voxel_size = 0.1              # meter 단위로 XY 그리드 해상도
grid_width, grid_height = 200, 200  # BEV 해상도 (cells)
time_window = 5               # 몇 프레임을 시간축으로 쌓을지
buffer = deque(maxlen=time_window)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        #--- 3) PointCloud 생성
        pc.map_to(color)
        points = pc.calculate(depth)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # N×3

        #--- 4) Z 제거 & XY voxelization
        xy = vtx[:, :2]  # x, y 만
        # Shift & scale to positive 인덱스로
        xy_shifted = xy + np.array([grid_width*voxel_size/2, grid_height*voxel_size/2])
        idx = np.floor(xy_shifted / voxel_size).astype(np.int32)
        # 유효 영역 필터링
        mask = (idx[:,0] >= 0) & (idx[:,0] < grid_width) & (idx[:,1] >= 0) & (idx[:,1] < grid_height)
        idx = idx[mask]

        # occupancy grid 생성 (H×W)
        occ = np.zeros((grid_height, grid_width), dtype=np.uint8)
        occ[idx[:,1], idx[:,0]] = 1

        #--- 5) Temporal stacking
        buffer.append(occ)
        if len(buffer) < time_window:
            # 아직 볼륨 크기 미달 → 단순 BEV만 시각화
            bev = cv2.resize(occ*255, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("BEV Occupancy", bev)
        else:
            # 볼륨 (T, H, W)
            vol = np.stack(buffer, axis=0)  # shape: (T, H, W)
            # 6-connectivity: 시간+xy 모두 이웃으로 간주
            structure = np.ones((3,3,3), dtype=np.int)
            labeled, num = label(vol, structure=structure)

            # 시각화: 최신 프레임(T-1) 의 레이블만 2D로 그리기
            labels_2d = labeled[-1]
            # 레이블 값별로 임의 색상 매핑
            colormap = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            for seg_id in range(1, num+1):
                mask2 = (labels_2d == seg_id)
                color_rand = tuple(np.random.randint(50, 255, size=3).tolist())
                colormap[mask2] = color_rand

            bev = cv2.resize(colormap, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Spatio-Temporal Segments (BEV)", bev)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

