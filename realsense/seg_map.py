import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from scipy.ndimage import label
from collections import deque

# ---- 1) 세그멘테이션 모델 로드 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"
model = smp.from_pretrained(checkpoint).eval().to(device)
preproc = A.Compose.from_pretrained(checkpoint)

# ---- 2) RealSense 초기화 ----
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pc = rs.pointcloud()
pipeline.start(config)

# ---- 3) BEV 파라미터 ----
map_size    = 3.0  # meter (3m x 3m)
voxel_size  = 0.01 # meter (1cm 해상도)
grid_dim    = int(map_size / voxel_size)  # 300
half_width  = map_size/2
time_window = 5
buffer_occ  = deque(maxlen=time_window)
buffer_seg  = deque(maxlen=time_window)

# 임의 색상 매핑 (클래스 0은 배경)
num_classes = 19  # Cityscapes 기준 클래스 수
colors = [tuple(np.random.randint(50,255,3).tolist()) for _ in range(num_classes)]

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 3.1) 이미지 → 세그멘테이션 마스크
        color_img = np.asanyarray(color_frame.get_data())
        aug = preproc(image=color_img)
        inp = torch.from_numpy(aug["image"]).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        mask = torch.nn.functional.interpolate(out, size=color_img.shape[:2],
                                               mode="bilinear", align_corners=False)
        mask = mask[0].argmax(0).cpu().numpy().astype(np.uint8)  # H×W

        # 3.2) PointCloud 생성 & XY 투영
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,3)  # N×3
        tex   = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1,2)  # N×2

        # pixel 좌표로 변환
        u = (tex[:,0] * color_img.shape[1]).astype(np.int32)
        v = ((1-tex[:,1]) * color_img.shape[0]).astype(np.int32)
        valid_uv = (u>=0)&(u<color_img.shape[1])&(v>=0)&(v<color_img.shape[0])

        xy    = verts[valid_uv,:2]  # N_valid×2
        labels= mask[v[valid_uv], u[valid_uv]]  # N_valid

        # 3.3) Map 좌표계로 shift & voxel 인덱스 계산
        xy_shift = xy + half_width  # (0~map_size)
        idx      = np.floor(xy_shift/voxel_size).astype(np.int32)
        in_map   = (idx[:,0]>=0)&(idx[:,0]<grid_dim)&(idx[:,1]>=0)&(idx[:,1]<grid_dim)
        idx      = idx[in_map]
        labels   = labels[in_map]

        # 3.4) Occupancy & Segmentation Grid 생성
        occ_grid = np.zeros((grid_dim,grid_dim), dtype=np.uint8)
        seg_grid = np.zeros((grid_dim,grid_dim), dtype=np.uint8)

        # 단순히 마지막 포인트 레이블 덮어쓰기; 필요시 평균/최빈값 사용
        occ_grid[idx[:,1], idx[:,0]] = 1
        seg_grid[idx[:,1], idx[:,0]] = labels

        buffer_occ.append(occ_grid)
        buffer_seg.append(seg_grid)

        # 3.5) 시계열 스택 & 연결 요소 라벨링
        if len(buffer_occ) < time_window:
            bev = occ_grid*255
            bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        else:
            vol_occ = np.stack(buffer_occ, axis=0)
            vol_seg = np.stack(buffer_seg, axis=0)
            # 구조체: 시간+XY 3-연결
            structure = np.ones((3,3,3), dtype=np.int)
            labeled, num = label(vol_occ, structure=structure)

            # 최신 프레임 라벨만
            lab2d = labeled[-1]
            bev = np.zeros((grid_dim,grid_dim,3), dtype=np.uint8)
            # 각 component에 seg_grid 값 반영
            for comp_id in range(1, num+1):
                mask2 = (lab2d==comp_id)
                # component 내 최빈 세그멘테이션 클래스 추출
                comp_labels = vol_seg[:, mask2]
                flat = comp_labels.ravel()
                cls = np.bincount(flat).argmax()
                bev[mask2] = colors[cls]

        # 3.6) 화면에 띄우기
        disp = cv2.resize(bev, (600,600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("3m x 3m BEV Segmentation", disp)

        if cv2.waitKey(1)==27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
