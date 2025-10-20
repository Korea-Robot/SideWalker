#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision.transforms as T
from torchvision import models

# -----------------------
# 1) 세그멘테이션 모델 로드
# -----------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# 클래스별 컬러맵 (21개 클래스 중 일부 예시)
colormap = {
    0: [0.0, 0.0, 0.0],      # background – 검정
    15:[1.0, 0.0, 0.0],      # person     – 빨강
    1: [0.0, 1.0, 0.0],      # aeroplane  – 초록
    2: [0.0, 0.0, 1.0],      # bicycle    – 파랑
    # ... 필요에 따라 추가
}

# -----------------------
# 2) 세그멘테이션 함수 정의
# -----------------------
def get_segmentation_from_rgb(rgb_image: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((480, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # ← 여기만 .to(device)로 바꿔야 합니다
    input_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_t)['out'][0]
    mask = out.argmax(0).byte().cpu().numpy()
    mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


# -----------------------
# 3) RealSense 스트림 설정 및 align
# -----------------------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(cfg)

# depth → color로 align
align_to = rs.stream.color
align = rs.align(align_to)

# 깊이 스케일 (m)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    while True:
        # 프레임 획득 & align
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # NumPy 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())  # BGR

        # -----------------------
        # 4) 세그멘테이션 실행
        # -----------------------
        seg_mask = get_segmentation_from_rgb(color_image)

        # -----------------------
        # 5) 3D 포인트, 컬러 생성
        # -----------------------
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        pts, cols = [], []
        H, W = seg_mask.shape
        for v in range(H):
            for u in range(W):
                class_id = int(seg_mask[v, u])
                if class_id == 0 or class_id not in colormap:
                    continue
                z = depth_image[v, u] * depth_scale
                if z <= 0:
                    continue
                x, y, z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                pts.append([x, y, z])
                cols.append(colormap[class_id])

        # -----------------------
        # 6) Open3D로 시각화
        # -----------------------
        if pts:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(pts))
            pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
            o3d.visualization.draw_geometries([pcd])
        else:
            print("검출된 포인트가 없습니다.")

        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
