#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from erfpspnet import Net
from collections import OrderedDict
import torch.nn.functional as F

# 1) 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) 사용자 정의 PSPNet 모델 로드
model = Net(22)
# 모델 가중치 로드 (CPU/GPU 자동 매핑)
state_dict = torch.load("./model_best.pth", map_location=device)
# 'module.' prefix 제거
new_state = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state[name] = v
model.load_state_dict(new_state)
model.to(device).eval()

# 3) 컬러 및 깊이 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 4) 클래스 컬러 팔레트 정의 (0~255 인덱스 매핑)
palette_base = torch.tensor([2**25-1, 2**15-1, 2**21-1], dtype=torch.int64)
colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
colors = colors.numpy().astype('uint8')  # RGB 팔레트

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 깊이 맵 처리
        depth_img = np.asanyarray(depth_frame.get_data()) * depth_scale
        depth_vis = cv2.convertScaleAbs(depth_img, alpha=255.0 / max(depth_img.max(), 1e-6))
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 컬러 이미지 처리
        img_bgr = np.asanyarray(color_frame.get_data())  # BGR uint8
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 5) 전처리: numpy→tensor, 정규화(/255)
        tensor_img = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device).float() / 255.0

        # 6) 추론
        with torch.no_grad():
            logits = model(tensor_img)  # shape [1,22,H,W]
        # PSPNet은 dict 반환이 아니므로 바로 사용

        # 7) 예측 마스크
        preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

        # 8) 마스크 컬러화 & 오버레이
        mask_rgb = colors[preds]                 # HxWx3 RGB
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.5, mask_bgr, 0.5, 0)

        # 9) 결과 디스플레이: 세그멘테이션+깊이 나란히
        combined = np.hstack((overlay, depth_colormap))
        cv2.imshow("PSPNet Segmentation + Depth", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
