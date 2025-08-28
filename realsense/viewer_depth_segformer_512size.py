#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# 1) 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) SegFormer 체크포인트 및 모델/전처리 로드
checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"
model = smp.Segformer.from_pretrained(checkpoint).eval().to(device)

# 3) 전처리 파이프라인: 긴 쪽은 512로, 비율 유지 후 512x512 패딩, ToTensorV2 사용
img_size = 512
preprocessing = A.Compose([
    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(min_height=img_size, min_width=img_size,
                  border_mode=cv2.BORDER_CONSTANT, constant_values=0),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# 4) 팔레트 생성 (0~255 인덱스용)
palette_base = torch.tensor([2**25-1, 2**15-1, 2**21-1], dtype=torch.int64)
colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
colors = colors.numpy().astype('uint8')  # RGB 팔레트

# 5) RealSense 컬러 스트림 설정
pipeline = rs.pipeline()
config = rs.config()


# 컬러 및 깊이 스트림 설정
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 카메라 스트림 시작
pipeline.start(config)

# try:
while True:
    frames = pipeline.wait_for_frames()
    c_frame = frames.get_color_frame()
    
    depth_frame = frames.get_depth_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    if not c_frame:
        continue

    # 깊이 이미지를 색상 맵으로 변환
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


    # BGR→RGB 변환
    img_bgr = np.asanyarray(c_frame.get_data())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 6) 전처리 실행, 여기서 toTensorV2로 CHW torch.Tensor 반환
    augmented = preprocessing(image=img_rgb)
    tensor_img = augmented['image'].unsqueeze(0).to(device)  # 1xCxHxW

    # 7) 추론 및 업샘플링
    with torch.no_grad():
        logits = model(tensor_img)
    if isinstance(logits, dict):
        logits = logits['out']
    logits = F.interpolate(logits, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
    preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

    # 8) 마스크 컬러화 & 오버레이
    mask_rgb = colors[preds]            # HxWx3
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, mask_bgr, 0.5, 0)

    # 컬러 이미지와 깊이 이미지 나란히 표시
    images = np.hstack((overlay, depth_colormap))
    # 9) 디스플레이
    cv2.imshow("SegFormer Segmentation", images)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# except KeyboardInterrupt:
#     pass
# finally:
pipeline.stop()
cv2.destroyAllWindows()
