#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import matplotlib.pyplot as plt

# 1) 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) 모델 로드 및 전처리 함수
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights).eval().to(device)
transforms = weights.transforms()

# 3) 색상 팔레트 생성 (21 클래스)
palette_base = torch.tensor([2**25-1, 2**15-1, 2**21-1])
colors = (torch.arange(21, dtype=torch.int32)[:, None] * palette_base) % 255
colors = colors.numpy().astype('uint8')

# 4) RealSense 파이프라인 설정 (컬러 스트림)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

plt.ion()
fig, ax = plt.subplots(figsize=(8,6))
img_plot = None

try:
    while True:
        # 프레임 획득
        frames = pipeline.wait_for_frames()
        c_frame = frames.get_color_frame()
        if not c_frame:
            continue
        color_np = np.asanyarray(c_frame.get_data())  # BGR

        # PIL 변환 및 전처리
        pil_img = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
        input_tensor = transforms(pil_img).unsqueeze(0).to(device)

        # 추론
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        preds = output.argmax(0).byte().cpu().numpy()

        # PIL 인덱스 팔레트 이미지 생성
        seg_img = Image.fromarray(preds).resize(pil_img.size)
        seg_img.putpalette(colors.flatten().tolist())

        # Matplotlib으로 표시
        if img_plot is None:
            img_plot = ax.imshow(seg_img)
            ax.axis('off')
        else:
            img_plot.set_data(seg_img)
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    plt.ioff()
    plt.show()