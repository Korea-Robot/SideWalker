#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# 1) 디바이스 설정
# Sets the computation device to CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) SegFormer 체크포인트 및 모델/전처리 로드
# Loads a pretrained SegFormer model from the segmentation-models-pytorch library.
checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"
model = smp.Segformer.from_pretrained(checkpoint).eval().to(device)

# 3) 전처리 파이프라인: 긴 쪽은 512로, 비율 유지 후 512x512 패딩, ToTensorV2 사용
# Defines an image preprocessing pipeline using Albumentations.
img_size = 512
preprocessing = A.Compose([
    # Resizes the longest side of the image to 512 pixels, maintaining aspect ratio.
    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
    # Pads the image to a 512x512 canvas if it's smaller.
    A.PadIfNeeded(min_height=img_size, min_width=img_size,
                  border_mode=cv2.BORDER_CONSTANT, value=0),
    # Normalizes the image with ImageNet stats.
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Converts the NumPy image to a PyTorch tensor.
    ToTensorV2(),
])

# 4) 팔레트 생성 (0~255 인덱스용)
# Creates a color palette to visualize the segmentation mask with distinct colors.
palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
colors = colors.numpy().astype('uint8')  # RGB palette

# 5) RTSP 스트림 설정
rtsp_url = "192.168.168.105:4001/front_left"
cap = cv2.VideoCapture(rtsp_url)

# Check if the video stream was opened successfully.
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Successfully connected to the stream. Press 'ESC' to exit.")

try:
    while True:
        # Read a frame from the video stream.
        ret, frame = cap.read()

        # If a frame was not received (e.g., end of stream), break the loop.
        if not ret:
            print("Stream ended or connection lost.")
            break

        # The 'frame' from cap.read() is already a BGR NumPy array.
        img_bgr = frame
        
        # Convert BGR (OpenCV's default) to RGB for the model.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 6) 전처리 실행, 여기서 toTensorV2로 CHW torch.Tensor 반환
        # Preprocess the image and convert it to a tensor for the model.
        augmented = preprocessing(image=img_rgb)
        tensor_img = augmented['image'].unsqueeze(0).to(device)  # Add batch dim: 1xCxHxW

        # 7) 추론 및 업샘플링
        # Perform inference without calculating gradients to save memory and computation.
        with torch.no_grad():
            logits = model(tensor_img)
        
        # Some models return a dictionary; handle this case for compatibility.
        if isinstance(logits, dict):
            logits = logits['out']
            
        # Upsample the prediction to match the original image size.
        original_shape = img_rgb.shape[:2]
        logits = F.interpolate(logits, size=original_shape, mode='bilinear', align_corners=False)
        
        # Get the class index with the highest score for each pixel.
        preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

        # 8) 마스크 컬러화 & 오버레이
        # Colorize the prediction mask using the generated palette.
        mask_rgb = colors[preds]  # HxWx3
        
        # Convert the RGB mask back to BGR to use with OpenCV.
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        
        # Blend the original BGR image and the BGR mask.
        overlay = cv2.addWeighted(img_bgr, 0.5, mask_bgr, 0.5, 0)

        # To display side-by-side, ensure both images are in the same color format (BGR).
        images = np.hstack((img_bgr, overlay))
        
        # 9) 디스플레이
        cv2.imshow("Original vs. SegFormer Segmentation", images)
        
        # Exit the loop if the 'ESC' key is pressed.
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting...")
            break

finally:
    # Always release resources when finished.
    print("Releasing resources.")
    cap.release()
    cv2.destroyAllWindows()
