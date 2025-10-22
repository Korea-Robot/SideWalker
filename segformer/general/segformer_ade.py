import matplotlib.pyplot as plt
import torch
import requests
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# 1. Image Processor 및 SegFormer 모델 로드
# mit-b0는 백본(Encoder) 이름이므로, SegFormer 모델 중 이 백본을 사용하는 체크포인트를 사용합니다.
# ADE20k 데이터셋으로 파인튜닝된 SegFormer-B0 모델을 사용합니다.
checkpoint = "nvidia/segformer-b0-finetuned-ade-512-512" 
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)

# 2. 이미지 다운로드 및 전처리
# MaskFormer 예시와 동일한 ADE20k 데이터셋의 이미지를 사용합니다.
url = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
)
image = Image.open(requests.get(url, stream=True).raw)

# 이미지 전처리 (모델 입력 형식에 맞게 변환)
inputs = image_processor(images=image, return_tensors="pt")

# 3. 모델 추론
with torch.no_grad():
    outputs = model(**inputs)

# 4. Semantic Segmentation 맵 후처리 및 시각화 준비
# SegFormer의 후처리 함수를 사용하여 원본 이미지 크기에 맞는 맵을 얻습니다.
predicted_semantic_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]

# 텐서를 NumPy 배열로 변환하고 CPU로 이동
semantic_map_np = predicted_semantic_map.cpu().numpy()

# 5. 시각화 (plt imshow)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# 원본 이미지 표시
axes[0].imshow(image)
axes[0].set_title(f"Original Image (H={image.height}, W={image.width})")
axes[0].axis('off')

# 예측된 Semantic Map 표시
# ADE20k는 150개의 클래스를 가지므로, 'viridis' 또는 'tab20'와 같은 컬러맵을 사용합니다.
im = axes[1].imshow(semantic_map_np, cmap='viridis')
axes[1].set_title(f"SegFormer-B0 Predicted Semantic Map (ADE20k)")
axes[1].axis('off')

# 컬러바 추가 (각 클래스 ID가 어떤 색상인지 표시)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
