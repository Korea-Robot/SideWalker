from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import torch # torch.no_grad()를 위해 추가
import matplotlib.pyplot as plt # 시각화를 위해 추가
import numpy as np # 텐서 변환을 위해 추가 (plt가 numpy 배열을 선호)

# load MaskFormer fine-tuned on COCO panoptic segmentation
image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

# 추론 시에는 gradient 계산이 필요 없으므로 torch.no_grad() 사용
with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to image_processor for postprocessing
result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]

# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
predicted_panoptic_map = result["segmentation"] # torch.Tensor

# -----------------------------------------------
# 1. 세그멘테이션 맵(ID 맵)만 시각화하기
# -----------------------------------------------
# Matplotlib은 CPU의 NumPy 배열을 입력으로 받는 것이 표준입니다.
panoptic_map_np = predicted_panoptic_map.cpu().numpy()

plt.figure(figsize=(10, 8))
# cmap="tab20"은 ID별로 색상을 다르게 잘 구분해 줍니다.
plt.imshow(panoptic_map_np, cmap="tab20")
plt.axis('off') # 축 정보 숨기기
plt.title("Panoptic Segmentation Map (ID-based)")
plt.show()


# -----------------------------------------------
# 2. 원본 이미지 위에 겹쳐서 시각화하기 (추천) 🖼️
# -----------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(image) # 1. 원본 이미지를 먼저 그립니다.
# 2. 그 위에 세그멘테이션 맵을 alpha(투명도) 0.6으로 겹쳐 그립니다.
plt.imshow(panoptic_map_np, cmap="tab20", alpha=0.6)
plt.axis('off')
plt.title("Original Image + Panoptic Segmentation Overlay")
plt.show()
