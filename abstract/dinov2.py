import torch
from PIL import Image
import requests
from torchvision import transforms

# 1. DINOv2 모델 불러오기 (ViT-Small, 14x14 패치 크기)
# 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14' 등 다양한 모델 선택 가능

# 2. Downsteam Task
# Pretrained heads - Image Classification 
# Pretrained heads - Depth estimation
# Pretrained heads - Semantic Segmentation
# Pretrained heads - Zero shot task with dino

import torch

# DINOv2
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# # DINOv2 with registers
# dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
# dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
# dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
# dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')       # 55 fps
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')   # 55 fps 
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg') # 40 fps
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') # 38 fps

# GPU가 있다면 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

# 2. 이미지 준비하기
# 인터넷에서 이미지 불러오기
url = "https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Breeds.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# 또는 로컬 파일에서 이미지 불러오기
# img = Image.open("my_image.jpg").convert('RGB')

# 3. 이미지 전처리
# DINOv2 모델이 요구하는 형식으로 이미지를 변환합니다.
transform = transforms.Compose([
    transforms.Resize((360,640), interpolation=transforms.InterpolationMode.BICUBIC),
    # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

img_tensor = transform(img).unsqueeze(0).to(device) # unsqueeze(0)으로 배치 차원 추가

# 4. 모델로 특징 추출 실행
import time 
start = time.time()
for i in range(100):
    with torch.no_grad():
        features = dinov2_vits14(img_tensor)

print()
total_time= time.time()-start
print('fps :',100/total_time) 
print()
# 5. 결과 확인
# features 텐서는 [CLS] 토큰에 대한 이미지 전체의 특징을 담고 있습니다.
print("추출된 특징 벡터의 형태:", features.shape)
# 출력 예시: torch.Size([1, 384]) -> 1개 이미지, 384차원 벡터

# 패치(patch) 레벨의 특징을 원한다면
features_dict = dinov2_vits14.forward_features(img_tensor)
patch_features = features_dict['x_norm_patchtokens']
print("패치 특징의 형태:", patch_features.shape)
# 출력 예시: torch.Size([1, 256, 384]) -> 1개 이미지, 256개 패치, 각 패치는 384차원 벡터