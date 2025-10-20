import torch
import torch.nn as nn
from reward_estimation_model import HALORewardModel
import os


MODEL_PATH = './best_model.pth'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론 시 생성할 후보 액션(궤적) 관련 설정
NUM_CANDIDATE_ACTIONS = 17  # 후보 궤적 개수 (홀수로 설정하는 것이 좋음)
FIXED_LINEAR_V = 0.6        # 후보 궤적의 기본 직진 속도
IMG_SIZE_MASK = 32          # 생성할 궤적 마스크 이미지 크기
NUM_DEPTH_BINS = 16         # 모델이 예측하는 거리 bin의 개수



# --- 1. 모델 로드 ---
model = HALORewardModel(freeze_dino=True).to(DEVICE)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
# 모델의 state_dict를 로드합니다.
# 학습 시 저장된 키와 현재 모델의 키가 다를 경우를 대비하여 strict=False 옵션을 사용할 수 있습니다.
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


mask = torch.randn(1,1,32,32).to(DEVICE)
image = torch.randn(1,3,224,224).to(DEVICE)

predicted_rewards, predicted_depths = model(image, mask)

breakpoint()
