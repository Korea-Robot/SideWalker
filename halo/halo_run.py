import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
#from skimage.draw import line

# --- 프로젝트의 다른 파일에서 클래스를 임포트합니다 ---
from reward_estimation_dataset import NavigationDataset
#from reward_estimation_model import HALORewardModel

# ==============================================================================
# --- 🚀 Configuration ---
# ==============================================================================

# ### 중요 ###: 아래 경로들을 자신의 환경에 맞게 수정해주세요.
# 사용할 데이터 디렉토리
DATA_DIR = '../../data/ilrl/0903_inside_night' 
# 학습된 모델 가중치 파일 경로
MODEL_PATH = './best_model.pth' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_trajectory_mask_from_df(df, img_size):
    """
    속도 명령이 담긴 DataFrame을 기반으로 자기 중심(egocentric) 궤적 마스크를 생성합니다.
    이 함수는 NavigationDataset의 로직을 간소화하여 가져왔습니다.
    """
    if len(df) < 2:
        return np.zeros((img_size, img_size), dtype=np.uint8)

    # 1. Odometry 계산
    delta_t = df['timestamp'].diff().fillna(0) / 1000.0
    x, y, theta = 0.0, 0.0, 0.0
    odom_list = [[x, y, theta]]
    for i in range(1, len(df)):
        v = df['manual_linear_x'].iloc[i]
        w = df['manual_angular_z'].iloc[i]
        dt = delta_t.iloc[i]
        theta += w * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        odom_list.append([x, y, theta])
    odom_segment = np.array(odom_list)

    # 2. 자기 중심(Egocentric) 좌표계로 변환
    x0, y0, theta0 = odom_segment[0]
    coords_translated = odom_segment[:, :2] - np.array([x0, y0])
    c, s = np.cos(-theta0), np.sin(-theta0)
    rotation_matrix = np.array([[c, -s], [s, c]])
    ego_coords = (rotation_matrix @ coords_translated.T).T

    # 3. 마스크 이미지에 궤적 그리기
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    max_range = 1.0 # 궤적 시각화를 위한 가상 최대 거리

    # x축(전방)은 이미지 높이, y축(좌우)은 이미지 너비에 매핑
    u = np.clip(((ego_coords[:, 0] / max_range) * (img_size - 1)).astype(int), 0, img_size - 1)
    v = np.clip((ego_coords[:, 1] / (max_range / 1.3) * (img_size - 1) / 2 + (img_size / 2)).astype(int), 0, img_size - 1)

    for i in range(len(u) - 1):
        rr, cc = line(u[i], v[i], u[i+1], v[i+1])
        mask[rr, cc] = 1

    mask[0, img_size // 2] = 1 # 시작점 표시
    return mask


def generate_candidate_masks():
    """ NUM_CANDIDATE_ACTIONS 개수만큼 다양한 곡률의 후보 궤적 마스크를 생성합니다. """
    angular_velocities = np.linspace(-1.0, 1.0, NUM_CANDIDATE_ACTIONS)
    candidate_masks = []

    for w in angular_velocities:
        duration = 2.0; hz = 10; num_points = int(duration * hz)
        timestamps = np.arange(num_points) * (1000 / hz)
        # 회전이 클수록 직진 속도를 약간 줄여 현실적인 궤적 생성
        linear_v = FIXED_LINEAR_V / (1 + 0.5 * abs(w))

        dummy_df = pd.DataFrame({
            'timestamp': timestamps,
            'manual_linear_x': [linear_v] * num_points,
            'manual_angular_z': [w] * num_points,
        })

        mask_np = generate_trajectory_mask_from_df(dummy_df, img_size=IMG_SIZE_MASK)
        candidate_masks.append(torch.from_numpy(mask_np).float())

    return torch.stack(candidate_masks).unsqueeze(1).to(DEVICE), angular_velocities


def main():
    print(f"Using device: {DEVICE}")

    # --- 1. 모델 로드 ---
    model = HALORewardModel(freeze_dino=True).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    # 모델의 state_dict를 로드합니다.
    # 학습 시 저장된 키와 현재 모델의 키가 다를 경우를 대비하여 strict=False 옵션을 사용할 수 있습니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()


    # --- 3. 후보 궤적 마스크 생성 ---
    candidate_masks, angular_velocities = generate_candidate_masks()


    with torch.no_grad():
            # 단일 이미지를 후보 궤적 개수(K)만큼 복제하여 배치 생성
            rgb_expanded = rgb_tensor_inf.repeat(NUM_CANDIDATE_ACTIONS, 1, 1, 1)

            # 모든 후보 액션에 대한 보상과 거리 동시 예측
            predicted_rewards, predicted_depths = model(rgb_expanded, candidate_masks)

            # 가장 보상이 높은 액션 하나에 대한 거리 예측값만 사용
            # (모든 액션에 대해 동일한 이미지이므로 거리 예측값은 거의 동일함)
            predicted_dist = predicted_depths[0].cpu().numpy()
            rewards = predicted_rewards.squeeze().cpu().numpy()

        # --- 5. 시각화 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # 왼쪽: 원본 RGB 이미지
        ax1.imshow(rgb_tensor_disp.permute(1, 2, 0))
        ax1.set_title(f"Input Image (Sample #{random_idx})")
        ax1.axis('off')

        # 오른쪽: 보상 및 거리 비교 그래프
        ax2.set_title('Reward vs. Distance Analysis')
        ax2.set_xlabel('Angular Velocity (rad/s)')

        # 파란색 막대: 예측된 보상 (왼쪽 Y축)
        color_reward = 'cornflowerblue'
        ax2.set_ylabel('Predicted Reward', color=color_reward, fontsize=12)
        ax2.bar(angular_velocities, rewards, width=0.1, color=color_reward, alpha=0.9, label='Predicted Reward')
        ax2.tick_params(axis='y', labelcolor=color_reward)

        # 오른쪽 Y축을 공유하는 두 번째 축 생성
        ax3 = ax2.twinx()
        ax3.set_ylabel('Normalized Distance (1=Far)', fontsize=12)
        ax3.set_ylim(-0.05, 1.1)

        # 빨간색 점선: 실제(Ground-Truth) 거리
        distance_x_coords = np.linspace(angular_velocities.min(), angular_velocities.max(), NUM_DEPTH_BINS)
        ax3.plot(distance_x_coords, ground_truth_dist, color='crimson', marker='o', linestyle='--', label='Ground-Truth Distance')

        # 초록색 실선: 모델이 예측한 거리
        ax3.plot(distance_x_coords, predicted_dist, color='limegreen', marker='x', linestyle='-', label='Predicted Distance')

        # 범례(legend)를 하나로 합치기
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper center')

        fig.tight_layout()
        plt.show()
