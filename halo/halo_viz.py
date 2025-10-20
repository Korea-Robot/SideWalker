# 파일명: model_inference_viz.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import time

# skimage 라이브러리가 필요합니다. 설치: pip install scikit-image
from skimage.draw import line

# 별도 파일로부터 모델 클래스를 임포트
from reward_estimation_model import HALORewardModel

# ==============================================================================
# --- 🚀 Configuration ---
# ==============================================================================
MODEL_PATH = './best_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CANDIDATE_ACTIONS = 21
FIXED_LINEAR_V = 0.5
IMG_SIZE_MASK = 224
IMG_SIZE_RGB = 224
NUM_DEPTH_BINS = 25

# ==============================================================================
# --- 궤적 생성 함수들 (원본 유지) ---
# ==============================================================================
def generate_trajectory_mask_from_df(df, img_size):
    if len(df) < 2: return np.zeros((img_size, img_size), dtype=np.uint8)
    delta_t = df['timestamp'].diff().fillna(0) / 1000.0
    x, y, theta = 0.0, 0.0, 0.0
    odom_list = [[x, y, theta]]
    for i in range(1, len(df)):
        v = df['manual_linear_x'].iloc[i]; w = df['manual_angular_z'].iloc[i]; dt = delta_t.iloc[i]
        theta += w * dt; x += v * np.cos(theta) * dt; y += v * np.sin(theta) * dt
        odom_list.append([x, y, theta])
    odom_segment = np.array(odom_list)
    x0, y0, theta0 = odom_segment[0]
    coords_translated = odom_segment[:, :2] - np.array([x0, y0])
    c, s = np.cos(-theta0), np.sin(-theta0)
    rotation_matrix = np.array([[c, -s], [s, c]])
    ego_coords = (rotation_matrix @ coords_translated.T).T
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    max_range = 1.0
    u = np.clip(((ego_coords[:, 0] / max_range) * (img_size - 1)).astype(int), 0, img_size - 1)
    v = np.clip((ego_coords[:, 1] / (max_range / 1.3) * (img_size - 1) / 2 + (img_size / 2)).astype(int), 0, img_size - 1)
    for i in range(len(u) - 1):
        rr, cc = line(u[i], v[i], u[i+1], v[i+1])
        mask[rr, cc] = 1
    mask[0, img_size // 2] = 1
    return mask

def generate_candidate_masks():
    angular_velocities = np.linspace(-1.0, 1.0, NUM_CANDIDATE_ACTIONS)
    candidate_masks = []
    for w in angular_velocities:
        duration = 2.0; hz = 10; num_points = int(duration * hz)
        timestamps = np.arange(num_points) * (1000 / hz)
        linear_v = FIXED_LINEAR_V / (1 + 0.5 * abs(w))
        dummy_df = pd.DataFrame({'timestamp': timestamps, 'manual_linear_x': [linear_v] * num_points, 'manual_angular_z': [w] * num_points})
        mask_np = generate_trajectory_mask_from_df(dummy_df, img_size=IMG_SIZE_MASK)
        candidate_masks.append(torch.from_numpy(mask_np).float())
    return torch.stack(candidate_masks).unsqueeze(1).to(DEVICE), angular_velocities

# ==============================================================================
# --- 🚀 Main Execution ---
# ==============================================================================
def main():
    print(f"Using device: {DEVICE}")

    # --- 1. 모델 및 후보 궤적 로드 (한 번만 수행) ---
    model = HALORewardModel(freeze_dino=True).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"'{MODEL_PATH}' 에서 학습된 모델을 로드합니다.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"경고: '{MODEL_PATH}' 파일을 찾을 수 없습니다. 초기화된 모델로 계속합니다.")
    model.eval()

    candidate_masks, angular_velocities = generate_candidate_masks()

    # --- 2. Matplotlib 대화형 모드 설정 ---
    plt.ion() # 대화형 모드 켜기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    print("실시간 시각화를 시작합니다. 중지하려면 터미널에서 Ctrl+C를 누르세요.")

    # --- 3. 실시간 추론 및 시각화 루프 ---
    try:
        while True:
            # --- 가상 데이터 실시간 생성 (실제 사용 시 카메라 데이터로 대체) ---
            random_idx = random.randint(0, 1000)
            rgb_tensor_disp = torch.rand(3, IMG_SIZE_RGB, IMG_SIZE_RGB)
            rgb_tensor_inf = rgb_tensor_disp.unsqueeze(0).to(DEVICE)
            # Ground-truth 거리가 시간에 따라 변하는 것처럼 보이게 함
            phase_shift = time.time() * 2
            ground_truth_dist = np.cos(np.linspace(0, np.pi * 2, NUM_DEPTH_BINS) + phase_shift) * 0.4 + 0.5

            # --- 모델 추론 ---
            with torch.no_grad():
                rgb_expanded = rgb_tensor_inf.repeat(NUM_CANDIDATE_ACTIONS, 1, 1, 1)
                predicted_rewards, predicted_depths = model(rgb_expanded, candidate_masks)
                predicted_dist = predicted_depths[0].cpu().numpy()
                rewards = predicted_rewards.squeeze().cpu().numpy()

            # --- 시각화 업데이트 ---
            # 이전 프레임의 내용 지우기
            ax1.cla()
            ax2.cla()
            # twinx()로 생성된 세 번째 축도 수동으로 지워야 함
            if 'ax3' in locals() and ax3.figure == fig:
                ax3.remove()

            # 왼쪽: RGB 이미지
            ax1.imshow(rgb_tensor_disp.permute(1, 2, 0))
            ax1.set_title(f"Input Image (Frame #{random_idx})")
            ax1.axis('off')

            # 오른쪽: 보상 및 거리 비교 그래프
            ax2.set_title('Reward vs. Distance Analysis')
            ax2.set_xlabel('Angular Velocity (rad/s)')
            
            color_reward = 'cornflowerblue'
            ax2.set_ylabel('Predicted Reward', color=color_reward, fontsize=12)
            ax2.bar(angular_velocities, rewards, width=0.1, color=color_reward, alpha=0.9, label='Predicted Reward')
            ax2.tick_params(axis='y', labelcolor=color_reward)
            
            ax3 = ax2.twinx()
            ax3.set_ylabel('Normalized Distance (1=Far)', fontsize=12)
            ax3.set_ylim(-0.05, 1.1)
            
            distance_x_coords = np.linspace(angular_velocities.min(), angular_velocities.max(), NUM_DEPTH_BINS)
            ax3.plot(distance_x_coords, ground_truth_dist, color='crimson', marker='o', linestyle='--', label='Ground-Truth Distance')
            ax3.plot(distance_x_coords, predicted_dist, color='limegreen', marker='x', linestyle='-', label='Predicted Distance')
            
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='upper center')
            
            fig.tight_layout()
            
            # 그래프 창을 업데이트하고 잠시 대기
            plt.pause(0.1)

    except KeyboardInterrupt:
        print("\n시각화가 중지되었습니다.")
    finally:
        plt.ioff() # 대화형 모드 끄기
        print("마지막 프레임을 보려면 창을 닫으세요.")
        plt.show() # 마지막 프레임을 보여주고 스크립트 종료 대기

if __name__ == '__main__':
    main()
