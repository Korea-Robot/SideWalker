import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

# 기존 학습 코드에서 모델과 데이터셋 클래스를 그대로 가져옵니다.
from train import NavigationModel, WorldModel, NavigationDataset

# 이미지 저장을 위한 헬퍼 함수
def unnormalize_and_save_image(tensor, mean, std, filepath):
    """텐서를 정규화 해제하고 PIL 이미지로 저장합니다."""
    # 올바른 mean/std를 위한 텐서 차원 변경
    mean = torch.tensor(mean).view(len(mean), 1, 1)
    std = torch.tensor(std).view(len(std), 1, 1)
    
    # 정규화 해제: (tensor * std) + mean
    tensor = tensor.clone().cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1) # 값을 [0, 1] 범위로 클리핑
    
    # PIL 이미지로 변환
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    
    img.save(filepath)

def inference(args):
    # --- 1. 설정 및 모델 로드 ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 초기화
    nav_model = NavigationModel().to(device)
    world_model = WorldModel(hidden_size=512).to(device)

    # 학습된 가중치 로드
    print(f"Loading Navigation Model from: {args.nav_model_path}")
    nav_model.load_state_dict(torch.load(args.nav_model_path)['nav_model_state_dict'])
    print(f"Loading World Model from: {args.world_model_path}")
    world_model.load_state_dict(torch.load(args.world_model_path)['world_model_state_dict'])

    # 추론 모드로 설정
    nav_model.eval()
    world_model.eval()

    # --- 2. 초기 데이터 로드 ---
    # 데이터셋 로드 (validation set 사용)
    with open(os.path.join(args.dataset_path, 'train_test_split.json'), 'r') as f:
        split_data = json.load(f)
    val_episodes = split_data['validation']

    # 설정값은 학습 때와 동일하게 맞춰줍니다.
    dataset = NavigationDataset(
        dataset_path=args.dataset_path,
        episode_list=val_episodes,
        sequence_length=args.sequence_length
    )
    
    if args.sample_idx >= len(dataset):
        print(f"Error: sample_idx {args.sample_idx} is out of bounds for dataset size {len(dataset)}")
        return

    # 추론에 사용할 하나의 샘플 데이터 가져오기
    initial_data = dataset[args.sample_idx]

    # 결과를 저장할 폴더 생성
    output_dir = os.path.join(args.output_dir, f"sample_{args.sample_idx}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- 3. 초기 상태 및 결과물 저장 ---
    # 초기 시퀀스(Ground Truth)를 이미지로 저장
    print("Saving initial ground truth sequence...")
    for i in range(args.sequence_length):
        unnormalize_and_save_image(initial_data['rgb_seq'][i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], os.path.join(output_dir, f"ground_truth_rgb_{i:02d}.png"))
        unnormalize_and_save_image(initial_data['depth_seq'][i], [0.5], [0.5], os.path.join(output_dir, f"ground_truth_depth_{i:02d}.png"))

    # 추론에 사용할 현재 상태 변수들 초기화 (배치 차원 추가)
    current_rgb_seq = initial_data['rgb_seq'].unsqueeze(0).to(device)
    current_depth_seq = initial_data['depth_seq'].unsqueeze(0).to(device)
    current_semantic_seq = initial_data['semantic_seq'].unsqueeze(0).to(device)
    current_goal_seq = initial_data['goals'].unsqueeze(0).to(device)
    
    # 초기 히든 상태는 0으로 시작 (또는 초기 시퀀스를 한번 실행하여 얻을 수 있음)
    nav_hidden = None
    world_hidden = None

    # --- 4. K-Step 예측 루프 (Imagination Loop) ---
    predicted_rgbs = []
    predicted_depths = []

    with torch.no_grad():
        for k in tqdm(range(args.k_steps), desc="Imagining future steps"):
            # a. 행동 예측
            # NavigationModel은 전체 시퀀스를 받아 마지막 스텝의 행동을 예측하도록 학습되었음
            pred_actions, nav_hidden = nav_model(current_depth_seq, current_goal_seq, nav_hidden)
            last_action = pred_actions[:, -1, :] # 시퀀스의 마지막 예측 행동 사용

            # b. 다음 상태 예측 (상상)
            # WorldModel도 전체 시퀀스와 마지막 행동을 받아 다음 상태를 예측
            # WorldModel 입력에 맞는 action 시퀀스 생성 (여기서는 마지막 행동만 사용)
            action_for_world_model = last_action.unsqueeze(1) # [B, 1, ActionDim] 형태로 만듦

            # WorldModel은 1-step 예측을 위해 마지막 상태와 행동만 필요로 할 수 있지만,
            # 현재 구현은 시퀀스 전체를 받으므로, 마지막 행동만 교체해준다.
            # 여기서는 간결하게 마지막 상태와 예측된 행동으로 다음 상태를 예측하는 컨셉으로 진행
            # (입력 시퀀스를 1로 줄여서 전달)
            next_state_predictions, world_hidden = world_model(
                current_rgb_seq[:, -1:, ...], # 마지막 이미지
                current_depth_seq[:, -1:, ...], # 마지막 이미지
                current_semantic_seq[:, -1:, ...], # 마지막 이미지
                action_for_world_model, # 예측된 행동
                world_hidden # 이전 hidden state
            )

            # 예측된 다음 상태 이미지들
            pred_next_rgb = next_state_predictions['next_rgb']
            pred_next_depth = next_state_predictions['next_depth']
            pred_next_semantic = next_state_predictions['next_semantic']

            # 결과 저장
            predicted_rgbs.append(pred_next_rgb.squeeze(0))
            predicted_depths.append(pred_next_depth.squeeze(0))

            # c. 상태 업데이트 (시퀀스를 한 칸씩 민다)
            # 가장 오래된 프레임을 버리고, 새로 예측된 프레임을 추가
            current_rgb_seq = torch.cat([current_rgb_seq[:, 1:, ...], pred_next_rgb.unsqueeze(1)], dim=1)
            current_depth_seq = torch.cat([current_depth_seq[:, 1:, ...], pred_next_depth.unsqueeze(1)], dim=1)
            current_semantic_seq = torch.cat([current_semantic_seq[:, 1:, ...], pred_next_semantic.unsqueeze(1)], dim=1)
            # Goal은 보통 고정되어 있다고 가정
            current_goal_seq = torch.cat([current_goal_seq[:, 1:, ...], current_goal_seq[:, -1:, ...]], dim=1)

    # --- 5. 예측 결과 저장 ---
    print("Saving predicted future sequence...")
    for i, (rgb_img, depth_img) in enumerate(zip(predicted_rgbs, predicted_depths)):
        unnormalize_and_save_image(rgb_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], os.path.join(output_dir, f"predicted_rgb_{i:02d}.png"))
        unnormalize_and_save_image(depth_img, [0.5], [0.5], os.path.join(output_dir, f"predicted_depth_{i:02d}.png"))

    print("Inference completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for World Model")
    parser.add_argument('--nav_model_path', type=str, required=True, help="Path to the trained navigation model checkpoint.")
    parser.add_argument('--world_model_path', type=str, required=True, help="Path to the trained world model checkpoint.")
    parser.add_argument('--dataset_path', type=str, default='../world_data/imitation_dataset', help="Path to the dataset.")
    parser.add_argument('--output_dir', type=str, default='inference_results', help="Directory to save the output images.")
    parser.add_argument('--k_steps', type=int, default=10, help="Number of future steps to predict (imagine).")
    parser.add_argument('--sequence_length', type=int, default=10, help="Sequence length used during training.")
    parser.add_argument('--sample_idx', type=int, default=0, help="Index of the sample from validation set to use for inference.")
    
    args = parser.parse_args()
    inference(args)