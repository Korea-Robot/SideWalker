import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import cv2

# 기존 모델 클래스들을 import 한다고 가정 (동일한 구조)
from train import NavigationModel, WorldModel

class FutureStatePredictor:
    def __init__(self, 
                 nav_model_path: str, 
                 world_model_path: str, 
                 device: str = 'cuda:0'):
        """
        Future state prediction을 위한 클래스
        
        Args:
            nav_model_path: 학습된 navigation model의 체크포인트 경로
            world_model_path: 학습된 world model의 체크포인트 경로
            device: 사용할 디바이스 ('cuda:0', 'cpu' 등)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.nav_model = self._load_navigation_model(nav_model_path)
        self.world_model = self._load_world_model(world_model_path)
        
        # 전처리 transforms
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Models loaded successfully on device: {self.device}")
    
    def _load_navigation_model(self, model_path: str) -> NavigationModel:
        """Navigation model 로드"""
        model = NavigationModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['nav_model_state_dict'])
        model.eval()
        return model
    
    def _load_world_model(self, world_model_path: str) -> WorldModel:
        """World model 로드"""
        model = WorldModel(hidden_size=512).to(self.device)
        checkpoint = torch.load(world_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['world_model_state_dict'])
        model.eval()
        return model
    
    def preprocess_images(self, rgb_path: str, depth_path: str, semantic_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """이미지 전처리"""
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        semantic_img = Image.open(semantic_path).convert('RGB')
        
        rgb_tensor = self.rgb_transform(rgb_img).unsqueeze(0)  # (1, 3, 224, 224)
        depth_tensor = self.depth_transform(depth_img).unsqueeze(0)  # (1, 1, 224, 224)
        semantic_tensor = self.rgb_transform(semantic_img).unsqueeze(0)  # (1, 3, 224, 224)
        
        return rgb_tensor.to(self.device), depth_tensor.to(self.device), semantic_tensor.to(self.device)
    
    def predict_future_states(self, 
                            initial_rgb: torch.Tensor,
                            initial_depth: torch.Tensor, 
                            initial_semantic: torch.Tensor,
                            goal_position: List[float],
                            k_steps: int = 5,
                            use_predicted_actions: bool = True) -> Dict:
        """
        k step 이후의 future state들을 예측
        
        Args:
            initial_rgb: 초기 RGB 이미지 (1, 3, 224, 224)
            initial_depth: 초기 depth 이미지 (1, 1, 224, 224)
            initial_semantic: 초기 semantic 이미지 (1, 3, 224, 224)
            goal_position: 목표 위치 [x, y]
            k_steps: 예측할 미래 스텝 수
            use_predicted_actions: True면 nav model로 action 예측, False면 랜덤 action 사용
            
        Returns:
            Dictionary containing predicted future states
        """
        with torch.no_grad():
            # 초기 상태 설정
            current_rgb = initial_rgb
            current_depth = initial_depth
            current_semantic = initial_semantic
            current_position = torch.tensor([0.0, 0.0], device=self.device).unsqueeze(0)  # 임시 위치
            current_heading = torch.tensor([0.0], device=self.device).unsqueeze(0)  # 임시 방향
            
            goal_tensor = torch.tensor(goal_position, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # 예측 결과를 저장할 리스트들
            predicted_states = {
                'rgb_sequence': [current_rgb.cpu()],
                'depth_sequence': [current_depth.cpu()],
                'semantic_sequence': [current_semantic.cpu()],
                'action_sequence': [],
                'position_sequence': [current_position.cpu()],
                'heading_sequence': [current_heading.cpu()]
            }
            
            # GRU hidden states 초기화
            nav_hidden = None
            world_hidden = None
            
            # k step 동안 반복 예측
            for step in range(k_steps):
                print(f"Predicting step {step + 1}/{k_steps}")
                
                # 1. Navigation model로 action 예측 (depth + goal -> action)
                if use_predicted_actions:
                    # Depth sequence를 만들기 위해 현재 depth를 sequence 형태로 변환
                    depth_seq = current_depth.unsqueeze(1)  # (1, 1, 1, 224, 224)
                    goal_seq = goal_tensor.unsqueeze(1)     # (1, 1, 2)
                    
                    predicted_actions, nav_hidden = self.nav_model(depth_seq, goal_seq, nav_hidden)
                    current_action = predicted_actions[:, 0]  # 첫 번째 (유일한) timestep의 action
                else:
                    # 랜덤 action 생성
                    current_action = torch.randn(1, 2, device=self.device) * 0.5  # [-1, 1] 범위로 제한
                    current_action = torch.tanh(current_action)
                
                predicted_states['action_sequence'].append(current_action.cpu())
                
                # 2. World model로 다음 상태 예측
                # Sequence 형태로 변환 (모든 텐서를 (batch, seq_len, ...) 형태로)
                rgb_seq = current_rgb.unsqueeze(1)          # (1, 1, 3, 224, 224)
                depth_seq = current_depth.unsqueeze(1)      # (1, 1, 1, 224, 224)
                semantic_seq = current_semantic.unsqueeze(1) # (1, 1, 3, 224, 224)
                action_seq = current_action.unsqueeze(1)    # (1, 1, 2)
                
                next_state_pred, world_hidden = self.world_model(
                    rgb_seq, depth_seq, semantic_seq, action_seq, world_hidden
                )
                
                # 3. 예측된 다음 상태를 현재 상태로 업데이트
                current_rgb = next_state_pred['next_rgb'].unsqueeze(1)        # (1, 1, 3, H, W) -> (1, 3, H, W)
                current_depth = next_state_pred['next_depth'].unsqueeze(1)    # (1, 1, 1, H, W) -> (1, 1, H, W)  
                current_semantic = next_state_pred['next_semantic'].unsqueeze(1) # (1, 1, 3, H, W) -> (1, 3, H, W)
                current_position = next_state_pred['next_position']           # (1, 2)
                current_heading = next_state_pred['next_heading']             # (1, 1)
                
                # 결과 저장
                predicted_states['rgb_sequence'].append(current_rgb.cpu())
                predicted_states['depth_sequence'].append(current_depth.cpu())
                predicted_states['semantic_sequence'].append(current_semantic.cpu())
                predicted_states['position_sequence'].append(current_position.cpu())
                predicted_states['heading_sequence'].append(current_heading.cpu())
            
            return predicted_states
    
    def tensor_to_image(self, tensor: torch.Tensor, image_type: str = 'rgb') -> np.ndarray:
        """텐서를 이미지로 변환"""
        if image_type == 'rgb':
            # RGB denormalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = tensor.squeeze().permute(1, 2, 0).numpy()
            img = img * std + mean
            img = np.clip(img, 0, 1)
        elif image_type == 'depth':
            # Depth denormalization
            img = tensor.squeeze().numpy()
            img = (img * 0.5) + 0.5  # [-1, 1] -> [0, 1]
            img = np.clip(img, 0, 1)
        else:  # semantic
            # Semantic은 RGB와 동일하게 처리
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = tensor.squeeze().permute(1, 2, 0).numpy()
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        return img
    
    def visualize_predictions(self, predicted_states: Dict, save_path: str = None):
        """예측 결과를 시각화"""
        k_steps = len(predicted_states['rgb_sequence']) - 1  # 초기 상태 제외
        
        fig, axes = plt.subplots(3, k_steps + 1, figsize=(4 * (k_steps + 1), 12))
        if k_steps == 0:
            axes = axes.reshape(3, 1)
        
        for step in range(k_steps + 1):
            # RGB 이미지
            rgb_img = self.tensor_to_image(predicted_states['rgb_sequence'][step], 'rgb')
            axes[0, step].imshow(rgb_img)
            axes[0, step].set_title(f'RGB - Step {step}')
            axes[0, step].axis('off')
            
            # Depth 이미지
            depth_img = self.tensor_to_image(predicted_states['depth_sequence'][step], 'depth')
            axes[1, step].imshow(depth_img, cmap='viridis')
            axes[1, step].set_title(f'Depth - Step {step}')
            axes[1, step].axis('off')
            
            # Semantic 이미지
            semantic_img = self.tensor_to_image(predicted_states['semantic_sequence'][step], 'semantic')
            axes[2, step].imshow(semantic_img)
            axes[2, step].set_title(f'Semantic - Step {step}')
            axes[2, step].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_predicted_sequence(self, predicted_states: Dict, output_dir: str):
        """예측된 시퀀스를 개별 이미지 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'semantic'), exist_ok=True)
        
        k_steps = len(predicted_states['rgb_sequence']) - 1
        
        for step in range(k_steps + 1):
            # RGB 저장
            rgb_img = self.tensor_to_image(predicted_states['rgb_sequence'][step], 'rgb')
            rgb_img = (rgb_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'rgb', f'{step:04d}.png'), 
                       cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            # Depth 저장
            depth_img = self.tensor_to_image(predicted_states['depth_sequence'][step], 'depth')
            depth_img = (depth_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'depth', f'{step:04d}.png'), depth_img)
            
            # Semantic 저장
            semantic_img = self.tensor_to_image(predicted_states['semantic_sequence'][step], 'semantic')
            semantic_img = (semantic_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'semantic', f'{step:04d}.png'), 
                       cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR))
        
        # 메타데이터 저장
        metadata = {
            'k_steps': k_steps,
            'actions': [action.numpy().tolist() for action in predicted_states['action_sequence']],
            'positions': [pos.numpy().tolist() for pos in predicted_states['position_sequence']],
            'headings': [head.numpy().tolist() for head in predicted_states['heading_sequence']]
        }
        
        with open(os.path.join(output_dir, 'prediction_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Predicted sequence saved to: {output_dir}")


def main():
    """사용 예시"""
    # 모델 경로 설정
    nav_model_path = 'checkpoints/nav_model_epoch_100.pth'
    world_model_path = 'checkpoints/world_model_epoch_100.pth'
    
    # Predictor 초기화
    predictor = FutureStatePredictor(nav_model_path, world_model_path, device='cuda:0')
    
    # 초기 상태 이미지 로드 (예시)
    rgb_path = '../imitation/imitation_dataset/episode_0001/rgb/0000.png'
    depth_path = '../imitation/imitation_dataset/episode_0001/depth/0000.png'
    semantic_path = '../imitation/imitation_dataset/episode_0001/semantic/0000.png'
    
    initial_rgb, initial_depth, initial_semantic = predictor.preprocess_images(
        rgb_path, depth_path, semantic_path
    )
    
    # 목표 위치 설정
    goal_position = [5.0, 3.0]  # [x, y] 좌표
    
    # Future k=10 steps 예측
    print("Predicting future 10 steps...")
    predicted_states = predictor.predict_future_states(
        initial_rgb=initial_rgb,
        initial_depth=initial_depth,
        initial_semantic=initial_semantic,
        goal_position=goal_position,
        k_steps=10,
        use_predicted_actions=True
    )
    
    # 결과 시각화
    predictor.visualize_predictions(predicted_states, save_path='future_prediction.png')
    
    # 결과를 파일로 저장
    predictor.save_predicted_sequence(predicted_states, 'predicted_sequence_output')
    
    print("Prediction completed!")


if __name__ == "__main__":
    main()