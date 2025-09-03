#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import deque
import random

from dataset_loader import create_dataloaders
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class ReinforcementLearningModel(nn.Module):
    """강화학습을 위한 모델 - RGB를 Segformer로 처리하고 discrete action 출력"""
    def __init__(self, action_dim=7, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        
        # Segformer for RGB processing
        self.segformer_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        
        # Freeze segformer parameters
        for param in self.segformer.parameters():
            param.requires_grad = False
        
        # Depth image encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512)
        )
        
        # Segmentation feature encoder
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(150, 128, 3, padding=1),  # ADE20K has 150 classes
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512)
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=1024,  # depth + seg features
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Action heads for discrete actions
        self.linear_action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 linear actions: stop, slow, fast
        )
        
        self.angular_action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 angular actions: -1.0 to 1.0
        )
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def process_rgb_with_segformer(self, rgb_batch):
        """RGB 이미지를 Segformer로 처리하여 segmentation map 생성"""
        batch_size, seq_len, c, h, w = rgb_batch.shape
        rgb_batch = rgb_batch.view(-1, c, h, w)  # [B*T, C, H, W]
        
        # Convert to PIL format for segformer
        rgb_list = []
        for i in range(rgb_batch.shape[0]):
            rgb_np = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
            rgb_np = (rgb_np * 255).astype(np.uint8)
            rgb_list.append(rgb_np)
        
        # Process with segformer
        inputs = self.segformer_processor(rgb_list, return_tensors="pt")
        inputs = {k: v.to(rgb_batch.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.segformer(**inputs)
            segmentation_maps = outputs.logits  # [B*T, num_classes, H, W]
        
        # Reshape back to sequence format
        segmentation_maps = segmentation_maps.view(batch_size, seq_len, *segmentation_maps.shape[1:])
        return segmentation_maps
    
    def forward(self, depth_sequence, rgb_sequence):
        batch_size, seq_len = depth_sequence.shape[:2]
        
        # Process RGB with Segformer
        seg_maps = self.process_rgb_with_segformer(rgb_sequence)
        
        # Process each frame in the sequence
        sequence_features = []
        
        for t in range(seq_len):
            # Depth features
            depth_frame = depth_sequence[:, t]  # [B, C, H, W]
            depth_features = self.depth_encoder(depth_frame)
            
            # Segmentation features
            seg_frame = seg_maps[:, t]  # [B, num_classes, H, W]
            seg_features = self.seg_encoder(seg_frame)
            
            # Combine features
            combined_features = torch.cat([depth_features, seg_features], dim=1)
            sequence_features.append(combined_features)
        
        # Stack sequence features
        sequence_features = torch.stack(sequence_features, dim=1)  # [B, T, 1024]
        
        # LSTM processing
        lstm_out, _ = self.lstm(sequence_features)  # [B, T, 256]
        
        # Use last timestep for action prediction
        final_features = lstm_out[:, -1]  # [B, 256]
        
        # Action predictions
        linear_logits = self.linear_action_head(final_features)
        angular_logits = self.angular_action_head(final_features)
        
        # Value prediction
        value = self.value_head(final_features)
        
        return linear_logits, angular_logits, value

class PPOTrainer:
    """PPO 알고리즘을 사용한 강화학습 트레이너"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = ReinforcementLearningModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # PPO 하이퍼파라미터
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 학습 기록
        self.writer = SummaryWriter(config['log_dir'])
        
        # Experience buffer
        self.buffer = deque(maxlen=config.get('buffer_size', 10000))
    
    def collect_experience(self, dataloader):
        """데이터로더에서 경험 수집"""
        self.model.eval()
        experiences = []
        
        with torch.no_grad():
            for batch in dataloader:
                depth_seq = batch['depth'].to(self.device)
                rgb_seq = batch['rgb'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                discrete_actions = batch['discrete_actions'].to(self.device)
                
                # 모델 예측
                linear_logits, angular_logits, values = self.model(depth_seq, rgb_seq)
                
                # Action probabilities
                linear_probs = F.softmax(linear_logits, dim=-1)
                angular_probs = F.softmax(angular_logits, dim=-1)
                
                # 실제 행동의 확률 계산
                batch_size, seq_len = discrete_actions.shape[:2]
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        linear_action = discrete_actions[b, t, 0].item()
                        angular_action = discrete_actions[b, t, 1].item()
                        
                        linear_prob = linear_probs[b, linear_action].item()
                        angular_prob = angular_probs[b, angular_action].item()
                        
                        experience = {
                            'depth': depth_seq[b, t].cpu(),
                            'rgb': rgb_seq[b, t].cpu(),
                            'linear_action': linear_action,
                            'angular_action': angular_action,
                            'linear_prob': linear_prob,
                            'angular_prob': angular_prob,
                            'reward': rewards[b, t].item(),
                            'value': values[b].item()
                        }
                        experiences.append(experience)
        
        return experiences
    
    def compute_advantages(self, experiences, gamma=0.99, lam=0.95):
        """GAE를 사용한 advantage 계산"""
        rewards = [exp['reward'] for exp in experiences]
        values = [exp['value'] for exp in experiences]
        
        advantages = []
        returns = []
        
        # Compute returns and advantages
        gae = 0
        for i in reversed(range(len(experiences))):
            if i == len(experiences) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def train_step(self, experiences):
        """PPO 학습 스텝"""
        self.model.train()
        
        # 배치 준비
        depth_batch = torch.stack([exp['depth'] for exp in experiences]).unsqueeze(1).to(self.device)
        rgb_batch = torch.stack([exp['rgb'] for exp in experiences]).unsqueeze(1).to(self.device)
        
        linear_actions = torch.tensor([exp['linear_action'] for exp in experiences]).to(self.device)
        angular_actions = torch.tensor([exp['angular_action'] for exp in experiences]).to(self.device)
        
        old_linear_probs = torch.tensor([exp['linear_prob'] for exp in experiences]).to(self.device)
        old_angular_probs = torch.tensor([exp['angular_prob'] for exp in experiences]).to(self.device)
        
        advantages, returns = self.compute_advantages(experiences)
        advantages = torch.tensor(advantages).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        linear_logits, angular_logits, values = self.model(depth_batch, rgb_batch)
        
        # New probabilities
        linear_probs = F.softmax(linear_logits, dim=-1)
        angular_probs = F.softmax(angular_logits, dim=-1)
        
        new_linear_probs = linear_probs.gather(1, linear_actions.unsqueeze(1)).squeeze()
        new_angular_probs = angular_probs.gather(1, angular_actions.unsqueeze(1)).squeeze()
        
        # PPO loss calculation
        linear_ratio = new_linear_probs / (old_linear_probs + 1e-8)
        angular_ratio = new_angular_probs / (old_angular_probs + 1e-8)
        
        # Combined ratio (geometric mean)
        ratio = torch.sqrt(linear_ratio * angular_ratio)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss
        linear_entropy = -(linear_probs * torch.log(linear_probs + 1e-8)).sum(dim=-1).mean()
        angular_entropy = -(angular_probs * torch.log(angular_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -(linear_entropy + angular_entropy)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, dataloader):
        print(f"Starting reinforcement learning training on {self.device}")
        
        for epoch in range(self.config['num_epochs']):
            # 경험 수집
            experiences = self.collect_experience(dataloader)
            
            # 배치 단위로 학습
            batch_size = self.config['batch_size']
            random.shuffle(experiences)
            
            epoch_losses = []
            for i in range(0, len(experiences), batch_size):
                batch_exp = experiences[i:i+batch_size]
                if len(batch_exp) < batch_size:
                    continue
                
                losses = self.train_step(batch_exp)
                epoch_losses.append(losses)
            
            # 평균 손실 계산
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # 로깅
            for key, value in avg_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            print(f"Epoch {epoch}: {avg_losses}")
            
            # 체크포인트 저장
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = Path(self.config['checkpoint_dir']) / f'rl_model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to dataset')
    args = parser.parse_args()
    
    # 설정
    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 200,
        'checkpoint_dir': './checkpoints/reinforcement',
        'log_dir': './logs/reinforcement',
        'save_interval': 20,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'buffer_size': 10000
    }
    
    # 디렉토리 생성
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # 데이터로더 생성
    train_loader, _ = create_dataloaders(
        args.dataset_path,
        batch_size=config['batch_size'],
        dataset_type='reinforcement'
    )
    
    # 트레이너 생성 및 학습
    trainer = PPOTrainer(config)
    trainer.train(train_loader)

if __name__ == '__main__':
    main()
