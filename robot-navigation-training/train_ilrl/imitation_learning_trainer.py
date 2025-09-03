#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from dataset_loader import create_dataloaders

class ImitationLearningModel(nn.Module):
    """모방학습을 위한 PlannerNet 기반 모델"""
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        
        # Depth/RGB 이미지 인코더
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, hidden_dim)
        )
        
        # Goal 인코더
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Waypoint prediction head (5 waypoints, each with x,y,z)
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * 3)  # 5 waypoints * 3 coordinates
        )
        
        # Collision prediction head
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, depth_img, local_goal):
        # 이미지 인코딩
        img_features = self.image_encoder(depth_img)
        
        # Goal 인코딩
        goal_features = self.goal_encoder(local_goal)
        
        # Feature fusion
        fused_features = self.fusion(torch.cat([img_features, goal_features], dim=1))
        
        # Predictions
        waypoints = self.waypoint_head(fused_features).view(-1, 5, 3)
        collision_prob = self.collision_head(fused_features).squeeze()
        
        return waypoints, collision_prob

class ImitationLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = ImitationLearningModel().to(self.device)
        
        # 옵티마이저와 손실함수
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.waypoint_criterion = nn.MSELoss()
        self.collision_criterion = nn.BCELoss()
        
        # 학습 기록
        self.writer = SummaryWriter(config['log_dir'])
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        waypoint_loss_sum = 0
        collision_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 데이터 준비 (첫 번째 프레임만 사용)
            depth = batch['depth'][:, 0].to(self.device)  # [B, C, H, W]
            local_goal = batch['local_goal'].to(self.device)  # [B, 3]
            
            # Ground truth (실제 명령어를 waypoint로 변환)
            commands = batch['commands'][:, 0].to(self.device)  # [B, 2]
            collision_gt = batch['collision'].to(self.device)  # [B]
            
            # 간단한 waypoint GT 생성 (실제로는 더 정교한 방법 필요)
            waypoint_gt = self.generate_waypoint_gt(commands, local_goal)
            
            # Forward pass
            waypoint_pred, collision_pred = self.model(depth, local_goal)
            
            # Loss 계산
            waypoint_loss = self.waypoint_criterion(waypoint_pred, waypoint_gt)
            collision_loss = self.collision_criterion(collision_pred, collision_gt)
            
            total_loss_batch = waypoint_loss + 0.5 * collision_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += total_loss_batch.item()
            waypoint_loss_sum += waypoint_loss.item()
            collision_loss_sum += collision_loss.item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'WP': f'{waypoint_loss.item():.4f}',
                'Col': f'{collision_loss.item():.4f}'
            })
            
            # TensorBoard 로깅
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Total_Loss', total_loss_batch.item(), global_step)
                self.writer.add_scalar('Train/Waypoint_Loss', waypoint_loss.item(), global_step)
                self.writer.add_scalar('Train/Collision_Loss', collision_loss.item(), global_step)
        
        avg_loss = total_loss / len(train_loader)
        avg_waypoint_loss = waypoint_loss_sum / len(train_loader)
        avg_collision_loss = collision_loss_sum / len(train_loader)
        
        return avg_loss, avg_waypoint_loss, avg_collision_loss
    
    def generate_waypoint_gt(self, commands, local_goal):
        """명령어와 local goal로부터 waypoint GT 생성"""
        batch_size = commands.shape[0]
        waypoints = torch.zeros(batch_size, 5, 3).to(self.device)
        
        for i in range(batch_size):
            linear_x, angular_z = commands[i]
            goal_x, goal_y = local_goal[i][:2]
            
            # 간단한 waypoint 생성 (실제로는 더 정교한 방법 필요)
            for j in range(5):
                t = (j + 1) * 0.2  # 0.2초 간격
                
                if abs(angular_z) < 0.1:  # 직진
                    x = linear_x * t
                    y = 0
                else:  # 회전
                    radius = linear_x / angular_z if angular_z != 0 else 1.0
                    angle = angular_z * t
                    x = radius * torch.sin(angle)
                    y = radius * (1 - torch.cos(angle))
                
                waypoints[i, j] = torch.tensor([x, y, 0])
        
        return waypoints
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                depth = batch['depth'][:, 0].to(self.device)
                local_goal = batch['local_goal'].to(self.device)
                commands = batch['commands'][:, 0].to(self.device)
                collision_gt = batch['collision'].to(self.device)
                
                waypoint_gt = self.generate_waypoint_gt(commands, local_goal)
                waypoint_pred, collision_pred = self.model(depth, local_goal)
                
                waypoint_loss = self.waypoint_criterion(waypoint_pred, waypoint_gt)
                collision_loss = self.collision_criterion(collision_pred, collision_gt)
                
                total_loss += waypoint_loss.item() + 0.5 * collision_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = Path(self.config['checkpoint_dir']) / f'imitation_model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = Path(self.config['checkpoint_dir']) / 'best_imitation_model.pt'
            torch.save(checkpoint, best_path)
    
    def train(self, train_loader, val_loader):
        print(f"Starting imitation learning training on {self.device}")
        
        for epoch in range(self.config['num_epochs']):
            # 학습
            train_loss, wp_loss, col_loss = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss = self.validate(val_loader, epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 체크포인트 저장
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss)
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to dataset')
    parser.add_argument('--config', default='config/imitation_config.json', help='Config file')
    args = parser.parse_args()
    
    # 설정 로드
    config = {
        'learning_rate': 1e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'checkpoint_dir': './checkpoints/imitation',
        'log_dir': './logs/imitation',
        'save_interval': 10
    }
    
    # 디렉토리 생성
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # 데이터로더 생성
    waypoints = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
    train_loader, val_loader = create_dataloaders(
        args.dataset_path,
        batch_size=config['batch_size'],
        dataset_type='imitation',
        waypoints=waypoints
    )
    
    # 트레이너 생성 및 학습
    trainer = ImitationLearningTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
