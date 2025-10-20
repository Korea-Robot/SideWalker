import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import wandb
from tqdm import tqdm

# ====================================
# Dataset Class (수정된 부분)
# ====================================
class NavigationDataset(Dataset):
    def __init__(self, 
                dataset_path: str, 
                episode_list: List[int], 
                sequence_length: int = 10,
                min_move_threshold: float = 0.5,
                max_static_steps: int = 30):
        """
        Args:
            dataset_path: 데이터셋 루트 경로
            episode_list: 사용할 에피소드 번호 리스트
            sequence_length: GRU를 위한 시퀀스 길이 (H-step)
            min_move_threshold: 최소 이동 거리 임계값
            max_static_steps: 정적 상태 최대 스텝 수
        """
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.min_move_threshold = min_move_threshold
        self.max_static_steps = max_static_steps
        
        # 각 이미지 타입에 맞는 Transform 정의
        self.rgb_transform = self._default_rgb_transform()
        self.depth_transform = self._default_depth_transform()
        # Semantic 이미지는 3채널이므로 RGB transform을 재사용
        self.semantic_transform = self._default_rgb_transform()
        
        self.valid_sequences = []
        self._load_valid_sequences(episode_list)
    
    def _default_rgb_transform(self):
        # 3채널 이미지 (RGB, Semantic)를 위한 Transform
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _default_depth_transform(self):
        # 1채널 이미지 (Depth)를 위한 Transform
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # 1채널 정규화
        ])
    
    def _load_valid_sequences(self, episode_list: List[int]):
        """유효한 시퀀스만 필터링하여 로드"""
        for episode_idx in episode_list:
            episode_path = os.path.join(self.dataset_path, f"episode_{episode_idx:04d}")
            
            # 데이터 로드
            trajectory_path = os.path.join(episode_path, "trajectory.json")
            if not os.path.exists(trajectory_path):
                continue

            with open(trajectory_path, 'r') as f:
                episode_data = json.load(f)
            
            if len(episode_data) < self.sequence_length:
                continue
                
            # 정적 상태 체크 및 유효한 시퀀스 추출
            valid_start_indices = self._find_valid_sequences(episode_data)
            
            for start_idx in valid_start_indices:
                self.valid_sequences.append({
                    'episode_idx': episode_idx,
                    'start_idx': start_idx,
                    'episode_path': episode_path,
                    'episode_data': episode_data
                })
    
    def _find_valid_sequences(self, episode_data: List[Dict]) -> List[int]:
        """정적 상태가 아닌 유효한 시퀀스 시작점들을 찾음"""
        valid_starts = []
        
        for i in range(len(episode_data) - self.sequence_length + 1):
            # 시퀀스 내에서 충분한 이동이 있는지 확인
            positions = [episode_data[j]['position'] for j in range(i, i + self.sequence_length)]
            
            # 연속된 정적 상태 체크
            static_count = 0
            is_valid = True
            
            for j in range(1, len(positions)):
                dist = np.linalg.norm(np.array(positions[j]) - np.array(positions[j-1]))
                if dist < self.min_move_threshold:
                    static_count += 1
                    if static_count >= self.max_static_steps:
                        is_valid = False
                        break
                else:
                    static_count = 0
            
            if is_valid:
                valid_starts.append(i)
                
        return valid_starts
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        seq_info = self.valid_sequences[idx]
        episode_path = seq_info['episode_path']
        start_idx = seq_info['start_idx']
        episode_data = seq_info['episode_data']
        
        # 시퀀스 데이터 로드
        rgb_sequence = []
        depth_sequence = []
        semantic_sequence = []
        actions = []
        rewards = []
        goals = []
        positions = []
        headings = []
        
        for i in range(start_idx, start_idx + self.sequence_length):
            # 이미지 로드
            rgb_img = Image.open(os.path.join(episode_path, "rgb", f"{i:04d}.png")).convert('RGB')
            depth_img = Image.open(os.path.join(episode_path, "depth", f"{i:04d}.png")).convert('L')
            semantic_img = Image.open(os.path.join(episode_path, "semantic", f"{i:04d}.png")).convert('RGB')
            
            # Transform 적용 (이미지 종류에 맞게)
            rgb_sequence.append(self.rgb_transform(rgb_img))
            depth_sequence.append(self.depth_transform(depth_img))
            semantic_sequence.append(self.semantic_transform(semantic_img))
            
            # 액션 및 상태 정보
            step_data = episode_data[i]
            actions.append(step_data['action'])
            rewards.append(step_data['reward'])
            goals.append(step_data['goal_position'])
            positions.append(step_data['position'])
            headings.append(step_data['heading'])
        
        # Next state (t+1)을 위한 데이터
        if start_idx + self.sequence_length < len(episode_data):
            next_rgb_path = os.path.join(episode_path, "rgb", f"{start_idx + self.sequence_length:04d}.png")
            next_depth_path = os.path.join(episode_path, "depth", f"{start_idx + self.sequence_length:04d}.png")
            next_semantic_path = os.path.join(episode_path, "semantic", f"{start_idx + self.sequence_length:04d}.png")

            next_rgb = Image.open(next_rgb_path).convert('RGB')
            next_depth = Image.open(next_depth_path).convert('L')
            next_semantic = Image.open(next_semantic_path).convert('RGB')
            
            next_rgb = self.rgb_transform(next_rgb)
            next_depth = self.depth_transform(next_depth)
            next_semantic = self.semantic_transform(next_semantic)
            
            next_step_data = episode_data[start_idx + self.sequence_length]
            next_position = next_step_data['position']
            next_heading = next_step_data['heading']
        else:
            # 에피소드 끝인 경우
            next_rgb = rgb_sequence[-1].clone()
            next_depth = depth_sequence[-1].clone()
            next_semantic = semantic_sequence[-1].clone()
            next_position = positions[-1]
            next_heading = headings[-1]
        
        return {
            'rgb_seq': torch.stack(rgb_sequence),
            'depth_seq': torch.stack(depth_sequence),
            'semantic_seq': torch.stack(semantic_sequence),
            'actions': torch.tensor(actions, dtype=torch.float32),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'goals': torch.tensor(goals, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'headings': torch.tensor(headings, dtype=torch.float32),
            'next_rgb': next_rgb,
            'next_depth': next_depth,
            'next_semantic': next_semantic,
            'next_position': torch.tensor(next_position, dtype=torch.float32),
            'next_heading': torch.tensor(next_heading, dtype=torch.float32)
        }

# ====================================
# Navigation Model (Depth + Goal → Action)
# ====================================
class NavigationModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, action_dim=2):
        super(NavigationModel, self).__init__()
        
        # Depth image encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),  # 7x7
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, hidden_size)
        )
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size)
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        
        # Action predictor
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Action range [-1, 1]
        )
        
    def forward(self, depth_seq, goal_seq, hidden=None):
        batch_size, seq_len = depth_seq.shape[:2]
        
        # Encode depth and goal for each timestep
        depth_features = []
        goal_features = []
        
        for t in range(seq_len):
            depth_feat = self.depth_encoder(depth_seq[:, t])
            goal_feat = self.goal_encoder(goal_seq[:, t])
            depth_features.append(depth_feat)
            goal_features.append(goal_feat)
        
        depth_features = torch.stack(depth_features, dim=1)
        goal_features = torch.stack(goal_features, dim=1)
        
        # Concatenate features
        combined_features = torch.cat([depth_features, goal_features], dim=-1)
        
        # GRU processing
        gru_output, hidden = self.gru(combined_features, hidden)
        
        # Predict actions for each timestep
        actions = self.action_head(gru_output)
        
        return actions, hidden

# ====================================
# World Model (State + Action → Next State)
# ====================================
class WorldModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, action_dim=2):
        super(WorldModel, self).__init__()
        
        # State encoder (RGB + Depth + Semantic)
        self.rgb_encoder = self._make_image_encoder(3, hidden_size // 3)
        self.depth_encoder = self._make_image_encoder(1, hidden_size // 3)
        self.semantic_encoder = self._make_image_encoder(3, hidden_size // 3)
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # GRU for dynamics modeling
        self.gru = nn.GRU(hidden_size + 128, hidden_size, num_layers, batch_first=True)
        
        # Decoders for next state prediction
        # The output size of decoder should match the input size of encoder
        self.rgb_decoder = self._make_image_decoder(hidden_size, 3, (224, 224))
        self.depth_decoder = self._make_image_decoder(hidden_size, 1, (224, 224))
        self.semantic_decoder = self._make_image_decoder(hidden_size, 3, (224, 224))

        # Position and heading predictors
        self.position_head = nn.Linear(hidden_size, 2)
        self.heading_head = nn.Linear(hidden_size, 1)
        
    def _make_image_encoder(self, in_channels, out_features):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), # 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_features)
        )
    
    def _make_image_decoder(self, in_features, out_channels, output_size):
        # The decoder architecture should be able to reconstruct the original image size
        # Let's adjust it to produce 224x224 images
        return nn.Sequential(
            nn.Linear(in_features, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(-1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1), # 224x224
            nn.Sigmoid() # Output pixels in [0, 1]
        )
    
    def forward(self, rgb_seq, depth_seq, semantic_seq, action_seq, hidden=None):
        batch_size, seq_len = rgb_seq.shape[:2]
        
        # Encode states and actions for each timestep
        combined_features = []
        
        for t in range(seq_len):
            rgb_feat = self.rgb_encoder(rgb_seq[:, t])
            depth_feat = self.depth_encoder(depth_seq[:, t])
            semantic_feat = self.semantic_encoder(semantic_seq[:, t])
            action_feat = self.action_encoder(action_seq[:, t])
            
            state_feat = torch.cat([rgb_feat, depth_feat, semantic_feat], dim=-1)
            combined_feat = torch.cat([state_feat, action_feat], dim=-1)
            combined_features.append(combined_feat)
        
        combined_features = torch.stack(combined_features, dim=1)
        
        # GRU processing
        gru_output, hidden = self.gru(combined_features, hidden)
        
        # Predict next states (only for the last timestep)
        last_output = gru_output[:, -1]  # Take last timestep output
        
        next_rgb = self.rgb_decoder(last_output)
        next_depth = self.depth_decoder(last_output)
        next_semantic = self.semantic_decoder(last_output)
        next_position = self.position_head(last_output)
        next_heading = self.heading_head(last_output)
        
        return {
            'next_rgb': next_rgb,
            'next_depth': next_depth,
            'next_semantic': next_semantic,
            'next_position': next_position,
            'next_heading': next_heading
        }, hidden

# ====================================
# Training Functions
# ====================================
def train_navigation_model(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Nav Training Epoch {epoch}")):
        depth_seq = batch['depth_seq'].to(device)
        goal_seq = batch['goals'].to(device)
        target_actions = batch['actions'].to(device)
        
        optimizer.zero_grad()
        
        pred_actions, _ = model(depth_seq, goal_seq)
        loss = F.mse_loss(pred_actions, target_actions)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # if batch_idx % 100 == 0:
        #     print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_world_model(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"World Training Epoch {epoch}")):
        rgb_seq = batch['rgb_seq'].to(device)
        depth_seq = batch['depth_seq'].to(device)
        semantic_seq = batch['semantic_seq'].to(device)
        action_seq = batch['actions'].to(device)
        
        # Targets
        target_rgb = batch['next_rgb'].to(device)
        target_depth = batch['next_depth'].to(device)
        target_semantic = batch['next_semantic'].to(device)
        target_position = batch['next_position'].to(device)
        target_heading = batch['next_heading'].to(device)
        
        optimizer.zero_grad()
        
        predictions, _ = model(rgb_seq, depth_seq, semantic_seq, action_seq)
        
        # Calculate losses
        # Note: Since decoders output with Sigmoid, targets should be in [0, 1] range.
        # The Normalize transform needs to be considered. For simplicity here we use MSE.
        # A better approach might be to un-normalize predictions or use perceptual loss.
        rgb_loss = F.mse_loss(predictions['next_rgb'], target_rgb)
        depth_loss = F.mse_loss(predictions['next_depth'], target_depth)
        semantic_loss = F.mse_loss(predictions['next_semantic'], target_semantic)
        position_loss = F.mse_loss(predictions['next_position'], target_position)
        heading_loss = F.mse_loss(predictions['next_heading'], target_heading)
        
        # Combined loss
        loss = rgb_loss + depth_loss + semantic_loss + position_loss * 10 + heading_loss * 10
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # if batch_idx % 100 == 0:
        #     print(f'Batch {batch_idx}, Total Loss: {loss.item():.6f}, '
        #           f'RGB: {rgb_loss.item():.6f}, Depth: {depth_loss.item():.6f}, '
        #           f'Semantic: {semantic_loss.item():.6f}, Pos: {position_loss.item():.6f}, '
        #           f'Head: {heading_loss.item():.6f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ====================================
# Main Training Script
# ====================================
def main():
    
    # Configuration
    config = {
        'dataset_path': '../world_data/imitation_dataset',
        'batch_size': 64,
        'sequence_length': 10,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'save_interval': 10,
        'min_move_threshold': 0.5,
        'max_static_steps': 30
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load train/test split
    split_file_path = os.path.join(config['dataset_path'], 'train_test_split.json')
    if not os.path.exists(split_file_path):
        print(f"Error: train_test_split.json not found at {split_file_path}")
        return
        
    with open(split_file_path, 'r') as f:
        split_data = json.load(f)
    
    train_episodes = split_data['train']
    val_episodes = split_data['validation']
    
    # Create datasets
    train_dataset = NavigationDataset(
        config['dataset_path'], 
        train_episodes, 
        config['sequence_length'],
        config['min_move_threshold'],
        config['max_static_steps']
    )
    
    val_dataset = NavigationDataset(
        config['dataset_path'], 
        val_episodes, 
        config['sequence_length'],
        config['min_move_threshold'],
        config['max_static_steps']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    nav_model = NavigationModel().to(device)
    # Corrected the decoder architecture to match 224x224 input
    world_model = WorldModel(hidden_size=512).to(device)
    
    # Initialize optimizers
    nav_optimizer = optim.Adam(nav_model.parameters(), lr=config['learning_rate'])
    world_optimizer = optim.Adam(world_model.parameters(), lr=config['learning_rate'])
    
    # Initialize wandb (optional)
    # wandb.init(project="navigation-world-model", config=config)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
        
        # Train Navigation Model
        nav_train_loss = train_navigation_model(nav_model, train_dataloader, nav_optimizer, device, epoch+1)
        
        # Train World Model
        world_train_loss = train_world_model(world_model, train_dataloader, world_optimizer, device, epoch+1)
        
        print(f"Navigation Model - Train Loss: {nav_train_loss:.6f}")
        print(f"World Model - Train Loss: {world_train_loss:.6f}")
        
        # Save models
        if (epoch + 1) % config['save_interval'] == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'nav_model_state_dict': nav_model.state_dict(),
                'nav_optimizer_state_dict': nav_optimizer.state_dict(),
                'nav_loss': nav_train_loss,
            }, f'checkpoints/navigation_model_epoch_{epoch+1}.pth')
            
            torch.save({
                'epoch': epoch,
                'world_model_state_dict': world_model.state_dict(),
                'world_optimizer_state_dict': world_optimizer.state_dict(),
                'world_loss': world_train_loss,
            }, f'checkpoints/world_model_epoch_{epoch+1}.pth')
            
            print(f"Models saved at epoch {epoch+1}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()