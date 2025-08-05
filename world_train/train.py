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
# Dataset Class
# ====================================
class NavigationDataset(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 episode_list: List[int], 
                 sequence_length: int = 10,
                 min_move_threshold: float = 0.5,
                 max_static_steps: int = 30):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.min_move_threshold = min_move_threshold
        self.max_static_steps = max_static_steps
        
        self.rgb_transform = self._default_rgb_transform()
        self.depth_transform = self._default_depth_transform()
        self.semantic_transform = self._default_rgb_transform()
        
        self.valid_sequences = []
        self._load_valid_sequences(episode_list)
    
    def _default_rgb_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _default_depth_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _load_valid_sequences(self, episode_list: List[int]):
        for episode_idx in tqdm(episode_list, desc="Loading dataset"):
            episode_path = os.path.join(self.dataset_path, f"episode_{episode_idx:04d}")
            
            trajectory_path = os.path.join(episode_path, "trajectory.json")
            if not os.path.exists(trajectory_path):
                continue

            with open(trajectory_path, 'r') as f:
                episode_data = json.load(f)
            
            if len(episode_data) < self.sequence_length:
                continue
                
            valid_start_indices = self._find_valid_sequences(episode_data)
            
            for start_idx in valid_start_indices:
                self.valid_sequences.append({
                    'episode_idx': episode_idx,
                    'start_idx': start_idx,
                    'episode_path': episode_path,
                    'episode_data': episode_data
                })
    
    def _find_valid_sequences(self, episode_data: List[Dict]) -> List[int]:
        valid_starts = []
        for i in range(len(episode_data) - self.sequence_length + 1):
            positions = [episode_data[j]['position'] for j in range(i, i + self.sequence_length)]
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
        
        rgb_sequence, depth_sequence, semantic_sequence = [], [], []
        actions, rewards, goals, positions, headings = [], [], [], [], []
        
        for i in range(start_idx, start_idx + self.sequence_length):
            rgb_img = Image.open(os.path.join(episode_path, "rgb", f"{i:04d}.png")).convert('RGB')
            depth_img = Image.open(os.path.join(episode_path, "depth", f"{i:04d}.png")).convert('L')
            semantic_img = Image.open(os.path.join(episode_path, "semantic", f"{i:04d}.png")).convert('RGB')
            
            rgb_sequence.append(self.rgb_transform(rgb_img))
            depth_sequence.append(self.depth_transform(depth_img))
            semantic_sequence.append(self.semantic_transform(semantic_img))
            
            step_data = episode_data[i]
            actions.append(step_data['action'])
            rewards.append(step_data['reward'])
            goals.append(step_data['goal_position'])
            positions.append(step_data['position'])
            headings.append(step_data['heading'])
        
        if start_idx + self.sequence_length < len(episode_data):
            next_step_idx = start_idx + self.sequence_length
            next_rgb = self.rgb_transform(Image.open(os.path.join(episode_path, "rgb", f"{next_step_idx:04d}.png")).convert('RGB'))
            next_depth = self.depth_transform(Image.open(os.path.join(episode_path, "depth", f"{next_step_idx:04d}.png")).convert('L'))
            next_semantic = self.semantic_transform(Image.open(os.path.join(episode_path, "semantic", f"{next_step_idx:04d}.png")).convert('RGB'))
            
            next_step_data = episode_data[next_step_idx]
            next_position = next_step_data['position']
            next_heading = next_step_data['heading']
        else:
            next_rgb = rgb_sequence[-1].clone()
            next_depth = depth_sequence[-1].clone()
            next_semantic = semantic_sequence[-1].clone()
            next_position = positions[-1]
            next_heading = headings[-1]
        
        return {
            'rgb_seq': torch.stack(rgb_sequence), 'depth_seq': torch.stack(depth_sequence), 
            'semantic_seq': torch.stack(semantic_sequence), 'actions': torch.tensor(actions, dtype=torch.float32), 
            'rewards': torch.tensor(rewards, dtype=torch.float32), 'goals': torch.tensor(goals, dtype=torch.float32), 
            'positions': torch.tensor(positions, dtype=torch.float32), 'headings': torch.tensor(headings, dtype=torch.float32),
            'next_rgb': next_rgb, 'next_depth': next_depth, 'next_semantic': next_semantic,
            'next_position': torch.tensor(next_position, dtype=torch.float32), 'next_heading': torch.tensor(next_heading, dtype=torch.float32)
        }

# ====================================
# Navigation Model (Depth + Goal → Action)
# ====================================
class NavigationModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, action_dim=2):
        super(NavigationModel, self).__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten(),
            nn.Linear(256 * 7 * 7, hidden_size)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, hidden_size)
        )
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        
    def forward(self, depth_seq, goal_seq, hidden=None):
        batch_size, seq_len = depth_seq.shape[:2]
        depth_features, goal_features = [], []
        for t in range(seq_len):
            depth_features.append(self.depth_encoder(depth_seq[:, t]))
            goal_features.append(self.goal_encoder(goal_seq[:, t]))
        
        depth_features = torch.stack(depth_features, dim=1)
        goal_features = torch.stack(goal_features, dim=1)
        
        combined_features = torch.cat([depth_features, goal_features], dim=-1)
        gru_output, hidden = self.gru(combined_features, hidden)
        return self.action_head(gru_output), hidden

# ====================================
# World Model (State + Action → Next State)
# ====================================
class WorldModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, action_dim=2):
        super(WorldModel, self).__init__()
        
        # ====================================================================
        # 수정된 부분: hidden_size를 3으로 나눌 때 나머지를 분배하여 합이 정확히 hidden_size가 되도록 수정
        # ====================================================================
        base_feat_size = hidden_size // 3  # 512 // 3 = 170
        remainder = hidden_size % 3      # 512 % 3 = 2
        
        # [170, 170, 170] 리스트를 만들고 남는 2를 앞의 두 요소에 더해줌 -> [171, 171, 170]
        feat_sizes = [base_feat_size] * 3
        for i in range(remainder):
            feat_sizes[i] += 1
        
        # State encoder (RGB + Depth + Semantic)
        self.rgb_encoder = self._make_image_encoder(3, feat_sizes[0])      # 출력 크기: 171
        self.depth_encoder = self._make_image_encoder(1, feat_sizes[1])    # 출력 크기: 171
        self.semantic_encoder = self._make_image_encoder(3, feat_sizes[2]) # 출력 크기: 170
        # 총합: 171 + 171 + 170 = 512 (hidden_size와 일치)
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64), nn.ReLU(), nn.Linear(64, 128)
        )
        
        # GRU for dynamics modeling (입력 크기: 512 + 128 = 640)
        self.gru = nn.GRU(hidden_size + 128, hidden_size, num_layers, batch_first=True)
        
        # Decoders for next state prediction
        self.rgb_decoder = self._make_image_decoder(hidden_size, 3)
        self.depth_decoder = self._make_image_decoder(hidden_size, 1)
        self.semantic_decoder = self._make_image_decoder(hidden_size, 3)
        self.position_head = nn.Linear(hidden_size, 2)
        self.heading_head = nn.Linear(hidden_size, 1)
        
    def _make_image_encoder(self, in_channels, out_features):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_features)
        )
    
    def _make_image_decoder(self, in_features, out_channels):
        return nn.Sequential(
            nn.Linear(in_features, 256 * 7 * 7), nn.ReLU(),
            nn.Unflatten(-1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1), nn.Sigmoid()
        )
    
    def forward(self, rgb_seq, depth_seq, semantic_seq, action_seq, hidden=None):
        combined_features = []
        for t in range(rgb_seq.shape[1]):
            rgb_feat = self.rgb_encoder(rgb_seq[:, t])
            depth_feat = self.depth_encoder(depth_seq[:, t])
            semantic_feat = self.semantic_encoder(semantic_seq[:, t])
            action_feat = self.action_encoder(action_seq[:, t])
            
            state_feat = torch.cat([rgb_feat, depth_feat, semantic_feat], dim=-1)
            combined_feat = torch.cat([state_feat, action_feat], dim=-1)
            combined_features.append(combined_feat)
        
        combined_features = torch.stack(combined_features, dim=1)
        
        gru_output, hidden = self.gru(combined_features, hidden)
        
        last_output = gru_output[:, -1]
        
        return {
            'next_rgb': self.rgb_decoder(last_output),
            'next_depth': self.depth_decoder(last_output),
            'next_semantic': self.semantic_decoder(last_output),
            'next_position': self.position_head(last_output),
            'next_heading': self.heading_head(last_output)
        }, hidden

# ====================================
# Training Functions
# ====================================
def train_navigation_model(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Nav Training Epoch {epoch}"):
        depth_seq, goal_seq, target_actions = batch['depth_seq'].to(device), batch['goals'].to(device), batch['actions'].to(device)
        optimizer.zero_grad()
        pred_actions, _ = model(depth_seq, goal_seq)
        loss = F.mse_loss(pred_actions, target_actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_world_model(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"World Training Epoch {epoch}"):
        rgb_seq, depth_seq, semantic_seq, action_seq = batch['rgb_seq'].to(device), batch['depth_seq'].to(device), batch['semantic_seq'].to(device), batch['actions'].to(device)
        targets = {k: batch[k].to(device) for k in ['next_rgb', 'next_depth', 'next_semantic', 'next_position', 'next_heading']}
        
        optimizer.zero_grad()
        predictions, _ = model(rgb_seq, depth_seq, semantic_seq, action_seq)
        
        rgb_loss = F.mse_loss(predictions['next_rgb'], targets['next_rgb'])
        depth_loss = F.mse_loss(predictions['next_depth'], targets['next_depth'])
        semantic_loss = F.mse_loss(predictions['next_semantic'], targets['next_semantic'])
        position_loss = F.mse_loss(predictions['next_position'], targets['next_position'])
        heading_loss = F.mse_loss(predictions['next_heading'], targets['next_heading'].unsqueeze(-1)) # Match shape
        
        loss = rgb_loss + depth_loss + semantic_loss + position_loss * 10 + heading_loss * 10
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ====================================
# Main Training Script
# ====================================
def main():
    config = {
        'dataset_path': '../world_data/imitation_dataset', 'batch_size': 64, 'sequence_length': 10,
        'learning_rate': 1e-4, 'num_epochs': 100, 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'save_interval': 10, 'min_move_threshold': 0.01, 'max_static_steps': 30
    }
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    with open(os.path.join(config['dataset_path'], 'train_test_split.json'), 'r') as f:
        split_data = json.load(f)
    
    train_dataset = NavigationDataset(config['dataset_path'], split_data['train'], **{k: v for k, v in config.items() if k in ['sequence_length', 'min_move_threshold', 'max_static_steps']})
    val_dataset = NavigationDataset(config['dataset_path'], split_data['validation'], **{k: v for k, v in config.items() if k in ['sequence_length', 'min_move_threshold', 'max_static_steps']})
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    nav_model = NavigationModel().to(device)
    world_model = WorldModel(hidden_size=512).to(device)
    
    nav_optimizer = optim.Adam(nav_model.parameters(), lr=config['learning_rate'])
    world_optimizer = optim.Adam(world_model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
        nav_loss = train_navigation_model(nav_model, train_loader, nav_optimizer, device, epoch+1)
        world_loss = train_world_model(world_model, train_loader, world_optimizer, device, epoch+1)
        print(f"Nav Model Train Loss: {nav_loss:.6f}, World Model Train Loss: {world_loss:.6f}")
        
        if (epoch + 1) % config['save_interval'] == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({'epoch': epoch, 'nav_model_state_dict': nav_model.state_dict()}, f'checkpoints/nav_model_epoch_{epoch+1}.pth')
            torch.save({'epoch': epoch, 'world_model_state_dict': world_model.state_dict()}, f'checkpoints/world_model_epoch_{epoch+1}.pth')
            print(f"Models saved at epoch {epoch+1}")
            
    print("Training completed!")

if __name__ == "__main__":
    main()