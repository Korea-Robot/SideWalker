#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

class NavigationDataset(Dataset):
    def __init__(self, dataset_path, transform=None, sequence_length=5):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.sequence_length = sequence_length
        
        # 메타데이터 로드
        metadata_path = self.dataset_path / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.depth_paths = self.metadata['depth_images']
        self.rgb_paths = self.metadata['rgb_images']
        self.commands = np.array(self.metadata['commands'])
        self.odometry = np.array(self.metadata['odometry'])
        
        # 시퀀스 데이터를 위한 유효한 인덱스 계산
        self.valid_indices = list(range(len(self.commands) - sequence_length + 1))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # 시퀀스 데이터 로드
        depth_sequence = []
        rgb_sequence = []
        
        for i in range(start_idx, end_idx):
            # Depth 이미지 로드
            depth = np.load(self.depth_paths[i])
            if self.transform:
                depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
                depth = self.transform(depth_pil)
            else:
                depth = torch.from_numpy(depth).unsqueeze(0).float()
            depth_sequence.append(depth)
            
            # RGB 이미지 로드
            rgb = cv2.imread(self.rgb_paths[i])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if self.transform:
                rgb_pil = Image.fromarray(rgb)
                rgb = self.transform(rgb_pil)
            else:
                rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            rgb_sequence.append(rgb)
        
        # 시퀀스를 텐서로 변환
        depth_sequence = torch.stack(depth_sequence)
        rgb_sequence = torch.stack(rgb_sequence)
        
        # 명령어와 odometry 시퀀스
        cmd_sequence = torch.from_numpy(self.commands[start_idx:end_idx]).float()
        odom_sequence = torch.from_numpy(self.odometry[start_idx:end_idx]).float()
        
        return {
            'depth': depth_sequence,
            'rgb': rgb_sequence,
            'commands': cmd_sequence,
            'odometry': odom_sequence,
            'start_idx': start_idx
        }

class ImitationLearningDataset(NavigationDataset):
    """모방학습용 데이터셋"""
    def __init__(self, dataset_path, waypoints, transform=None, sequence_length=5):
        super().__init__(dataset_path, transform, sequence_length)
        self.waypoints = waypoints
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # 현재 위치에서 목표 waypoint까지의 local goal 계산
        start_idx = data['start_idx']
        current_odom = self.odometry[start_idx]  # [x, y, yaw]
        
        # 가장 가까운 waypoint 찾기
        current_pos = current_odom[:2]
        distances = [np.linalg.norm(np.array(wp) - current_pos) for wp in self.waypoints]
        target_wp_idx = np.argmin(distances)
        target_wp = self.waypoints[target_wp_idx]
        
        # Global to local coordinate transformation
        dx_global = target_wp[0] - current_odom[0]
        dy_global = target_wp[1] - current_odom[1]
        yaw = current_odom[2]
        
        local_x = dx_global * np.cos(yaw) + dy_global * np.sin(yaw)
        local_y = -dx_global * np.sin(yaw) + dy_global * np.cos(yaw)
        
        data['local_goal'] = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32)
        
        # Collision label 생성 (간단한 휴리스틱)
        # 명령어가 0에 가까우면 collision 상황으로 간주
        cmd_magnitude = np.linalg.norm(data['commands'][0].numpy())
        collision_label = 1.0 if cmd_magnitude < 0.1 else 0.0
        data['collision'] = torch.tensor(collision_label, dtype=torch.float32)
        
        return data

class ReinforcementLearningDataset(NavigationDataset):
    """강화학습용 데이터셋"""
    def __init__(self, dataset_path, transform=None, sequence_length=5, action_bins=7):
        super().__init__(dataset_path, transform, sequence_length)
        self.action_bins = action_bins
        
        # Discrete action space 정의
        self.linear_actions = [0.0, 0.2, 0.5]  # 정지, 느림, 빠름
        self.angular_actions = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]  # 좌회전 ~ 우회전
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # 연속 action을 discrete action으로 변환
        discrete_actions = []
        for i in range(self.sequence_length):
            cmd = data['commands'][i].numpy()
            linear_x, angular_z = cmd[0], cmd[1]
            
            # Linear action discretization
            linear_discrete = np.argmin([abs(linear_x - a) for a in self.linear_actions])
            
            # Angular action discretization  
            angular_discrete = np.argmin([abs(angular_z - a) for a in self.angular_actions])
            
            discrete_actions.append([linear_discrete, angular_discrete])
        
        data['discrete_actions'] = torch.tensor(discrete_actions, dtype=torch.long)
        
        # Reward 계산 (간단한 휴리스틱)
        # 전진하고 장애물을 피하면 높은 보상
        rewards = []
        for i in range(self.sequence_length):
            cmd = data['commands'][i].numpy()
            linear_x = cmd[0]
            
            # 전진 보상
            forward_reward = linear_x * 1.0
            
            # 충돌 페널티 (depth 이미지 기반)
            depth_img = data['depth'][i].numpy()
            if depth_img.ndim == 3:
                depth_img = depth_img[0]  # 첫 번째 채널만 사용
            
            # 전방 중앙 영역의 최소 거리 확인
            h, w = depth_img.shape
            center_region = depth_img[h//3:2*h//3, w//3:2*w//3]
            min_distance = np.min(center_region[center_region > 0])
            
            collision_penalty = -5.0 if min_distance < 0.5 else 0.0
            
            reward = forward_reward + collision_penalty
            rewards.append(reward)
        
        data['rewards'] = torch.tensor(rewards, dtype=torch.float32)
        
        return data

def create_dataloaders(dataset_path, batch_size=16, train_split=0.8, 
                      dataset_type='imitation', waypoints=None):
    """데이터로더 생성"""
    
    # Transform 정의
    transform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor()
    ])
    
    # 데이터셋 생성
    if dataset_type == 'imitation':
        if waypoints is None:
            raise ValueError("Waypoints required for imitation learning dataset")
        dataset = ImitationLearningDataset(dataset_path, waypoints, transform)
    elif dataset_type == 'reinforcement':
        dataset = ReinforcementLearningDataset(dataset_path, transform)
    else:
        dataset = NavigationDataset(dataset_path, transform)
    
    # Train/Test 분할
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    # 테스트 코드
    dataset_path = "./dataset"
    waypoints = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
    
    train_loader, test_loader = create_dataloaders(
        dataset_path, 
        dataset_type='imitation',
        waypoints=waypoints
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Depth shape:", batch['depth'].shape)
        print("RGB shape:", batch['rgb'].shape)
        print("Commands shape:", batch['commands'].shape)
        break
