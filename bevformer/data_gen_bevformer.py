import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import cv2
import os
import pickle
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import threading
from queue import Queue, Empty
from PIL import Image

# MetaUrban imports (기존 코드에서 가져옴)
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

mp.set_start_method("spawn", force=True)

SENSOR_SIZE = (256, 160)
BEV_SIZE = (100, 100)
EMBED_DIM = 256

# MetaUrban environment configuration
BASE_ENV_CFG = dict(
    use_render=False,
    map='X',
    manual_control=False,
    crswalk_density=1,
    object_density=0.1,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=300,
    vehicle_config=dict(enable_reverse=True, image_source="rgb_camera"),
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    relax_out_of_road_done=True,
    max_lateral_dist=5.0,
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    image_on_cuda=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# ==================== BEVFormer Model Components ====================

class PositionalEncoding(nn.Module):
    def __init__(self, num_feats, temperature=10000):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        B, N, C = query.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class BEVFormerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, bev_query, img_features, pos_embed=None):
        q1 = self.norm1(bev_query)
        bev_query = bev_query + self.self_attn(q1, q1, q1)
        
        q2 = self.norm2(bev_query)
        bev_query = bev_query + self.cross_attn(q2, img_features, img_features)
        
        bev_query = bev_query + self.ffn(self.norm3(bev_query))
        
        return bev_query

class ImageEncoder(nn.Module):
    """MetaUrban의 RGB+Depth+Semantic을 처리하는 인코더"""
    def __init__(self, input_channels=6, embed_dim=256):  # RGB(3) + Depth(1) + Semantic(1) = 5, 하지만 코드상 6으로 설정
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        return features

class MetaUrbanBEVFormer(nn.Module):
    """MetaUrban용 BEVFormer"""
    def __init__(self, 
                 img_size=SENSOR_SIZE,
                 bev_size=BEV_SIZE,
                 embed_dim=EMBED_DIM,
                 num_layers=6,
                 num_heads=8,
                 num_classes=20,  # MetaUrban semantic classes
                 dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.bev_size = bev_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Image encoder (RGB + Depth + Semantic)
        self.img_encoder = ImageEncoder(input_channels=6, embed_dim=embed_dim)
        
        # Goal vector encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2)
        )
        
        # BEV queries
        self.bev_queries = nn.Parameter(torch.randn(bev_size[0] * bev_size[1], embed_dim))
        
        # BEVFormer layers
        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, images, goal_vec):
        B = images.shape[0]
        
        # Encode image features
        img_features = self.img_encoder(images)
        
        # Encode goal vector and broadcast to image features
        goal_features = self.goal_encoder(goal_vec)  # [B, embed_dim//2]
        goal_features = goal_features.unsqueeze(1).expand(-1, img_features.shape[1], -1)  # [B, N, embed_dim//2]
        
        # Concatenate image and goal features
        img_features = torch.cat([img_features, goal_features], dim=-1)  # [B, N, embed_dim + embed_dim//2]
        img_features = F.linear(img_features, torch.randn(self.embed_dim, img_features.shape[-1]).to(img_features.device))  # Project back to embed_dim
        
        # Expand BEV queries for batch
        bev_queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Apply BEVFormer layers
        for layer in self.layers:
            bev_queries = layer(bev_queries, img_features)
        
        # Generate segmentation map
        seg_logits = self.seg_head(bev_queries)
        seg_logits = seg_logits.reshape(B, self.bev_size[0], self.bev_size[1], self.num_classes)
        seg_logits = seg_logits.permute(0, 3, 1, 2)
        
        return seg_logits

# ==================== Data Generation and Processing ====================

def semantic_to_bev_map(semantic_img, depth_img, vehicle_pos, vehicle_heading):
    """Semantic 이미지를 BEV segmentation map으로 변환"""
    H, W = semantic_img.shape
    bev_map = np.zeros(BEV_SIZE, dtype=np.uint8)
    
    # 간단한 IPM (Inverse Perspective Mapping) 변환
    # 실제로는 더 정교한 카메라 calibration과 IPM이 필요
    for y in range(H//2, H):  # 하단 절반만 처리 (도로 부분)
        for x in range(W):
            # 픽셀을 BEV 좌표로 변환 (간단한 근사)
            world_x = (x - W//2) * 0.1  # 스케일 조정
            world_y = (H - y) * 0.2
            
            # Vehicle 좌표계에서 BEV 좌표계로 변환
            bev_x = int(world_x + BEV_SIZE[1]//2)
            bev_y = int(world_y + BEV_SIZE[0]//2)
            
            if 0 <= bev_x < BEV_SIZE[1] and 0 <= bev_y < BEV_SIZE[0]:
                bev_map[bev_y, bev_x] = semantic_img[y, x]
    
    return bev_map

def preprocess_metaurban_observation(obs):
    """MetaUrban observation을 BEVFormer 입력으로 변환"""
    # RGB 이미지
    rgb = obs["image"]  # [H, W, 3]
    
    # Depth 이미지 (3채널로 복제)
    depth = obs["depth"][..., -1:] if obs["depth"].ndim == 3 else obs["depth"][..., None]
    depth = np.repeat(depth, 3, axis=-1)
    
    # Semantic 이미지를 BEV로 변환
    semantic = obs["semantic"][..., -1] if obs["semantic"].ndim == 3 else obs["semantic"]
    
    # 이미지 결합 (RGB + Depth)
    combined_img = np.concatenate([rgb, depth], axis=-1)  # [H, W, 6]
    combined_img = combined_img.astype(np.float32) / 255.0
    combined_img = np.transpose(combined_img, (2, 0, 1))  # [6, H, W]
    
    # Goal vector
    goal_vec = obs["goal_vec"].astype(np.float32)
    
    # BEV ground truth 생성 (semantic에서)
    vehicle_pos = [0, 0]  # 현재 차량 위치 (원점으로 가정)
    vehicle_heading = 0   # 현재 차량 방향
    bev_gt = semantic_to_bev_map(semantic, obs["depth"][..., -1], vehicle_pos, vehicle_heading)
    
    return {
        'images': torch.tensor(combined_img, dtype=torch.float32),
        'goal_vec': torch.tensor(goal_vec, dtype=torch.float32),
        'bev_gt': torch.tensor(bev_gt, dtype=torch.long)
    }

class MetaUrbanBEVDataset(Dataset):
    """MetaUrban 데이터를 위한 Dataset"""
    def __init__(self, data_buffer, max_size=10000):
        self.data_buffer = data_buffer
        self.max_size = max_size
        
    def __len__(self):
        return min(len(self.data_buffer), self.max_size)
    
    def __getitem__(self, idx):
        data = self.data_buffer[idx % len(self.data_buffer)]
        return data['images'], data['goal_vec'], data['bev_gt']

def compute_reward(obs, action, next_obs, done, info):
    """보상 함수 (기존 코드 그대로)"""
    reward = 0.0
    goal_vec = next_obs["goal_vec"]
    goal_distance = np.linalg.norm(goal_vec)
    reward += max(0, 1.0 - goal_distance)
    speed = info.get('speed', 0)
    reward += min(speed / 10.0, 1.0)
    if info.get('crash', False):
        reward -= 10.0
    if info.get('out_of_road', False):
        reward -= 5.0
    if info.get('arrive_dest', False):
        reward += 20.0
    reward -= 0.01
    return reward

def env_worker(worker_id: int, data_queue: mp.Queue, is_training: bool = True):
    """환경 워커 (BEV 데이터 생성 추가)"""
    print(f"Worker {worker_id} started")
    env_config = BASE_ENV_CFG.copy()
    env_config.update({
        'num_scenarios': 1000 if is_training else 200,
        'start_seed': 1000 + worker_id if is_training else worker_id,
        'training': is_training,
        'seed': 1000 + worker_id if is_training else worker_id
    })
    
    env = SidewalkStaticMetaUrbanEnv(env_config)
    
    try:
        obs, _ = env.reset()
        nav = env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        
        processed_obs = preprocess_metaurban_observation(obs)
        episode_reward = 0
        step_count = 0
        
        while True:
            action = env.action_space.sample()
            next_obs, _, done, truncated, info = env.step(action)
            nav = env.vehicle.navigation.get_navi_info()
            next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            
            reward = compute_reward(obs, action, next_obs, done, info)
            next_processed_obs = preprocess_metaurban_observation(next_obs)
            
            # BEV 학습용 데이터
            bev_data = {
                'worker_id': worker_id,
                'images': processed_obs['images'],
                'goal_vec': processed_obs['goal_vec'],
                'bev_gt': processed_obs['bev_gt'],
                'action': torch.tensor(action, dtype=torch.float32),
                'reward': reward,
                'done': done or truncated
            }
            
            try:
                data_queue.put(bev_data, timeout=1.0)
            except:
                print(f"Worker {worker_id}: Queue full, skipping data")
            
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                print(f"Worker {worker_id}: Episode finished - Reward: {episode_reward:.2f}, Steps: {step_count}")
                obs, _ = env.reset()
                nav = env.vehicle.navigation.get_navi_info()
                obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
                processed_obs = preprocess_metaurban_observation(obs)
                episode_reward = 0
                step_count = 0
            else:
                obs = next_obs
                processed_obs = next_processed_obs
            
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        env.close()
        print(f"Worker {worker_id} finished")

class BEVFormerTrainer:
    """BEVFormer 학습 클래스"""
    def __init__(self, model, device='cuda', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.data_buffer = deque(maxlen=20000)
        
    def add_data(self, data):
        """데이터 버퍼에 추가"""
        self.data_buffer.append(data)
    
    def train_step(self, batch_size=16):
        """한 스텝 학습"""
        if len(self.data_buffer) < batch_size:
            return None
        
        # 배치 샘플링
        indices = np.random.choice(len(self.data_buffer), batch_size, replace=False)
        batch_data = [self.data_buffer[i] for i in indices]
        
        images = torch.stack([d['images'] for d in batch_data]).to(self.device)
        goal_vecs = torch.stack([d['goal_vec'] for d in batch_data]).to(self.device)
        bev_gts = torch.stack([d['bev_gt'] for d in batch_data]).to(self.device)
        
        self.optimizer.zero_grad()
        pred_bev = self.model(images, goal_vecs)
        loss = self.criterion(pred_bev, bev_gts)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def main():
    """메인 학습 루프"""
    NUM_WORKERS = 16  # 워커 수 줄임 (GPU 메모리 고려)
    BATCH_SIZE = 8
    TRAIN_INTERVAL = 100  # 매 100 데이터마다 학습
    SAVE_INTERVAL = 1000  # 매 1000 스텝마다 모델 저장
    
    # 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MetaUrbanBEVFormer(num_classes=20)  # MetaUrban semantic classes
    trainer = BEVFormerTrainer(model, device)
    
    # 데이터 수집 프로세스 시작
    data_queue = mp.Queue(maxsize=2048)
    workers = []
    
    for i in range(NUM_WORKERS):
        worker = mp.Process(target=env_worker, args=(i, data_queue, True), name=f"EnvWorker-{i}")
        worker.start()
        workers.append(worker)
    print(f"{NUM_WORKERS} data collection workers started.")
    
    # 학습 루프
    step_count = 0
    try:
        while True:
            # 데이터 수집
            try:
                data = data_queue.get(timeout=10.0)
                trainer.add_data(data)
                
                # 주기적으로 학습
                if step_count % TRAIN_INTERVAL == 0 and len(trainer.data_buffer) >= BATCH_SIZE:
                    loss = trainer.train_step(BATCH_SIZE)
                    if loss is not None:
                        print(f"Step {step_count}: Loss = {loss:.4f}, Buffer size = {len(trainer.data_buffer)}")
                
                # 주기적으로 모델 저장
                if step_count % SAVE_INTERVAL == 0 and step_count > 0:
                    trainer.save_model(f'metaurban_bevformer_step_{step_count}.pth')
                    print(f"Model saved at step {step_count}")
                
                step_count += 1
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        trainer.save_model('metaurban_bevformer_final.pth')
    finally:
        # 워커 종료
        for worker in workers:
            worker.terminate()
            worker.join(timeout=5.0)
        print("All workers terminated.")

if __name__ == "__main__":
    main()