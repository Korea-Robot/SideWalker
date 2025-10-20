# improved_walle_dataset.py
import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import List, Dict, Any, Optional


########## --- 개선된 데이터셋 수집기 클래스 --- #########
class ImitationDatasetCollector:
    """모방학습용 데이터셋을 수집하는 클래스 - 에피소드별 디렉토리 구조"""
    
    def __init__(self, dataset_root="imitation_dataset"):
        self.dataset_root = dataset_root
        self.create_directory_structure() # root_data 생성 
        
        # 현재 에피소드 데이터 저장
        self.current_episode_data = []
        self.current_episode_actions = []
        self.current_episode_ego_states = []
        self.current_episode_waypoints = []
        self.episode_counter = 0
        
        # 메타데이터
        self.dataset_info = {
            "created_at": datetime.now().isoformat(),
            "total_episodes": 0,
            "total_samples": 0,
            "image_size": None,
            "action_space": {"steer": [-1.0, 1.0], "throttle": [-1.0, 1.0]},
            "sensors": ["rgb_camera", "depth_camera", "semantic_camera"]
        }
        
    def create_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        os.makedirs(self.dataset_root, exist_ok=True)
    
    def start_new_episode(self, waypoints=None):
        """새로운 에피소드 시작"""
        self.current_episode_data = []
        self.current_episode_actions = []
        self.current_episode_ego_states = []
        self.current_episode_waypoints = waypoints if waypoints is not None else []
        self.episode_counter += 1
        
        # 에피소드 디렉토리 생성
        episode_dir = os.path.join(self.dataset_root, f"episode_{self.episode_counter:04d}")
        os.makedirs(episode_dir, exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "semantic"), exist_ok=True)
        
    def collect_sample(self, obs, action, agent_state, goal_position, reward, step):
        """한 스텝의 데이터를 수집"""
        episode_dir = os.path.join(self.dataset_root, f"episode_{self.episode_counter:04d}")
        
        # 이미지 파일명
        rgb_filename = f"{step:04d}.png"
        depth_filename = f"{step:04d}.png"
        semantic_filename = f"{step:04d}.png"
        
        
        # RGB 이미지 저장
        # if 'rgb_camera' in obs:
        rgb_image = (obs['image'][:,:,:,-1] * 255).astype(np.uint8)
        Image.fromarray(rgb_image).save(
            os.path.join(episode_dir, "rgb", rgb_filename)
        )
        
        # Depth 이미지 저장
    
        depth_image = obs['depth'][:,:,0,-1]
        # Depth를 0-255 범위로 정규화
        if depth_image.max() > depth_image.min():
            depth_normalized = ((depth_image - depth_image.min()) / 
                                (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
        Image.fromarray(depth_normalized, mode='L').save(
            os.path.join(episode_dir, "depth", depth_filename)
        )
    
        # Semantic 이미지 저장
        semantic_image = (obs['semantic'][:,:,:,-1] * 255).astype(np.uint8)
        Image.fromarray(semantic_image).save(
            os.path.join(episode_dir, "semantic", semantic_filename)
        )
        
        
        # Action과 reward 데이터 수집
        action_data = {
            "step": step,
            "action": [float(action[0]), float(action[1])],
            "reward": float(reward),
            "goal":goal_position,
            "done": False  # 에피소드 종료 시 업데이트될 예정
        }
        self.current_episode_actions.append(action_data)
        
        # Ego state 데이터 수집
        ego_state_data = {
            "step": step,
            "position": agent_state["position"].tolist() if hasattr(agent_state["position"], 'tolist') else list(agent_state["position"]),
            "heading": float(agent_state["heading"]),
            "velocity": float(agent_state["velocity"]),
            "angular_velocity": float(agent_state.get("angular_velocity", 0.0)),
            "goal_position": [float(goal_position[0]), float(goal_position[1])]
        }
        self.current_episode_ego_states.append(ego_state_data)
        
        # 이미지 크기 정보 저장 (첫 번째 샘플에서만)
        if self.dataset_info["image_size"] is None and 'rgb_camera' in obs:
            self.dataset_info["image_size"] = list(obs['rgb_camera'].shape[:2])
    
    def finish_episode(self, episode_info):
        """에피소드 종료 및 데이터 저장"""
        if not self.current_episode_actions:
            return
        
        episode_dir = os.path.join(self.dataset_root, f"episode_{self.episode_counter:04d}")
        
        # 마지막 스텝을 done=True로 설정
        if self.current_episode_actions:
            self.current_episode_actions[-1]["done"] = True
        
        # action_reward.json 저장
        with open(os.path.join(episode_dir, "action_reward_goal.json"), 'w') as f:
            json.dump(self.current_episode_actions, f, indent=2)
        
        # ego_state.json 저장
        with open(os.path.join(episode_dir, "ego_state.json"), 'w') as f:
            json.dump(self.current_episode_ego_states, f, indent=2)
        
        # waypoints.json 저장
        waypoints_data = {
            "episode_id": self.episode_counter,
            "waypoints": [wp.tolist() if hasattr(wp, 'tolist') else list(wp) for wp in self.current_episode_waypoints],
            "num_waypoints": len(self.current_episode_waypoints)
        }
        with open(os.path.join(episode_dir, "waypoints.json"), 'w') as f:
            json.dump(waypoints_data, f, indent=2)
        
        # 에피소드 메타정보 저장
        episode_meta = {
            "episode_id": self.episode_counter,
            "episode_info": episode_info,
            "total_steps": len(self.current_episode_actions),
            "episode_length": len(self.current_episode_actions)
        }
        with open(os.path.join(episode_dir, "episode_meta.json"), 'w') as f:
            json.dump(episode_meta, f, indent=2)
        
        # 전체 메타데이터 업데이트
        self.dataset_info["total_episodes"] = self.episode_counter
        self.dataset_info["total_samples"] += len(self.current_episode_actions)
        
        print(f"Episode {self.episode_counter} saved with {len(self.current_episode_actions)} samples")
    
    def save_dataset_info(self):
        """데이터셋 메타정보 저장"""
        with open(os.path.join(self.dataset_root, "dataset_info.json"), 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def create_train_test_split(self, test_ratio=0.2, val_ratio=0.1):
        """학습/검증/테스트 데이터 분할"""
        total_episodes = self.episode_counter
        if total_episodes == 0:
            return
            
        episodes = list(range(1, total_episodes + 1))
        
        # 랜덤 셔플
        np.random.shuffle(episodes)
        
        # 분할 계산
        test_size = int(total_episodes * test_ratio)
        val_size = int(total_episodes * val_ratio)
        train_size = total_episodes - test_size - val_size
        
        # 분할
        train_episodes = episodes[:train_size]
        val_episodes = episodes[train_size:train_size + val_size]
        test_episodes = episodes[train_size + val_size:]
        
        split_info = {
            "train": sorted(train_episodes),
            "validation": sorted(val_episodes),
            "test": sorted(test_episodes),
            "split_info": {
                "total_episodes": total_episodes,
                "train_episodes": len(train_episodes),
                "val_episodes": len(val_episodes),
                "test_episodes": len(test_episodes)
            }
        }
        
        with open(os.path.join(self.dataset_root, "train_test_split.json"), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Dataset split created: Train({len(train_episodes)}), Val({len(val_episodes)}), Test({len(test_episodes)})")


########## --- PyTorch Dataset 클래스들 --- #########

class TransitionDataset(Dataset):
    """Transition 기반 데이터셋 (t -> t+1)"""
    
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((360, 640)),
            T.ToTensor()
        ])
        
        self.transitions = []  # (episode_path, step_idx)
        self.episode_paths = []
        
        self._load_split_info()
        self._build_transition_index()
    
    def _load_split_info(self):
        """Train/test split 정보 로드"""
        split_file = os.path.join(self.root_dir, "train_test_split.json")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            episode_ids = split_info.get(self.split, [])
        else:
            # split 파일이 없으면 모든 에피소드 사용
            episode_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("episode_")]
            episode_ids = [int(d.split("_")[1]) for d in episode_dirs]
        
        # 에피소드 경로 생성
        for ep_id in episode_ids:
            ep_path = os.path.join(self.root_dir, f"episode_{ep_id:04d}")
            if os.path.exists(ep_path):
                self.episode_paths.append(ep_path)
    
    def _build_transition_index(self):
        """Transition 인덱스 구축"""
        for ep_path in self.episode_paths:
            action_file = os.path.join(ep_path, "action_reward.json")
            if not os.path.exists(action_file):
                continue
                
            with open(action_file, 'r') as f:
                actions = json.load(f)
            
            # t -> t+1 transition을 위해 length-1까지만
            for i in range(len(actions) - 1):
                self.transitions.append((ep_path, i))
    
    def __len__(self):
        return len(self.transitions)
    
    def _load_image(self, folder: str, ep_path: str, step: int):
        """이미지 로드 및 전처리"""
        img_path = os.path.join(ep_path, folder, f"{step:04d}.png")
        if not os.path.exists(img_path):
            # 이미지가 없으면 검은색 이미지 반환
            if folder == "depth":
                img = Image.new('L', (256, 160), 0)
            else:
                img = Image.new('RGB', (256, 160), (0, 0, 0))
        else:
            img = Image.open(img_path)
            if folder == "depth" and img.mode != 'L':
                img = img.convert('L')
            elif folder != "depth" and img.mode != 'RGB':
                img = img.convert('RGB')
        
        return self.transform(img)
    
    def __getitem__(self, idx):
        ep_path, step_idx = self.transitions[idx]
        
        # 시간 t의 관측
        rgb_t = self._load_image("rgb", ep_path, step_idx)
        depth_t = self._load_image("depth", ep_path, step_idx)
        semantic_t = self._load_image("semantic", ep_path, step_idx)
        
        # 시간 t+1의 관측
        rgb_tp1 = self._load_image("rgb", ep_path, step_idx + 1)
        depth_tp1 = self._load_image("depth", ep_path, step_idx + 1)
        semantic_tp1 = self._load_image("semantic", ep_path, step_idx + 1)
        
        # Action과 reward 로드
        with open(os.path.join(ep_path, "action_reward.json"), 'r') as f:
            actions = json.load(f)
        
        # Ego state 로드
        with open(os.path.join(ep_path, "ego_state.json"), 'r') as f:
            ego_states = json.load(f)
        
        action = torch.tensor(actions[step_idx]["action"], dtype=torch.float32)
        reward = torch.tensor(actions[step_idx]["reward"], dtype=torch.float32)
        done = torch.tensor(actions[step_idx]["done"], dtype=torch.bool)
        
        # Ego state 정보
        ego_t = ego_states[step_idx]
        ego_tp1 = ego_states[step_idx + 1]
        
        position_t = torch.tensor(ego_t["position"], dtype=torch.float32)
        position_tp1 = torch.tensor(ego_tp1["position"], dtype=torch.float32)
        heading_t = torch.tensor(ego_t["heading"], dtype=torch.float32)
        heading_tp1 = torch.tensor(ego_tp1["heading"], dtype=torch.float32)
        goal_position = torch.tensor(ego_t["goal_position"], dtype=torch.float32)
        
        return {
            "rgb_t": rgb_t,
            "depth_t": depth_t,
            "semantic_t": semantic_t,
            "action": action,
            "reward": reward,
            "done": done,
            "rgb_tp1": rgb_tp1,
            "depth_tp1": depth_tp1,
            "semantic_tp1": semantic_tp1,
            "position_t": position_t,
            "position_tp1": position_tp1,
            "heading_t": heading_t,
            "heading_tp1": heading_tp1,
            "goal_position": goal_position
        }


class EpisodeDataset(Dataset):
    """전체 에피소드를 반환하는 데이터셋"""
    
    def __init__(self, root_dir: str, split: str = "train", transform=None, max_episode_length: Optional[int] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((360, 640)),
            T.ToTensor()
        ])
        self.max_episode_length = max_episode_length
        
        self.episode_paths = []
        self._load_split_info()
    
    def _load_split_info(self):
        """Train/test split 정보 로드"""
        split_file = os.path.join(self.root_dir, "train_test_split.json")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            episode_ids = split_info.get(self.split, [])
        else:
            # split 파일이 없으면 모든 에피소드 사용
            episode_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("episode_")]
            episode_ids = [int(d.split("_")[1]) for d in episode_dirs]
        
        # 에피소드 경로 생성
        for ep_id in episode_ids:
            ep_path = os.path.join(self.root_dir, f"episode_{ep_id:04d}")
            if os.path.exists(ep_path):
                self.episode_paths.append(ep_path)
    
    def __len__(self):
        return len(self.episode_paths)
    
    def _load_image(self, folder: str, ep_path: str, step: int):
        """이미지 로드 및 전처리"""
        img_path = os.path.join(ep_path, folder, f"{step:04d}.png")
        if not os.path.exists(img_path):
            if folder == "depth":
                img = Image.new('L', (256, 160), 0)
            else:
                img = Image.new('RGB', (256, 160), (0, 0, 0))
        else:
            img = Image.open(img_path)
            if folder == "depth" and img.mode != 'L':
                img = img.convert('L')
            elif folder != "depth" and img.mode != 'RGB':
                img = img.convert('RGB')
        
        return self.transform(img)
    
    def __getitem__(self, idx):
        ep_path = self.episode_paths[idx]
        
        # 에피소드 메타정보 로드
        with open(os.path.join(ep_path, "action_reward.json"), 'r') as f:
            actions = json.load(f)
        
        with open(os.path.join(ep_path, "ego_state.json"), 'r') as f:
            ego_states = json.load(f)
        
        episode_length = len(actions)
        if self.max_episode_length is not None:
            episode_length = min(episode_length, self.max_episode_length)
        
        # 에피소드 데이터 구성
        episode_data = []
        for step in range(episode_length):
            # 이미지 로드
            rgb = self._load_image("rgb", ep_path, step)
            depth = self._load_image("depth", ep_path, step)
            semantic = self._load_image("semantic", ep_path, step)
            
            # 액션 및 상태 정보
            action = torch.tensor(actions[step]["action"], dtype=torch.float32)
            reward = torch.tensor(actions[step]["reward"], dtype=torch.float32)
            done = torch.tensor(actions[step]["done"], dtype=torch.bool)
            
            ego_state = ego_states[step]
            position = torch.tensor(ego_state["position"], dtype=torch.float32)
            heading = torch.tensor(ego_state["heading"], dtype=torch.float32)
            goal_position = torch.tensor(ego_state["goal_position"], dtype=torch.float32)
            
            step_data = {
                "rgb": rgb,
                "depth": depth,
                "semantic": semantic,
                "action": action,
                "reward": reward,
                "done": done,
                "position": position,
                "heading": heading,
                "goal_position": goal_position,
                "step": step
            }
            episode_data.append(step_data)
        
        return {
            "episode": episode_data,
            "episode_length": episode_length,
            "episode_path": ep_path
        }


########## --- 커스텀 Collate 함수들 --- #########

def pad_sequences(sequences, max_len=None):
    """시퀀스들을 패딩"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    masks = []
    
    for seq in sequences:
        seq_len = len(seq)
        if seq_len < max_len:
            # 패딩 추가
            padding = [torch.zeros_like(seq[0]) for _ in range(max_len - seq_len)]
            padded_seq = seq + padding
            mask = [1] * seq_len + [0] * (max_len - seq_len)
        else:
            padded_seq = seq[:max_len]
            mask = [1] * max_len
        
        padded.append(torch.stack(padded_seq))
        masks.append(torch.tensor(mask, dtype=torch.bool))
    
    return torch.stack(padded), torch.stack(masks)


def collate_episodes(batch):
    """에피소드 배치를 위한 collate 함수"""
    episodes = [item["episode"] for item in batch]
    episode_lengths = [item["episode_length"] for item in batch]
    
    # 각 모달리티별로 시퀀스 추출
    rgb_sequences = [[step["rgb"] for step in episode] for episode in episodes]
    depth_sequences = [[step["depth"] for step in episode] for episode in episodes]
    semantic_sequences = [[step["semantic"] for step in episode] for episode in episodes]
    action_sequences = [[step["action"] for step in episode] for episode in episodes]
    reward_sequences = [[step["reward"] for step in episode] for episode in episodes]
    position_sequences = [[step["position"] for step in episode] for episode in episodes]
    heading_sequences = [[step["heading"] for step in episode] for episode in episodes]
    goal_sequences = [[step["goal_position"] for step in episode] for episode in episodes]
    
    # 패딩 적용
    max_len = max(episode_lengths)
    
    rgb_padded, rgb_mask = pad_sequences(rgb_sequences, max_len)
    depth_padded, depth_mask = pad_sequences(depth_sequences, max_len)
    semantic_padded, semantic_mask = pad_sequences(semantic_sequences, max_len)
    action_padded, action_mask = pad_sequences(action_sequences, max_len)
    reward_padded, reward_mask = pad_sequences(reward_sequences, max_len)
    position_padded, position_mask = pad_sequences(position_sequences, max_len)
    heading_padded, heading_mask = pad_sequences(heading_sequences, max_len)
    goal_padded, goal_mask = pad_sequences(goal_sequences, max_len)
    
    return {
        "rgb": rgb_padded,  # [batch, max_len, C, H, W]
        "depth": depth_padded,
        "semantic": semantic_padded,
        "actions": action_padded,  # [batch, max_len, action_dim]
        "rewards": reward_padded,  # [batch, max_len]
        "positions": position_padded,
        "headings": heading_padded,
        "goals": goal_padded,
        "mask": rgb_mask,  # [batch, max_len]
        "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long)
    }


########## --- 사용 예시 --- #########

def create_dataloaders(dataset_root: str, batch_size: int = 4, num_workers: int = 2):
    """데이터로더 생성 함수"""
    
    # Transform 정의
    transform = T.Compose([
        T.Resize((360, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])
    
    # Transition Dataset (개별 transition 학습용)
    train_transition_dataset = TransitionDataset(dataset_root, split="train", transform=transform)
    val_transition_dataset = TransitionDataset(dataset_root, split="validation", transform=transform)
    
    train_transition_loader = DataLoader(
        train_transition_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_transition_loader = DataLoader(
        val_transition_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Episode Dataset (시퀀스 학습용)
    train_episode_dataset = EpisodeDataset(dataset_root, split="train", transform=transform)
    val_episode_dataset = EpisodeDataset(dataset_root, split="validation", transform=transform)
    
    train_episode_loader = DataLoader(
        train_episode_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_episodes
    )
    
    val_episode_loader = DataLoader(
        val_episode_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_episodes
    )
    
    return {
        "train_transition": train_transition_loader,
        "val_transition": val_transition_loader,
        "train_episode": train_episode_loader,
        "val_episode": val_episode_loader
    }


if __name__ == "__main__":
    # 사용 예시
    dataset_root = "imitation_dataset"
    
    # 데이터로더 생성
    dataloaders = create_dataloaders(dataset_root, batch_size=4)
    
    # Transition Dataset 테스트
    print("=== Transition Dataset Test ===")
    for i, batch in enumerate(dataloaders["train_transition"]):
        print(f"Batch {i}:")
        print(f"  RGB shape: {batch['rgb_t'].shape}")
        print(f"  Action shape: {batch['action'].shape}")
        print(f"  Reward shape: {batch['reward'].shape}")
        if i >= 2:  # 처음 3개 배치만 테스트
            break
    
    # Episode Dataset 테스트
    print("\n=== Episode Dataset Test ===")
    for i, batch in enumerate(dataloaders["train_episode"]):
        print(f"Batch {i}:")
        print(f"  RGB shape: {batch['rgb'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        print(f"  Episode lengths: {batch['episode_lengths']}")
        print(f"  Mask shape: {batch['mask'].shape}")
        if i >= 2:  # 처음 3개 배치만 테스트
            break