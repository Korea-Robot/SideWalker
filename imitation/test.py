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

########## --- PyTorch Dataset 클래스들 --- #########
# The Dataset classes from the previous fix are correct and need no changes.
class TransitionDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transforms: Optional[Dict[str, T.Compose]] = None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = {
                'rgb': T.Compose([T.Resize((360, 640)), T.ToTensor()]),
                'grayscale': T.Compose([T.Resize((360, 640)), T.ToTensor()])
            }
        self.transitions, self.episode_paths = [], []
        self._load_split_info()
        self._build_transition_index()
    def _load_split_info(self):
        split_file = os.path.join(self.root_dir, "train_test_split.json")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                episode_ids = json.load(f).get(self.split, [])
        else:
            episode_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("episode_")]
            episode_ids = [int(d.split("_")[1]) for d in episode_dirs]
        for ep_id in episode_ids:
            ep_path = os.path.join(self.root_dir, f"episode_{ep_id:04d}")
            if os.path.exists(ep_path): self.episode_paths.append(ep_path)
    def _build_transition_index(self):
        for ep_path in self.episode_paths:
            action_file = os.path.join(ep_path, "action_reward.json")
            if not os.path.exists(action_file): continue
            with open(action_file, 'r') as f: actions = json.load(f)
            for i in range(len(actions) - 1): self.transitions.append((ep_path, i))
    def __len__(self): return len(self.transitions)
    def _load_image(self, folder: str, ep_path: str, step: int):
        img_path = os.path.join(ep_path, folder, f"{step:04d}.png")
        img = Image.open(img_path) if os.path.exists(img_path) else Image.new('RGB' if folder == 'rgb' else 'L', (640, 360), 0)
        transform = self.transforms.get("rgb" if folder == "rgb" else "grayscale")
        if folder == "rgb" and img.mode != 'RGB': img = img.convert('RGB')
        elif folder != "rgb" and img.mode != 'L': img = img.convert('L')
        return transform(img)
    def __getitem__(self, idx):
        ep_path, step_idx = self.transitions[idx]
        rgb_t, depth_t, semantic_t = self._load_image("rgb", ep_path, step_idx), self._load_image("depth", ep_path, step_idx), self._load_image("semantic", ep_path, step_idx)
        rgb_tp1, depth_tp1, semantic_tp1 = self._load_image("rgb", ep_path, step_idx + 1), self._load_image("depth", ep_path, step_idx + 1), self._load_image("semantic", ep_path, step_idx + 1)
        with open(os.path.join(ep_path, "action_reward.json"), 'r') as f: actions = json.load(f)
        with open(os.path.join(ep_path, "ego_state.json"), 'r') as f: ego_states = json.load(f)
        action, reward, done = torch.tensor(actions[step_idx]["action"], dtype=torch.float32), torch.tensor(actions[step_idx]["reward"], dtype=torch.float32), torch.tensor(actions[step_idx]["done"], dtype=torch.bool)
        ego_t, ego_tp1 = ego_states[step_idx], ego_states[step_idx + 1]
        position_t, position_tp1 = torch.tensor(ego_t["position"], dtype=torch.float32), torch.tensor(ego_tp1["position"], dtype=torch.float32)
        heading_t, heading_tp1 = torch.tensor(ego_t["heading"], dtype=torch.float32), torch.tensor(ego_tp1["heading"], dtype=torch.float32)
        goal_position = torch.tensor(ego_t["goal_position"], dtype=torch.float32)
        return {"rgb_t": rgb_t, "depth_t": depth_t, "semantic_t": semantic_t, "action": action, "reward": reward, "done": done, "rgb_tp1": rgb_tp1, "depth_tp1": depth_tp1, "semantic_tp1": semantic_tp1, "position_t": position_t, "position_tp1": position_tp1, "heading_t": heading_t, "heading_tp1": heading_tp1, "goal_position": goal_position}

class EpisodeDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transforms: Optional[Dict[str, T.Compose]] = None, max_episode_length: Optional[int] = None):
        self.root_dir, self.split, self.transforms, self.max_episode_length = root_dir, split, transforms, max_episode_length
        if self.transforms is None: self.transforms = {'rgb': T.Compose([T.Resize((360, 640)), T.ToTensor()]), 'grayscale': T.Compose([T.Resize((360, 640)), T.ToTensor()])}
        self.episode_paths = []
        self._load_split_info()
    def _load_split_info(self):
        split_file = os.path.join(self.root_dir, "train_test_split.json")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f: episode_ids = json.load(f).get(self.split, [])
        else:
            episode_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("episode_")]
            episode_ids = [int(d.split("_")[1]) for d in episode_dirs]
        for ep_id in episode_ids:
            ep_path = os.path.join(self.root_dir, f"episode_{ep_id:04d}")
            if os.path.exists(ep_path): self.episode_paths.append(ep_path)
    def __len__(self): return len(self.episode_paths)
    def _load_image(self, folder: str, ep_path: str, step: int):
        img_path = os.path.join(ep_path, folder, f"{step:04d}.png")
        img = Image.open(img_path) if os.path.exists(img_path) else Image.new('RGB' if folder == 'rgb' else 'L', (640, 360), 0)
        transform = self.transforms.get("rgb" if folder == "rgb" else "grayscale")
        if folder == "rgb" and img.mode != 'RGB': img = img.convert('RGB')
        elif folder != "rgb" and img.mode != 'L': img = img.convert('L')
        return transform(img)
    def __getitem__(self, idx):
        ep_path = self.episode_paths[idx]
        with open(os.path.join(ep_path, "action_reward.json"), 'r') as f: actions = json.load(f)
        with open(os.path.join(ep_path, "ego_state.json"), 'r') as f: ego_states = json.load(f)
        episode_length = min(len(actions), self.max_episode_length) if self.max_episode_length is not None else len(actions)
        episode_data = []
        for step in range(episode_length):
            rgb, depth, semantic = self._load_image("rgb", ep_path, step), self._load_image("depth", ep_path, step), self._load_image("semantic", ep_path, step)
            action, reward, done = torch.tensor(actions[step]["action"], dtype=torch.float32), torch.tensor(actions[step]["reward"], dtype=torch.float32), torch.tensor(actions[step]["done"], dtype=torch.bool)
            ego_state = ego_states[step]
            position, heading, goal_position = torch.tensor(ego_state["position"], dtype=torch.float32), torch.tensor(ego_state["heading"], dtype=torch.float32), torch.tensor(ego_state["goal_position"], dtype=torch.float32)
            episode_data.append({"rgb": rgb, "depth": depth, "semantic": semantic, "action": action, "reward": reward, "done": done, "position": position, "heading": heading, "goal_position": goal_position, "step": step})
        return {"episode": episode_data, "episode_length": episode_length, "episode_path": ep_path}

########## --- 커스텀 Collate 함수들 --- #########
# (No changes needed here)
def pad_sequences(sequences: List[List[torch.Tensor]], max_len=None):
    if max_len is None: max_len = max(len(seq) for seq in sequences) if sequences else 0
    padded, masks = [], []
    for seq in sequences:
        seq_len = len(seq)
        if seq_len > 0:
            if seq_len < max_len:
                padding = [torch.zeros_like(seq[0]) for _ in range(max_len - seq_len)]
                padded_seq, mask = seq + padding, [1] * seq_len + [0] * (max_len - seq_len)
            else:
                padded_seq, mask = seq[:max_len], [1] * max_len
            padded.append(torch.stack(padded_seq)); masks.append(torch.tensor(mask, dtype=torch.bool))
    if not padded: return torch.empty(0), torch.empty(0)
    return torch.stack(padded), torch.stack(masks)

def collate_episodes(batch):
    episodes = [item["episode"] for item in batch]
    episode_lengths = [item["episode_length"] for item in batch]
    max_len = max(episode_lengths) if episode_lengths else 0
    if max_len == 0: return {"rgb": torch.empty(0), "mask": torch.empty(0), "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long)}
    def extract_and_pad(key):
        sequences = [[step[key] for step in ep] for ep in episodes]
        if sequences and sequences[0] and sequences[0][0].ndim == 0: sequences = [[s.unsqueeze(0) for s in seq] for seq in sequences]
        padded, mask = pad_sequences(sequences, max_len)
        return padded, mask
    rgb_padded, mask = extract_and_pad("rgb")
    depth_padded, _ = extract_and_pad("depth")
    semantic_padded, _ = extract_and_pad("semantic")
    action_padded, _ = extract_and_pad("action")
    reward_padded, _ = extract_and_pad("reward")
    position_padded, _ = extract_and_pad("position")
    heading_padded, _ = extract_and_pad("heading")
    goal_padded, _ = extract_and_pad("goal_position")
    return {"rgb": rgb_padded, "depth": depth_padded, "semantic": semantic_padded, "actions": action_padded.squeeze(-1) if action_padded.ndim > 2 else action_padded, "rewards": reward_padded.squeeze(-1), "positions": position_padded, "headings": heading_padded.squeeze(-1), "goals": goal_padded, "mask": mask, "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long)}

########## --- 사용 예시 --- #########
def create_dataloaders(dataset_root: str, batch_size: int = 4, num_workers: int = 2):
    img_size = (360, 640)
    rgb_transform = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    grayscale_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    transforms = {'rgb': rgb_transform, 'grayscale': grayscale_transform}
    train_transition_dataset = TransitionDataset(dataset_root, split="train", transforms=transforms)
    val_transition_dataset = TransitionDataset(dataset_root, split="validation", transforms=transforms)
    train_transition_loader = DataLoader(train_transition_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_transition_loader = DataLoader(val_transition_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_episode_dataset = EpisodeDataset(dataset_root, split="train", transforms=transforms)
    val_episode_dataset = EpisodeDataset(dataset_root, split="validation", transforms=transforms)
    train_episode_loader = DataLoader(train_episode_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_episodes)
    val_episode_loader = DataLoader(val_episode_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_episodes)
    return {"train_transition": train_transition_loader, "val_transition": val_transition_loader, "train_episode": train_episode_loader, "val_episode": val_episode_loader}

if __name__ == "__main__":

    print("Creating a dummy dataset for testing...")
    dataset_root = "imitation_dataset"
    dataloaders = create_dataloaders(dataset_root, batch_size=2, num_workers=0)
    
    print("=== Transition Dataset Test ===")
    try:
        for i, batch in enumerate(dataloaders["train_transition"]):
            print(f"Batch {i}:")
            print(f"  RGB shape: {batch['rgb_t'].shape}")
            print(f"  Depth shape: {batch['depth_t'].shape}")
            print(f"  Action shape: {batch['action'].shape}")
            if i >= 1: break
    except Exception as e: print(f"Error during Transition Dataset test: {e}")

    print("\n=== Episode Dataset Test ===")
    try:
        for i, batch in enumerate(dataloaders["train_episode"]):
            print(f"Batch {i}:")
            print(f"  RGB shape: {batch['rgb'].shape}")
            print(f"  Actions shape: {batch['actions'].shape}")
            print(f"  Episode lengths: {batch['episode_lengths']}")
            if i >= 1: break
    except Exception as e: print(f"Error during Episode Dataset test: {e}")