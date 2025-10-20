import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import List, Dict, Any, Optional


########## --- 개선된 데이터셋 수집기 클래스 (JSON 통합 버전) --- #########
class ImitationDatasetCollector:
    """모방학습용 데이터셋을 수집하는 클래스 - 에피소드별 디렉토리 구조"""
    
    def __init__(self, dataset_root="imitation_dataset"):
        self.dataset_root = dataset_root
        self.create_directory_structure()
        
        # [수정] 현재 에피소드 데이터를 하나의 리스트로 통합
        self.current_episode_trajectory = []
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
        # [수정] 현재 에피소드 데이터 초기화
        self.current_episode_trajectory = []
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
        rgb_image = (obs['image'][:,:,:,-1] * 255).astype(np.uint8)
        Image.fromarray(rgb_image).save(
            os.path.join(episode_dir, "rgb", rgb_filename)
        )
        
        # Depth 이미지 저장
        depth_image = obs['depth'][:,:,0,-1]
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
        
        # [수정] Action, State 등 모든 데이터를 하나의 딕셔너리로 통합
        sample_data = {
            "step": step,
            # "rgb_path": os.path.join("rgb", rgb_filename),
            # "depth_path": os.path.join("depth", depth_filename),
            # "semantic_path": os.path.join("semantic", semantic_filename),
            "action": [float(a) for a in action],
            "reward": float(reward),
            "position": agent_state["position"].tolist() if hasattr(agent_state["position"], 'tolist') else list(agent_state["position"]),
            "heading": float(agent_state["heading"]),
            "velocity": float(agent_state["velocity"]),
            "goal": [float(g) for g in goal_position],
            "done": False  # 에피소드 종료 시 업데이트될 예정
        }
        self.current_episode_trajectory.append(sample_data)
        
        # 이미지 크기 정보 저장 (첫 번째 샘플에서만)
        if self.dataset_info["image_size"] is None and 'image' in obs:
             self.dataset_info["image_size"] = list(obs['image'].shape[:2])

    def finish_episode(self, episode_info):
        """에피소드 종료 및 데이터 저장"""
        if not self.current_episode_trajectory:
            return
        
        episode_dir = os.path.join(self.dataset_root, f"episode_{self.episode_counter:04d}")
        
        # [수정] 마지막 스텝을 done=True로 설정
        if self.current_episode_trajectory:
            self.current_episode_trajectory[-1]["done"] = True
        
        # [수정] 통합된 trajectory.json 파일 저장
        with open(os.path.join(episode_dir, "trajectory.json"), 'w') as f:
            json.dump(self.current_episode_trajectory, f, indent=2)
        
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
            "total_steps": len(self.current_episode_trajectory),
            "episode_length": len(self.current_episode_trajectory)
        }
        with open(os.path.join(episode_dir, "episode_meta.json"), 'w') as f:
            json.dump(episode_meta, f, indent=2)
        
        # [수정] 전체 메타데이터 업데이트
        self.dataset_info["total_episodes"] = self.episode_counter
        self.dataset_info["total_samples"] += len(self.current_episode_trajectory)
        
        print(f"Episode {self.episode_counter} saved with {len(self.current_episode_trajectory)} samples")
    
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
        
        np.random.shuffle(episodes)
        
        test_size = int(total_episodes * test_ratio)
        val_size = int(total_episodes * val_ratio)
        train_size = total_episodes - test_size - val_size
        
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