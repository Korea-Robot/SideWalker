# walle_dataset.py
import os
import json
import numpy as np
from datetime import datetime
from PIL import Image


########## --- 데이터셋 수집기 클래스 --- #########
class ImitationDatasetCollector:
    """모방학습용 데이터셋을 수집하는 클래스"""
    
    def __init__(self, dataset_root="imitation_dataset"):
        self.dataset_root = dataset_root
        self.create_directory_structure()
        
        # 현재 에피소드 데이터 저장
        self.current_episode_data = []
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
        directories = [
            self.dataset_root,
            os.path.join(self.dataset_root, "images"),
            os.path.join(self.dataset_root, "depth"),
            os.path.join(self.dataset_root, "semantic"),
            os.path.join(self.dataset_root, "episodes")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def start_new_episode(self):
        """새로운 에피소드 시작"""
        self.current_episode_data = []
        self.episode_counter += 1
        
    def collect_sample(self, obs, action, agent_state, goal_position, reward, step):
        """한 스텝의 데이터를 수집"""
        timestamp = step
        
        # 이미지 데이터 저장
        rgb_filename = f"ep{self.episode_counter:04d}_step{step:04d}_rgb.png"
        depth_filename = f"ep{self.episode_counter:04d}_step{step:04d}_depth.png"
        semantic_filename = f"ep{self.episode_counter:04d}_step{step:04d}_semantic.png"
        
        # RGB 이미지 저장
        if 'rgb_camera' in obs:
            rgb_image = (obs['rgb_camera'] * 255).astype(np.uint8)
            Image.fromarray(rgb_image).save(
                os.path.join(self.dataset_root, "images", rgb_filename)
            )
        
        # Depth 이미지 저장
        if 'depth_camera' in obs:
            depth_image = obs['depth_camera']
            # Depth를 0-255 범위로 정규화
            depth_normalized = ((depth_image - depth_image.min()) / 
                              (depth_image.max() - depth_image.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized, mode='L').save(
                os.path.join(self.dataset_root, "depth", depth_filename)
            )
        
        # Semantic 이미지 저장
        if 'semantic_camera' in obs:
            semantic_image = (obs['semantic_camera'] * 255).astype(np.uint8)
            Image.fromarray(semantic_image).save(
                os.path.join(self.dataset_root, "semantic", semantic_filename)
            )
        
        # 샘플 데이터 구성
        sample_data = {
            "timestamp": timestamp,
            "step": step,
            "images": {
                "rgb": rgb_filename,
                "depth": depth_filename,
                "semantic": semantic_filename
            },
            "action": {
                "steer": float(action[0]),
                "throttle": float(action[1])
            },
            "agent_state": {
                "position": agent_state["position"].tolist(),
                "heading": float(agent_state["heading"]),
                "velocity": float(agent_state["velocity"]),
                "angular_velocity": float(agent_state.get("angular_velocity", 0.0))
            },
            "goal_position": {
                "ego_x": float(goal_position[0]),
                "ego_y": float(goal_position[1])
            },
            "reward": float(reward)
        }
        
        self.current_episode_data.append(sample_data)
        
        # 이미지 크기 정보 저장 (첫 번째 샘플에서만)
        if self.dataset_info["image_size"] is None and 'rgb_camera' in obs:
            self.dataset_info["image_size"] = list(obs['rgb_camera'].shape[:2])
    
    def finish_episode(self, episode_info):
        """에피소드 종료 및 데이터 저장"""
        if not self.current_episode_data:
            return
        
        # 에피소드 데이터 구성
        episode_data = {
            "episode_id": self.episode_counter,
            "episode_info": episode_info,
            "samples": self.current_episode_data,
            "total_steps": len(self.current_episode_data)
        }
        
        # JSON 파일로 저장
        episode_filename = f"episode_{self.episode_counter:04d}.json"
        with open(os.path.join(self.dataset_root, "episodes", episode_filename), 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # 메타데이터 업데이트
        self.dataset_info["total_episodes"] = self.episode_counter
        self.dataset_info["total_samples"] += len(self.current_episode_data)
        
        print(f"Episode {self.episode_counter} saved with {len(self.current_episode_data)} samples")
    
    def save_dataset_info(self):
        """데이터셋 메타정보 저장"""
        with open(os.path.join(self.dataset_root, "dataset_info.json"), 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def create_train_test_split(self, test_ratio=0.2, val_ratio=0.1):
        """학습/검증/테스트 데이터 분할"""
        total_episodes = self.episode_counter
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