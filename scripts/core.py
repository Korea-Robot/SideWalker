# core.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# ============================================================================
# 1. 환경 설정
# ============================================================================
SENSOR_SIZE = (256, 160)
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
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=True, 
    max_lateral_dist=5.0,
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    sensors={
        "rgb_camera":      (RGBCamera,     *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# ============================================================================
# 2. 데이터 구조 및 설정
# ============================================================================
@dataclass
class PPOConfig:
    """PPO 하이퍼파라미터"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64  # 작게 조정
    buffer_size: int = 256  # 작게 조정
    update_frequency: int = 64  # 더 자주 업데이트
    
    # 학습 설정
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    save_frequency: int = 100
    eval_frequency: int = 50
    
    # 로깅 설정
    log_frequency: int = 10

@dataclass
class Experience:
    """경험 데이터 구조"""
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor

# ============================================================================
# 3. 신경망 모델
# ============================================================================
class CNNFeatureExtractor(nn.Module):
    """이미지 특징 추출 CNN"""
    def __init__(self, input_channels=6, feature_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, SENSOR_SIZE[1], SENSOR_SIZE[0])
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PPOPolicy(nn.Module):
    """PPO 정책 네트워크"""
    def __init__(self, feature_dim=512, goal_vec_dim=2, action_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        combined_dim = feature_dim + goal_vec_dim
        
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, images, goal_vec):
        img_features = self.feature_extractor(images)
        combined = torch.cat([img_features, goal_vec], dim=1)
        mean = self.policy_head(combined)
        std = torch.exp(self.log_std.clamp(-20, 2))  # 안정성을 위한 클램핑
        return mean, std
    
    def get_action_and_log_prob(self, images, goal_vec, action=None):
        """행동 선택 및 로그 확률 계산"""
        mean, std = self.forward(images, goal_vec)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        # 로그 확률 계산 (tanh 변환 전)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # tanh 변환 적용
        action_tanh = torch.tanh(action)
        
        # Jacobian 보정
        log_prob = log_prob - torch.log(1 - action_tanh.pow(2) + 1e-7).sum(dim=-1)
        
        return action_tanh, log_prob, mean, std

class PPOValue(nn.Module):
    """PPO 가치 네트워크"""
    def __init__(self, feature_dim=512, goal_vec_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        combined_dim = feature_dim + goal_vec_dim
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, images, goal_vec):
        img_features = self.feature_extractor(images)
        combined = torch.cat([img_features, goal_vec], dim=1)
        value = self.value_head(combined)
        return value.squeeze(-1)

# ============================================================================
# 4. 환경 및 데이터 처리
# ============================================================================
class ObservationProcessor:
    """관측 데이터 전처리 클래스"""
    
    @staticmethod
    def preprocess_observation(obs):
        """관측 데이터를 모델 입력 형태로 변환"""
        # 이미지 데이터 처리
        depth = obs["depth"][..., -1]  # (H, W, 1) -> (H, W, 3)로 확장
        depth = np.concatenate([depth, depth, depth], axis=-1)
        semantic = obs["semantic"][..., -1]  # (H, W, 3)
        
        # 채널 결합 (H, W, 6)
        combined_img = np.concatenate([depth, semantic], axis=-1)
        
        # 정규화 및 채널 순서 변경 (H, W, C) -> (C, H, W)
        combined_img = combined_img.astype(np.float32) / 255.0
        combined_img = np.transpose(combined_img, (2, 0, 1))
        
        # goal_vec 추출
        goal_vec = obs["goal_vec"].astype(np.float32)
        
        return {
            'images': torch.tensor(combined_img).unsqueeze(0),  # 배치 차원 추가
            'goal_vec': torch.tensor(goal_vec).unsqueeze(0)
        }

class RewardCalculator:
    """보상 계산 클래스"""
    
    @staticmethod
    def compute_reward(obs, action, next_obs, done, info):
        """종합적인 보상 함수 - 수정된 버전"""
        reward = 0.0
        
        # 1. 목표 방향 보상
        goal_vec = next_obs["goal_vec"]
        goal_distance = np.linalg.norm(goal_vec)
        
        # 목표 거리 보상 (더 점진적으로)
        if goal_distance > 0:
            reward += 1.0 / (1.0 + goal_distance)  # 거리 역수 보상
        
        # 2. 속도 보상 (적절한 속도 유지)
        speed = info.get('speed', 0)
        if speed > 0:
            reward += min(speed / 20.0, 0.3)  # 속도 보상 제한
        
        # 3. 방향 정렬 보상
        if goal_distance > 0.5:  # 목표가 충분히 멀 때만
            velocity = info.get('velocity', np.array([0, 0]))
            if np.linalg.norm(velocity) > 0.1:
                cos_angle = np.dot(velocity, -goal_vec) / (np.linalg.norm(velocity) * goal_distance)
                reward += cos_angle * 0.2
        
        # 4. 페널티 (더 강하게)
        if info.get('crash', False):
            reward -= 15.0
            # print("CRASH detected!")
        if info.get('out_of_road', False):
            reward -= 8.0
            # print("OUT OF ROAD detected!")
        
        # 5. 성공 보상 (조건 강화)
        if info.get('arrive_dest', False) and goal_distance < 1.0:
            reward += 30.0
            # print("DESTINATION REACHED!")
        
        # 6. 시간 페널티 (생존 인센티브)
        reward -= 0.01
        
        # 7. 진행 상황 보상
        prev_goal_vec = obs.get("goal_vec", goal_vec)
        prev_distance = np.linalg.norm(prev_goal_vec)
        if prev_distance > goal_distance:
            reward += (prev_distance - goal_distance) * 2.0  # 진전 보상
        
        return reward

# ============================================================================
# 5. PPO 알고리즘
# ============================================================================
class PPOAgent:
    """PPO 에이전트"""
    
    def __init__(self, config: PPOConfig, device='cuda'):
        self.config = config
        self.device = device
        
        # 네트워크 초기화
        self.policy = PPOPolicy().to(device)
        self.value = PPOValue().to(device)
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.lr)
        
        # 경험 버퍼
        self.buffer = []
        
        # 학습 통계
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
    
    def select_action(self, state):
        """행동 선택"""
        self.policy.eval()
        self.value.eval()
        
        with torch.no_grad():
            images = state['images'].to(self.device)
            goal_vec = state['goal_vec'].to(self.device)
            
            # 행동 및 로그 확률 계산
            action, log_prob, _, _ = self.policy.get_action_and_log_prob(images, goal_vec)
            
            # 가치 계산
            value = self.value(images, goal_vec)
        
        return action.cpu(), log_prob.cpu(), value.cpu()
    
    def add_experience(self, experience: Experience):
        """경험 추가"""
        self.buffer.append(experience)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self):
        """PPO 업데이트"""
        if len(self.buffer) < self.config.batch_size:
            print(f"Buffer size {len(self.buffer)} < batch size {self.config.batch_size}, skipping update")
            return
        
        print(f"Updating with {len(self.buffer)} experiences")
        
        self.policy.train()
        self.value.train()
        
        # 데이터 준비
        states_images = torch.cat([exp.state['images'] for exp in self.buffer]).to(self.device)
        states_goal_vec = torch.cat([exp.state['goal_vec'] for exp in self.buffer]).to(self.device)
        actions = torch.stack([exp.action for exp in self.buffer]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in self.buffer], dtype=torch.float32).to(self.device)
        next_states_images = torch.cat([exp.next_state['images'] for exp in self.buffer]).to(self.device)
        next_states_goal_vec = torch.cat([exp.next_state['goal_vec'] for exp in self.buffer]).to(self.device)
        dones = torch.tensor([exp.done for exp in self.buffer], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([exp.log_prob for exp in self.buffer]).to(self.device)
        old_values = torch.stack([exp.value for exp in self.buffer]).to(self.device)
        
        # GAE 계산
        with torch.no_grad():
            next_values = self.value(next_states_images, next_states_goal_vec)
            advantages = self.compute_gae(rewards, old_values, next_values, dones).to(self.device)
            returns = advantages + old_values
            
            # 정규화
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.ppo_epochs):
            # 미니배치 생성
            batch_size = min(self.config.batch_size, len(self.buffer))
            indices = torch.randperm(len(self.buffer))[:batch_size]
            
            batch_states_images = states_images[indices]
            batch_states_goal_vec = states_goal_vec[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            
            # 현재 정책으로 로그 확률 계산
            _, new_log_probs, mean, std = self.policy.get_action_and_log_prob(
                batch_states_images, batch_states_goal_vec, batch_actions
            )
            
            # 엔트로피 계산
            entropy = torch.distributions.Normal(mean, std).entropy().mean()
            
            # 비율 계산
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO 클리핑 손실
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 함수 손실
            current_values = self.value(batch_states_images, batch_states_goal_vec)
            value_loss = nn.MSELoss()(current_values, batch_returns)
            
            # 정책 업데이트
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            # 가치 함수 업데이트
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            breakpoint()
        
        # 통계 저장
        self.stats['policy_losses'].append(total_policy_loss / self.config.ppo_epochs)
        self.stats['value_losses'].append(total_value_loss / self.config.ppo_epochs)
        self.stats['entropies'].append(total_entropy / self.config.ppo_epochs)
        
        # 버퍼 초기화
        self.buffer.clear()
        
        print(f"Updated - Policy Loss: {total_policy_loss/self.config.ppo_epochs:.4f}, "
              f"Value Loss: {total_value_loss/self.config.ppo_epochs:.4f}, "
              f"Entropy: {total_entropy/self.config.ppo_epochs:.4f}")
    
    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'stats': self.stats
        }, filepath)
    
    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.stats = checkpoint['stats']

# ============================================================================
# 6. 학습 루프
# ============================================================================
class Trainer:
    """학습 관리 클래스"""
    
    def __init__(self, config: PPOConfig, save_dir: str = "checkpoints"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 디바이스 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 에이전트 초기화
        self.agent = PPOAgent(config, self.device)
        
        # 환경 초기화
        self.env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
        
        # 유틸리티 클래스
        self.obs_processor = ObservationProcessor()
        self.reward_calculator = RewardCalculator()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """메인 학습 루프"""
        total_steps = 0
        
        for episode in range(self.config.max_episodes):
            episode_reward = 0
            episode_steps = 0
            
            # 환경 리셋
            obs, _ = self.env.reset()
            nav = self.env.vehicle.navigation.get_navi_info()
            obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            state = self.obs_processor.preprocess_observation(obs)
            
            print(f"\nEpisode {episode} started - Initial goal distance: {np.linalg.norm(obs['goal_vec']):.2f}")
            
            while episode_steps < self.config.max_steps_per_episode:
                # 행동 선택
                action, log_prob, value = self.agent.select_action(state)
                
                # 환경 스텝
                next_obs, _, done, truncated, info = self.env.step(action.squeeze().numpy())
                
                # goal_vec 업데이트
                nav = self.env.vehicle.navigation.get_navi_info()
                next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
                
                # 보상 계산
                reward = self.reward_calculator.compute_reward(obs, action, next_obs, done, info)
                
                # 다음 상태 전처리
                next_state = self.obs_processor.preprocess_observation(next_obs)
                
                # 경험 저장
                experience = Experience(
                    state=state,
                    action=action.squeeze(),
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated,
                    log_prob=log_prob.squeeze(),
                    value=value.squeeze()
                )
                
                self.agent.add_experience(experience)
                
                # 업데이트 (더 자주)
                if len(self.agent.buffer) >= self.config.update_frequency:
                    self.agent.update()
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # 상태 출력 (디버깅용)
                if episode_steps % 100 == 0:
                    goal_dist = np.linalg.norm(next_obs["goal_vec"])
                    print(f"  Step {episode_steps}: Reward={reward:.2f}, Goal_dist={goal_dist:.2f}")
                
                # 에피소드 종료 조건
                # # print("왜 종료되는걸까?")
                # print(done,truncated)
                if done or truncated:
                    # print(f"Episode {episode} ended - Done: {done}, Truncated: {truncated}")
                    # if info.get('arrive_dest', False):
                    #     print("  -> Arrived at destination!")
                    #     continue
                    if info.get('crash', False):
                        print("  -> Crashed!")
                        break
                    if info.get('out_of_road', False):
                        print("  -> Out of road!")
                        break
                
                obs = next_obs
                state = next_state
            
            # 남은 경험들도 업데이트
            if len(self.agent.buffer) > 0:
                self.agent.update()
            
            # 통계 저장
            self.agent.stats['episode_rewards'].append(episode_reward)
            self.agent.stats['episode_lengths'].append(episode_steps)
            
            print(f"Episode {episode} completed - Length: {episode_steps}, Reward: {episode_reward:.2f}")
            
            # 로깅
            if episode % self.config.log_frequency == 0:
                avg_reward = np.mean(self.agent.stats['episode_rewards'][-self.config.log_frequency:])
                avg_length = np.mean(self.agent.stats['episode_lengths'][-self.config.log_frequency:])
                self.logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.0f}")
            
            # 모델 저장
            if episode % self.config.save_frequency == 0 and episode > 0:
                save_path = self.save_dir / f"model_episode_{episode}.pth"
                self.agent.save_model(save_path)
                self.logger.info(f"Model saved to {save_path}")
        
        # 최종 모델 저장
        final_save_path = self.save_dir / "final_model.pth"
        self.agent.save_model(final_save_path)
        self.logger.info(f"Final model saved to {final_save_path}")
        
        # 학습 결과 시각화
        self.plot_training_results()
        print('save image')                

    
    def plot_training_results(self):
        """학습 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 에피소드 보상
        axes[0, 0].plot(self.agent.stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 에피소드 길이
        axes[0, 1].plot(self.agent.stats['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # 정책 손실
        if self.agent.stats['policy_losses']:
            axes[1, 0].plot(self.agent.stats['policy_losses'])
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
        
        # 가치 손실
        if self.agent.stats['value_losses']:
            axes[1, 1].plot(self.agent.stats['value_losses'])
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_results.png')
        plt.show()

# ============================================================================
# 7. 메인 실행 함수
# ============================================================================
def main():
    """메인 함수"""
    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 설정 (수정됨)
    config = PPOConfig(
        max_episodes=500,
        update_frequency=64,   # 더 자주 업데이트
        batch_size=32,         # 더 작은 배치 크기
        log_frequency=5
    )
    
    # 학습 시작
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()