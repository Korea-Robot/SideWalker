# core_enhanced.py

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
import wandb

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Environment Configuration
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
    horizon=2000,  # Long horizon
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=False,  # More strict termination
    max_lateral_dist=10.0,  # Larger tolerance
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 8  # More epochs for replay
    batch_size: int = 128
    buffer_size: int = 1024
    update_frequency: int = 256
    
    # Mixture policy weights
    policy_weight_start: float = 0.3
    policy_weight_end: float = 0.8
    prior_weight_start: float = 0.5
    prior_weight_end: float = 0.1
    exploration_weight: float = 0.2  # Fixed
    
    max_episodes: int = 5000
    max_steps_per_episode: int = 2000  # Long horizon
    save_frequency: int = 100
    eval_frequency: int = 50
    log_frequency: int = 10

@dataclass
class Experience:
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor

class CNNFeatureExtractor(nn.Module):
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
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std
    
    def get_mixture_action_and_log_prob(self, images, goal_vec, weights, action=None):
        # Policy distribution
        policy_mean, policy_std = self.forward(images, goal_vec)
        
        # Prior distribution (goal-directed)
        goal_direction = torch.nn.functional.normalize(goal_vec, dim=1)
        prior_mean = goal_direction * 0.5  # Scale factor
        prior_std = torch.ones_like(prior_mean) * 0.3
        
        # Exploration distribution (uniform -> normal approximation)
        exploration_mean = torch.zeros_like(policy_mean)
        exploration_std = torch.ones_like(policy_mean) * 1.0
        
        # Mixture weights
        w_policy, w_prior, w_exploration = weights
        
        # Mixture parameters
        mixture_mean = (w_policy * policy_mean + 
                       w_prior * prior_mean + 
                       w_exploration * exploration_mean)
        
        mixture_std = torch.sqrt(w_policy * (policy_std**2 + (policy_mean - mixture_mean)**2) +
                               w_prior * (prior_std**2 + (prior_mean - mixture_mean)**2) +
                               w_exploration * (exploration_std**2 + (exploration_mean - mixture_mean)**2))
        
        dist = torch.distributions.Normal(mixture_mean, mixture_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_tanh = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action_tanh.pow(2) + 1e-7).sum(dim=-1)
        
        return action_tanh, log_prob, mixture_mean, mixture_std

class PPOValue(nn.Module):
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

class ObservationProcessor:
    @staticmethod
    def preprocess_observation(obs):
        depth = obs["depth"][..., -1]
        depth = np.concatenate([depth, depth, depth], axis=-1)
        semantic = obs["semantic"][..., -1]
        
        combined_img = np.concatenate([depth, semantic], axis=-1)
        combined_img = combined_img.astype(np.float32) / 255.0
        combined_img = np.transpose(combined_img, (2, 0, 1))
        
        goal_vec = obs["goal_vec"].astype(np.float32)
        
        return {
            'images': torch.tensor(combined_img).unsqueeze(0),
            'goal_vec': torch.tensor(goal_vec).unsqueeze(0)
        }

class RewardCalculator:
    @staticmethod
    def compute_reward(obs, action, next_obs, done, info):
        reward = 0.0
        
        goal_vec = next_obs["goal_vec"]
        goal_distance = np.linalg.norm(goal_vec)
        
        if goal_distance > 0:
            reward += 2.0 / (1.0 + goal_distance)
        
        speed = info.get('speed', 0)
        if speed > 0:
            reward += min(speed / 15.0, 0.5)
        
        if goal_distance > 0.5:
            velocity = info.get('velocity', np.array([0, 0]))
            if np.linalg.norm(velocity) > 0.1:
                cos_angle = np.dot(velocity, -goal_vec) / (np.linalg.norm(velocity) * goal_distance)
                reward += cos_angle * 0.3
        
        if info.get('crash', False):
            reward -= 20.0
        if info.get('out_of_road', False):
            reward -= 10.0
        
        if info.get('arrive_dest', False):
            reward += 50.0
        
        reward -= 0.005  # Time penalty
        
        prev_goal_vec = obs.get("goal_vec", goal_vec)
        prev_distance = np.linalg.norm(prev_goal_vec)
        if prev_distance > goal_distance:
            reward += (prev_distance - goal_distance) * 3.0
        
        return reward

class PPOAgent:
    def __init__(self, config: PPOConfig, device='cuda'):
        self.config = config
        self.device = device
        
        self.policy = PPOPolicy().to(device)
        self.value = PPOValue().to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.lr)
        
        self.buffer = []
        self.episode_count = 0
        
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
    
    def get_mixture_weights(self):
        progress = min(self.episode_count / (self.config.max_episodes * 0.7), 1.0)
        
        w_policy = self.config.policy_weight_start + progress * (self.config.policy_weight_end - self.config.policy_weight_start)
        w_prior = self.config.prior_weight_start + progress * (self.config.prior_weight_end - self.config.prior_weight_start)
        w_exploration = self.config.exploration_weight
        
        total = w_policy + w_prior + w_exploration
        return w_policy/total, w_prior/total, w_exploration/total
    
    def select_action(self, state):
        self.policy.eval()
        self.value.eval()
        
        with torch.no_grad():
            images = state['images'].to(self.device)
            goal_vec = state['goal_vec'].to(self.device)
            
            weights = self.get_mixture_weights()
            action, log_prob, _, _ = self.policy.get_mixture_action_and_log_prob(images, goal_vec, weights)
            value = self.value(images, goal_vec)
        
        return action.cpu(), log_prob.cpu(), value.cpu()
    
    def add_experience(self, experience: Experience):
        self.buffer.append(experience)
    
    def compute_gae(self, rewards, values, next_values, dones):
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
        if len(self.buffer) < self.config.batch_size:
            return
        
        self.policy.train()
        self.value.train()
        
        # Prepare data
        states_images = torch.cat([exp.state['images'] for exp in self.buffer]).to(self.device)
        states_goal_vec = torch.cat([exp.state['goal_vec'] for exp in self.buffer]).to(self.device)
        actions = torch.stack([exp.action for exp in self.buffer]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in self.buffer], dtype=torch.float32).to(self.device)
        next_states_images = torch.cat([exp.next_state['images'] for exp in self.buffer]).to(self.device)
        next_states_goal_vec = torch.cat([exp.next_state['goal_vec'] for exp in self.buffer]).to(self.device)
        dones = torch.tensor([exp.done for exp in self.buffer], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([exp.log_prob for exp in self.buffer]).to(self.device)
        old_values = torch.stack([exp.value for exp in self.buffer]).to(self.device)
        
        # GAE calculation
        with torch.no_grad():
            next_values = self.value(next_states_images, next_states_goal_vec)
            advantages = self.compute_gae(rewards, old_values, next_values, dones).to(self.device)
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs for replay
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        buffer_size = len(self.buffer)
        
        for epoch in range(self.config.ppo_epochs):
            # Random sampling for each epoch
            indices = torch.randperm(buffer_size)
            
            for start_idx in range(0, buffer_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states_images = states_images[batch_indices]
                batch_states_goal_vec = states_goal_vec[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                weights = self.get_mixture_weights()
                _, new_log_probs, mean, std = self.policy.get_mixture_action_and_log_prob(
                    batch_states_images, batch_states_goal_vec, weights, batch_actions
                )
                
                entropy = torch.distributions.Normal(mean, std).entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                current_values = self.value(batch_states_images, batch_states_goal_vec)
                value_loss = nn.MSELoss()(current_values, batch_returns)
                
                # Policy update
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                # Value update
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        num_updates = self.config.ppo_epochs * (buffer_size // self.config.batch_size + 1)
        
        # WandB logging
        weights = self.get_mixture_weights()
        wandb.log({
            'training/policy_loss': total_policy_loss / num_updates,
            'training/value_loss': total_value_loss / num_updates,
            'training/entropy': total_entropy / num_updates,
            'training/mixture_policy_weight': weights[0],
            'training/mixture_prior_weight': weights[1],
            'training/mixture_exploration_weight': weights[2],
            'training/buffer_size': len(self.buffer)
        })
        
        self.stats['policy_losses'].append(total_policy_loss / num_updates)
        self.stats['value_losses'].append(total_value_loss / num_updates)
        self.stats['entropies'].append(total_entropy / num_updates)
        
        self.buffer.clear()
    
    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'stats': self.stats
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.stats = checkpoint['stats']

class Trainer:
    def __init__(self, config: PPOConfig, save_dir: str = "checkpoints"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize WandB
        wandb.init(
            project="metaurban-ppo-enhanced",
            config=config.__dict__,
            tags=["ppo", "mixture-policy", "long-horizon"]
        )
        
        self.agent = PPOAgent(config, self.device)
        self.env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
        self.obs_processor = ObservationProcessor()
        self.reward_calculator = RewardCalculator()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        total_steps = 0
        
        for episode in range(self.config.max_episodes):
            self.agent.episode_count = episode
            episode_reward = 0
            episode_steps = 0
            
            obs, _ = self.env.reset()
            nav = self.env.vehicle.navigation.get_navi_info()
            obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            state = self.obs_processor.preprocess_observation(obs)
            
            episode_info = {
                'crashes': 0,
                'out_of_roads': 0,
                'destinations_reached': 0,
                'max_goal_distance': np.linalg.norm(obs['goal_vec']),
                'min_goal_distance': np.linalg.norm(obs['goal_vec'])
            }
            
            while episode_steps < self.config.max_steps_per_episode:
                action, log_prob, value = self.agent.select_action(state)
                
                next_obs, _, done, truncated, info = self.env.step(action.squeeze().numpy())
                
                nav = self.env.vehicle.navigation.get_navi_info()
                next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
                
                reward = self.reward_calculator.compute_reward(obs, action, next_obs, done, info)
                next_state = self.obs_processor.preprocess_observation(next_obs)
                
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
                
                if len(self.agent.buffer) >= self.config.update_frequency:
                    self.agent.update()
                
                # Episode tracking
                goal_dist = np.linalg.norm(next_obs["goal_vec"])
                episode_info['min_goal_distance'] = min(episode_info['min_goal_distance'], goal_dist)
                
                if info.get('crash', False):
                    episode_info['crashes'] += 1
                if info.get('out_of_road', False):
                    episode_info['out_of_roads'] += 1
                if info.get('arrive_dest', False):
                    episode_info['destinations_reached'] += 1
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Only terminate on severe conditions for long horizon
                if done or truncated:
                    if info.get('crash', False) or info.get('out_of_road', False):
                        break
                    # Continue even if destination reached for long horizon training
                
                obs = next_obs
                state = next_state
            
            if len(self.agent.buffer) > 0:
                self.agent.update()
            
            self.agent.stats['episode_rewards'].append(episode_reward)
            self.agent.stats['episode_lengths'].append(episode_steps)
            
            # Comprehensive WandB logging
            wandb.log({
                'episode/reward': episode_reward,
                'episode/length': episode_steps,
                'episode/crashes': episode_info['crashes'],
                'episode/out_of_roads': episode_info['out_of_roads'],
                'episode/destinations_reached': episode_info['destinations_reached'],
                'episode/max_goal_distance': episode_info['max_goal_distance'],
                'episode/min_goal_distance': episode_info['min_goal_distance'],
                'episode/final_goal_distance': np.linalg.norm(next_obs["goal_vec"]),
                'global/episode': episode,
                'global/total_steps': total_steps
            })
            
            # Rolling averages
            if episode % self.config.log_frequency == 0 and episode > 0:
                recent_rewards = self.agent.stats['episode_rewards'][-self.config.log_frequency:]
                recent_lengths = self.agent.stats['episode_lengths'][-self.config.log_frequency:]
                
                wandb.log({
                    'metrics/avg_reward': np.mean(recent_rewards),
                    'metrics/avg_length': np.mean(recent_lengths),
                    'metrics/reward_std': np.std(recent_rewards),
                    'metrics/length_std': np.std(recent_lengths)
                })
                
                self.logger.info(f"Episode {episode}: Avg Reward: {np.mean(recent_rewards):.2f}, Avg Length: {np.mean(recent_lengths):.0f}")
            
            if episode % self.config.save_frequency == 0 and episode > 0:
                save_path = self.save_dir / f"model_episode_{episode}.pth"
                self.agent.save_model(save_path)
                wandb.save(str(save_path))
        
        final_save_path = self.save_dir / "final_model.pth"
        self.agent.save_model(final_save_path)
        wandb.save(str(final_save_path))
        wandb.finish()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    config = PPOConfig(
        max_episodes=5000,
        max_steps_per_episode=2000,
        update_frequency=256,
        batch_size=128,
        ppo_epochs=8,
        log_frequency=5
    )
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()