# core_enhanced.py

# TODO : Reward, Policy D 수정

# multiprocessing 없이 단일 core를 통해 동작 확인 및 reward, loss확인.
# 이유는 braekpoint를 통해 출력을해보면서 디버깅 하기위함.
# 결과 : reward, loss가 매우 비정상적. goal, reward가 부정확해서 그런것으로 추정 

import multiprocessing as mp
# from queue import Queue,Full  # multiprocessing 없이 단일 core를 위한 Qeueu : 안전한 큐 구조체 : thread간 안전 데이터 공유
# 여러 thread의 동시 접근 제한

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import (
    Normal, Uniform, Categorical,
    TransformedDistribution, TanhTransform
)
from collections import deque # double-ended queue : threading 무관하여 안전하지않음. 대신 리스트보다 빠른 큐작업 제공
# 단일 스레드에서 큐를 사용하는경우 deque가 좋음.

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

# mp.set_start_method("spawn", force=True)


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
    horizon=300,  # Long horizon
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=False,  # More strict termination
    max_lateral_dist=10.0,  # Larger tolerance
    agent_observation=ThreeSourceMixObservation,
    
    # 이미지 관측 설정
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# ============================================================================
# 3. PPO 알고리즘 클래스 (메인 프로세스에서만 사용) # 왜냐하면 multi process로 동작하기위해서!
# ============================================================================

# @dataclass 목적 : 데이터 저장 전용 클래스 빠르고 간결하게 만들기 위함.
# @dataclass 가 붙은 클래스는 자동으로 __init__ 생성자를 만들어줌
#               __repr__ : 객체 출력 =>  print(cfg) 시 PPOConfig(lr=1e-4, gamma=0.98, ...)
#               __eq__ : 두 객체 비교 =>  cfg1 == cfg2로 값 비교 가능

# 장점 1. 코드 간결해짐.
# 장점 2. 속성값을 명확히 표현, 타입 힌트로 가독성 좋음
# 장점 3. 자동 비교, 출력 복사 유용

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01 # entropy : exploration
    max_grad_norm: float = 0.5
    ppo_epochs: int = 8  # More epochs for replay
    batch_size: int = 128 # backward시킬 batch 크기
    buffer_size: int = 1024
    update_frequency: int = 256
    
                                # buffer size가 batch에 비해 큰 이유 :  
                                # 일정량의 환경 경험을 버퍼에 모았다가, 그 전체(혹은 상당 부분)를 여러 번(Epoch) 반복해서 학습합니다.
                                # 즉, buffer는 PPO 업데이트를 위한 하나의 경험 집합(rollout, trajectory)
                                # 이때, buffer는 모두 같은 data D를 가짐. ~ iid approx
    
    # Mixture policy weight
    policy_weight_start: float = 0.3
    policy_weight_end: float = 0.8
    prior_weight_start: float = 0.5
    prior_weight_end: float = 0.0    # Prior D는 마지막에는 사용 안함
    exploration_weight: float = 0.2  # Fixed
    
    max_episodes: int = 5000
    max_steps_per_episode: int = 2000  # Long horizon
    save_frequency: int = 100
    eval_frequency: int = 50
    log_frequency: int = 10

# data관리
@dataclass
class Experience:
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor


class CNNFeatureExtractor(nn.Module): # -> 512 dim torch.tensor
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


# =================================
# Policy Network 
# =================================
# observation + goal vec력입력 
# -> MLP로 action N(mean,var) 학습 
# -> Mixture D sampling & prob 계산

class PPOPolicy(nn.Module):
    def __init__(self, feature_dim=512, goal_vec_dim=2, action_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        combined_dim = feature_dim + goal_vec_dim
        
        self.policy_mean = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        self.policy_std = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, action_dim),
        )
        
        self.sigma_linear = nn.Linear(128, action_dim)        

        b0 = math.log(math.exp(3.0)-1.0)
        nn.init.constant_(self.sigma_linear.bias, b0)
        self.softplus = nn.Softplus()
    
    def forward(self, images, goal_vec):
        img_features = self.feature_extractor(images)
        combined = torch.cat([img_features, goal_vec], dim=1)
        mean = self.policy_head(combined)

        
        # std = torch.exp(self.log_std.clamp(-20, 2))

        std = self.softplus(self.sigma_linear(self.policy_std(combined))) + 1e-6
        return mean, std

    # 먼저 policy 분포에 대해 만들자.

    def get_mixture_action_log_prob_mean_std(
        self,
        images: torch.Tensor,           # (batch, C, H, W)
        goal_vec: torch.Tensor,         # (batch, goal_dim)
        num_samples: int = 1,
        action: torch.Tensor = None     # unused
    ):
        # 1) policy mean/std
        policy_mean, policy_std = self.forward(images, goal_vec)  # (batch, action_dim)
        batch, action_dim = policy_mean.shape

        # 2) Tanh-스쿼시된 정책 분포 정의
        tanh = [TanhTransform(cache_size=1)]
        policy_dist = TransformedDistribution(
            Normal(policy_mean, policy_std),
            tanh
        )

        # 3) 샘플링: reparameterized sampling
        if num_samples == 1:
            # 결과 shape → (batch, action_dim)
            action_tanh = policy_dist.rsample()
            # log_prob shape → (batch,)
            log_prob = policy_dist.log_prob(action_tanh).sum(-1)
        else:
            # 결과 shape → (num_samples, batch, action_dim)
            action_tanh = policy_dist.rsample((num_samples,))
            # log_prob shape → (num_samples, batch)
            log_prob = policy_dist.log_prob(action_tanh).sum(-1)

        # 4) “mixture” 통계 (사실 policy 하나라서 policy 통계)
        # 평균은 tanh(policy_mean)로 근사
        mixture_mean = policy_mean.tanh()
        # std는 델타 방법(delta method)로 근사: std_y ≈ std_x * |d tanh / dx|
        mixture_std  = policy_std * (1 - mixture_mean.pow(2))

        return action_tanh, log_prob, mixture_mean, mixture_std
    
    
    # def get_mixture_action_log_prob_mean_std(
    #     self,
    #     images: torch.Tensor,
    #     goal_vec: torch.Tensor,
    #     weights: torch.Tensor,      # (3,), sums to 1
    #     num_samples: int = 1,
    #     action: torch.Tensor = None
    # ):
    #     # 1) policy mean/std
    #     policy_mean, policy_std = self.forward(images, goal_vec)  # → (batch, action_dim)
    #     batch, action_dim = policy_mean.shape

    #     # 2) prior mean/std (goal-directed)
    #     goal_dir    = F.normalize(goal_vec, dim=1)                # unit vector
    #     prior_mean  = goal_dir * 0.5                              # scale-down
    #     prior_std   = torch.ones_like(policy_std) * 0.3

    #     # 3) 분포 정의 (Tanh 스쿼시 포함)
    #     tanh = [TanhTransform(cache_size=1)]
    #     policy_dist = TransformedDistribution(
    #         Normal(policy_mean, policy_std), tanh
    #     )
    #     prior_dist  = TransformedDistribution(
    #         Normal(prior_mean,  prior_std ), tanh
    #     )
    #     expl_dist   = Uniform(-1.0, 1.0)  # already bounded

    #     # 4) 컴포넌트 선택
    #     cat    = Categorical(probs=weights)               # shape=(3,)
    #     comps  = cat.sample((num_samples, batch))          # (num_s, batch)

    #     # 5) 샘플 & 로그확률 저장용 텐서
    #     samples = torch.zeros((num_samples, batch, action_dim))
    #     logps   = torch.zeros((num_samples, batch, 3))

    #     # 6) policy component
    #     mask0 = (comps == 0)
    #     if mask0.any():
    #         p_samps = policy_dist.rsample((num_samples,))      # (num_s, batch, action_dim)
    #         p_logp  = policy_dist.log_prob(p_samps).sum(-1)    # (num_s, batch)
    #         samples[mask0]    = p_samps[mask0]
    #         logps[mask0, 0]   = p_logp[mask0]

    #     # 7) prior component
    #     mask1 = (comps == 1)
    #     if mask1.any():
    #         pr_samps = prior_dist.rsample((num_samples,))
    #         pr_logp  = prior_dist.log_prob(pr_samps).sum(-1)
    #         samples[mask1]   = pr_samps[mask1]
    #         logps[mask1, 1]  = pr_logp[mask1]

    #     # 8) exploration component
    #     mask2 = (comps == 2)
    #     if mask2.any():
    #         e_samps = torch.empty((num_samples, batch, action_dim)).uniform_(-1.0, 1.0)
    #         e_logp  = expl_dist.log_prob(e_samps).sum(-1)
    #         samples[mask2]   = e_samps[mask2]
    #         logps[mask2, 2]  = e_logp[mask2]

    #     # 9) mixture log-prob: log ∑ᵢ wᵢ pᵢ(a)
    #     log_w    = torch.log(weights).view(1,1,3)
    #     log_prob = torch.logsumexp(log_w + logps, dim=-1)  # (num_s, batch)

    #     # 10) mixture mean / std (가중합으로 간단 근사)
    #     #    Uniform(-1,1)의 mean=0, std=√((b−a)²/12)=√(4/12)=√(1/3)
    #     uni_mean = 0.
    #     uni_std  = (expl_dist.high - expl_dist.low) / torch.sqrt(torch.tensor(12.0))
    #     # component means: tanh(policy_mean), tanh(prior_mean), 0
    #     comp_means = torch.stack([
    #         policy_mean.tanh(),
    #         prior_mean.tanh(),
    #         torch.zeros_like(policy_mean)
    #     ], dim=-1)   # → (batch, action_dim, 3)
    #     comp_stds  = torch.stack([
    #         (policy_std),      # 이건 근사: std of tanh-N 약식
    #         (prior_std),
    #         uni_std.expand_as(policy_std)
    #     ], dim=-1)   # → (batch, action_dim, 3)

    #     mixture_mean = (weights.view(1,1,3) * comp_means).sum(-1)  # (batch, action_dim)
    #     mixture_std  = (weights.view(1,1,3) * comp_stds ).sum(-1)  # (batch, action_dim)

    #     # 11) 최종 action: 이미 샘플 되어 tanh 스쿼시 된 값
    #     #    (num_s, batch, action_dim) → 원하는 shape으로 reshape or select
    #     action_tanh = samples if num_samples>1 else samples.squeeze(0)

    #     return action_tanh, log_prob, mixture_mean, mixture_std


    """ 
    def get_mixture_action_log_prob_mean_std(self,images,goal_vec,weights,num_samples=1,action=None):
        # policy_mean = 0.2
        # policy_std  = 0.1
        
        # prior_mean  = -0.3
        # prior_std   = 0.2
        
        expl_low    = -1.0
        expl_high   = 1.0
                
        # Policy Distribution
        policy_mean, policy_std = self.forward(images,goal_vec)
        
        # Prior Distribution (goal-directed)
        goal_direction = torch.nn.functional.normalize(goal_vec,dim=1)
        prior_mean = goal_direction # 왜 ?*0.5
        prior_std  = torch.ones_like(policy_mean) * 0.3 # 분산을 작게 만들기 위함.
        
        # Exploration D (Uniform)
        # torch.distributions.Uniform(low,high)
        
        # sample action 
        # 1) 컴포넌트 선택 분포
        cat = torch.distributions.Categorical(probs=weights)
        comps = cat.sample((num_samples,))  # 0,1,2 인덱스

        # 2) 각 분포에서 미리 샘플 생성
        policy_samps = torch.normal(policy_mean, policy_std, size=(num_samples,))
        prior_samps  = torch.normal(prior_mean,  prior_std,  size=(num_samples,))
        expl_samps   = torch.rand(num_samples) * (expl_high - expl_low) + expl_low

        # 3) 컴포넌트별 샘플 매핑
        #  samples[mask] 는 mask가 True인 인덱스 자리만 골라내는 “Boolean 인덱싱”
        samples = torch.empty(num_samples)
        samples[comps == 0] = policy_samps[comps == 0]
        samples[comps == 1] = prior_samps[ comps == 1]
        samples[comps == 2] = expl_samps[  comps == 2]
            
        # mixture log probability
        
        
        
        
        return action, log_prob, policy_mean, policy_std
        
        
    # 가중치 할당된 정책, prior, 탐색 분포 합셕 -> mixture mean/variance 계산 -> 샘플 및 로그 확률 반환 : 여기 코드가 잘못됨.
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
    """
    
    
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
                
                # Replay Buffer를 통한 Batch 데이터
                batch_states_images = states_images[batch_indices]
                batch_states_goal_vec = states_goal_vec[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Mixture 분포로 부터 추출한 각 action에 대한 log prob, mean ,std
                weights = self.get_mixture_weights()
                _, new_log_probs, mean, std = self.policy.get_mixture_action_and_log_prob(
                    batch_states_images, batch_states_goal_vec, weights, batch_actions
                )
                
                # 이전 policy와 새로운 policy의 prob ratio : 이전 prob detach
                ratio = torch.exp(new_log_probs - batch_old_log_probs.detach())
                
                # surrogate Advantage : PPO 식에 따라 급격한 ratio에 대해 upper bound를 걸어버림.
                # 이때 Poligy Gradient에 따라 Advantage는 반드시 detach해서 업데이트 막아야함.
                surr1 = ratio * batch_advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Function MSE : bootstrap 방식으로 V(s') 계산함. 반드시 batch_returns.detach()
                current_values = self.value(batch_states_images, batch_states_goal_vec)
                value_loss = nn.MSELoss()(current_values, batch_returns.detach())
                
                # entropy 포함 (옵션) : 탐험(exploration)을 유도, 과도한 수렴 방지
                entropy = torch.distributions.Normal(mean, std).entropy().mean()

                # policy_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy  # if entropy_coef > 0
                
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
                # wandb.save(str(save_path))
        
        final_save_path = self.save_dir / "final_model.pth"
        self.agent.save_model(final_save_path)
        # wandb.save(str(final_save_path))
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