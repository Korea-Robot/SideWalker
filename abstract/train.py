import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import typing
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import wandb
import torchvision.models as models #  --- MODIFIED: Added torchvision for EfficientNet ---

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# main.py
from env_config import EnvConfig
from metaurban.envs import SidewalkStaticMetaUrbanEnv

# 설정 불러오기
env_config = EnvConfig()

from utils import convert_to_egocentric, extract_sensor_data,create_and_save_plots # 여러 필요한 함수들 import

def collect_trajectory(env, policy: typing.Callable, max_steps: int = 1000) -> tuple[list[dict], list[tuple[float, float]], list[float]]:
    """환경에서 trajectory 수집"""
    observations = []
    actions = []
    rewards = []
    
    obs, info = env.reset()
    step_count = 0
    
    while step_count < max_steps:
        # 목표 지점 계산
        ego_goal_position = np.array([0.0, 0.0])
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 관찰 데이터 준비
        rgb_data, depth_data,semantic_data = extract_sensor_data(obs)
        
        obs_data = {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic':semantic_data,
            'goal': ego_goal_position
        }
        observations.append(obs_data)
        
        # 행동 선택
        action = policy(obs_data)
        actions.append(action)
        
        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    return observations, actions, rewards

def deviceof(m: nn.Module) -> torch.device:
    """모듈의 device 반환"""
    return next(m.parameters()).device

def obs_batch_to_tensor(obs_batch: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """관찰 배치를 텐서로 변환"""
    rgb_batch = []
    depth_batch = []
    goal_batch = []
    
    for obs in obs_batch:
        # RGB: (H, W, 3) -> (3, H, W)
        # rgb = torch.tensor(obs['rgb'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        rgb = torch.tensor(obs['semantic'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        rgb_batch.append(rgb)
        
        # Depth: (H, W, 3) -> (3, H, W)
        depth = torch.tensor(obs['depth'], dtype=torch.float32).permute(2, 0, 1)
        depth_batch.append(depth)
        
        # Goal: (2,)
        goal = torch.tensor(obs['goal'], dtype=torch.float32)
        goal_batch.append(goal)
    
    rgb_tensor = torch.stack(rgb_batch).to(device)
    depth_tensor = torch.stack(depth_batch).to(device)
    goal_tensor = torch.stack(goal_batch).to(device)
    
    return rgb_tensor, depth_tensor, goal_tensor

############## MODELS ############
from perceptnet import PerceptNet
from actor_critic import Actor,Critic

class NNPolicy:
    def __init__(self, net: Actor):
        self.net = net

    def __call__(self, obs: dict) -> tuple[float, float]:
        """관찰을 받아 행동을 반환"""
        self.net.eval() # Set to evaluation mode for inference
        rgb_tensor, depth_tensor, goal_tensor = obs_batch_to_tensor([obs], deviceof(self.net))
        
        with torch.no_grad():
            dist = self.net(rgb_tensor, depth_tensor, goal_tensor)
            action = dist.sample()[0]
            # Clip action to be within a reasonable range, e.g., [-1, 1]
            throttle = torch.clamp(action[0], -1.0, 1.0)
            steering = torch.clamp(action[1], -1.0, 1.0)
            
        self.net.train() # Set back to training mode
        return throttle.item(), steering.item()

# --- PPO 관련 함수들 (unchanged) ---
def rewards_to_go(trajectory_rewards: list[float], gamma: float) -> list[float]:
    """감마 할인된 reward-to-go 계산"""
    trajectory_len = len(trajectory_rewards)
    v_batch = np.zeros(trajectory_len)
    v_batch[-1] = trajectory_rewards[-1]
    
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + gamma * v_batch[t + 1]
    
    return list(v_batch)

def compute_advantage(
    critic: Critic,
    trajectory_observations: list[dict],
    trajectory_rewards: list[float],
    gamma: float
) -> list[float]:
    """GAE를 사용한 advantage 계산"""
    trajectory_len = len(trajectory_rewards)
    
    # Value 계산
    critic.eval() # Set to evaluation mode
    with torch.no_grad():
        rgb_tensor, depth_tensor, goal_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        obs_values = critic.forward(rgb_tensor, depth_tensor, goal_tensor).detach().cpu().numpy()
    critic.train() # Set back to training mode
    
    # Advantage = Reward-to-go - Value
    trajectory_advantages = np.array(rewards_to_go(trajectory_rewards, gamma)) - obs_values
    
    return list(trajectory_advantages)

@dataclass
class PPOConfig:
    ppo_eps: float
    ppo_grad_descent_steps: int

def compute_ppo_loss(
    pi_thetak_given_st: torch.distributions.MultivariateNormal,
    pi_theta_given_st: torch.distributions.MultivariateNormal,
    a_t: torch.Tensor,
    A_pi_thetak_given_st_at: torch.Tensor,
    config: PPOConfig
) -> torch.Tensor:
    """PPO 클립 손실 계산"""
    # Detach the old policy probabilities from the computation graph
    log_prob_thetak = pi_thetak_given_st.log_prob(a_t).detach()
    likelihood_ratio = torch.exp(pi_theta_given_st.log_prob(a_t) - log_prob_thetak)
    
    ppo_loss_per_example = -torch.minimum(
        likelihood_ratio * A_pi_thetak_given_st_at,
        torch.clip(likelihood_ratio, 1 - config.ppo_eps, 1 + config.ppo_eps) * A_pi_thetak_given_st_at,
    )
    
    return ppo_loss_per_example.mean()

def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[dict],
    action_batch: list[tuple[float, float]],
    advantage_batch: list[float],
    reward_to_go_batch: list[float],
    config: PPOConfig
) -> tuple[list[float], list[float]]:
    """PPO 학습 함수"""
    device = deviceof(critic)
    
    # 데이터를 텐서로 변환
    rgb_tensor, depth_tensor, goal_tensor = obs_batch_to_tensor(observation_batch, device)
    true_value_batch_tensor = torch.tensor(reward_to_go_batch, dtype=torch.float32, device=device)
    chosen_action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=device)
    advantage_batch_tensor = torch.tensor(advantage_batch, dtype=torch.float32, device=device)
    
    # Normalize advantages
    advantage_batch_tensor = (advantage_batch_tensor - advantage_batch_tensor.mean()) / (advantage_batch_tensor.std() + 1e-8)
    
    # 이전 정책의 행동 확률
    with torch.no_grad():
        old_policy_action_dist = actor.forward(rgb_tensor, depth_tensor, goal_tensor)
    
    # Actor and Critic 학습
    actor_losses = []
    critic_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        # Critic Update
        critic_optimizer.zero_grad()
        breakpoint() 
        
        pred_value_batch_tensor = critic.forward(rgb_tensor, depth_tensor, goal_tensor)
        critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)

        
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))

        # Actor Update
        actor_optimizer.zero_grad()
        current_policy_action_dist = actor.forward(rgb_tensor, depth_tensor, goal_tensor)
        actor_loss = compute_ppo_loss(
            old_policy_action_dist,
            current_policy_action_dist,
            chosen_action_tensor,
            advantage_batch_tensor,
            config
        )
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(float(actor_loss.item()))
    
    return actor_losses, critic_losses

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """학습률 설정"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from config import Config  

# --- 메인 학습 코드 (unchanged) ---
def main():
    # WandB 초기화
    # wandb.init(
    #     project="metaurban-rl-efficientnet",
    #     config={
    #         "train_epochs": 200,
    #         "episodes_per_batch": 16,
    #         "gamma": 0.99,
    #         "ppo_eps": 0.2,
    #         "ppo_grad_descent_steps": 10,
    #         "actor_lr": 3e-4,
    #         "critic_lr": 1e-3,
    #         "hidden_dim": 512,
    #     }
    # )
    # config = wandb.config
    
    config = Config()
    
    # 디바이스 설정
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 네트워크 초기화
    actor = Actor(hidden_dim=config.hidden_dim).to(device)
    critic = Critic(hidden_dim=config.hidden_dim).to(device)
    
    # 옵티마이저
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    
    # 정책
    policy = NNPolicy(actor)
    
    # 환경 초기화
    # env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)

    
    # PPO 설정
    ppo_config = PPOConfig(
        ppo_eps=config.ppo_eps,
        ppo_grad_descent_steps=config.ppo_grad_descent_steps,
    )
    
    # 학습 통계
    returns = []
    all_actor_losses = []
    all_critic_losses = []
    
    # 학습 루프
    for epoch in range(config.train_epochs):
        obs_batch = []
        act_batch = []
        rtg_batch = []
        adv_batch = []
        trajectory_returns = []
        
        # 배치 수집
        for episode in range(config.episodes_per_batch):
            print(f"Epoch {epoch}/{config.train_epochs}, Collecting Episode {episode+1}/{config.episodes_per_batch}...")
            obs_traj, act_traj, rew_traj = collect_trajectory(env, policy)
            rtg_traj = rewards_to_go(rew_traj, config.gamma)
            adv_traj = compute_advantage(critic, obs_traj, rew_traj, config.gamma)
            
            obs_batch.extend(obs_traj)
            act_batch.extend(act_traj)
            rtg_batch.extend(rtg_traj)
            adv_batch.extend(adv_traj)
            trajectory_returns.append(sum(rew_traj))
        
        # PPO 업데이트
        batch_actor_losses, batch_critic_losses = train_ppo(
            actor, critic, actor_optimizer, critic_optimizer,
            obs_batch, act_batch, adv_batch, rtg_batch, ppo_config
        )
        
        # 통계 수집
        returns.append(trajectory_returns)
        all_actor_losses.extend(batch_actor_losses)
        all_critic_losses.extend(batch_critic_losses)
        
        # 로깅
        avg_return = np.mean(trajectory_returns)
        std_return = np.std(trajectory_returns)
        median_return = np.median(trajectory_returns)
        
        print(f"Epoch {epoch}, Avg Returns: {avg_return:.3f} +/- {std_return:.3f}, "
            f"Median: {median_return:.3f}, Actor Loss: {np.mean(batch_actor_losses):.3f}, "
            f"Critic Loss: {np.mean(batch_critic_losses):.3f}")
        
        # WandB 로깅
        # wandb.log({
        #     "epoch": epoch,
        #     "avg_return": avg_return,
        #     "std_return": std_return,
        #     "median_return": median_return,
        #     "actor_loss": np.mean(batch_actor_losses),
        #     "critic_loss": np.mean(batch_critic_losses),
        # })
    
    # 모델 저장
    torch.save(actor.state_dict(), 'metaurban_actor_efficientnet.pt')
    torch.save(critic.state_dict(), 'metaurban_critic_efficientnet.pt')
    
    # 그래프 생성 및 저장
    create_and_save_plots(returns, all_actor_losses, all_critic_losses)
    
    env.close()


if __name__ == "__main__":
    main()