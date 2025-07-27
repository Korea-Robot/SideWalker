## train.py
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
import torchvision.models as models
import transformers
from transformers import SegformerForSemanticSegmentation

 
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Import configurations and utilities
from env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, create_and_save_plots, PDController

# Import PPO model
from perceptnet import PerceptNet
from model import Actor,Critic

from collections import defaultdict

# Reward Definition
def calculate_all_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """
    Goal-position based navigation을 위한 최적화된 리워드 함수
    분석 결과를 바탕으로 효과적인 네비게이션 학습을 위해 설계됨
    """
    rewards = defaultdict(float)
    
    agent_pos = env.agent.position
    agent_heading = env.agent.heading_theta
    speed = info.get('speed', 0)
    nav = env.agent.navigation
    waypoints = nav.checkpoints
    
    # 1. 목표 근접 보상 (Goal Proximity Reward)
    if prev_info:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        if current_dist < prev_dist:
            rewards['goal_proximity'] = (prev_dist - current_dist) * 10.0

    # 2. 지능적 체크포인트 진행 보상 (Smart Checkpoint Progress)
    if prev_info and 'closest_checkpoint_idx' in prev_info:
        prev_closest = prev_info.get('closest_checkpoint_idx', 0)
        current_closest = info.get('closest_checkpoint_idx', 0)
        
        if current_closest > prev_closest:
            progress_steps = current_closest - prev_closest
            # 연속적으로 여러 체크포인트를 통과하면 보너스
            bonus_multiplier = 1.0 + (progress_steps - 1) * 0.3
            rewards['checkpoint_progress'] = 8.0 * progress_steps * bonus_multiplier
    
    # 3. 방향 정렬 보상 (Directional Alignment) - 핵심 네비게이션
    if len(waypoints) > 0:
        # 다음 몇 개의 체크포인트를 고려한 방향 계산
        look_ahead = min(3, len(waypoints) - 1)
        if look_ahead > 0:
            target_pos = waypoints[look_ahead]
            direction_to_target = np.arctan2(target_pos[1] - agent_pos[1], 
                                           target_pos[0] - agent_pos[0])
            
            # 헤딩과 목표 방향의 일치도
            heading_diff = abs(agent_heading - direction_to_target)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            
            alignment_score = 1.0 - (heading_diff / np.pi)
            rewards['direction_alignment'] = max(0, alignment_score) * 2.0

    # 4. 성공 보상 (Success Reward)
    if info.get('arrive_dest', False):
        rewards['success_reward'] = 50.0

    # 5. 충돌 페널티 (Collision Penalty)
    if info.get('crash_vehicle', False) or info.get('crash_object', False):
        rewards['collision_penalty'] = -20.0
    
    # 6. 차선 이탈 페널티 (Out of Road Penalty)
    if info.get('out_of_road', False):
        rewards['out_of_road_penalty'] = -15.0

    # 7. 기본 환경 보상 (Environment's Default Reward)
    rewards['env_default_reward'] = info.get('original_reward', 0)

    return rewards


class DiscreteNNPolicy:
    """Discrete Neural Network Policy"""
    def __init__(self, actor: Actor):
        self.actor = actor
        
    def __call__(self, obs_data: dict) -> tuple[int, int]:
        device = deviceof(self.actor)
        
        # Convert observation to tensors
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        rgb = rgb.to(device)
        depth = depth.to(device)
        semantic = semantic.to(device)
        goal = goal.to(device)
        
        with torch.no_grad():
            self.actor.eval()
            (steering_idx, throttle_idx), (steering_val, throttle_val) = self.actor.sample_action(rgb, semantic, depth, goal)
            self.actor.train()
        
        return (int(steering_idx.cpu().item()), int(throttle_idx.cpu().item())), (float(steering_val.cpu().item()), float(throttle_val.cpu().item()))


# Load configurations
env_config = EnvConfig()

# PD controller for steering
pd_controller = PDController(p_gain=0.5, d_gain=0.3)

def collect_trajectory(env, policy: typing.Callable, max_steps: int = 512) -> tuple[list[dict], list[tuple[int, int]], list[float]]:
    """환경에서 trajectory 수집 - discrete action 버전"""
    observations = []
    actions = []  # 이제 discrete indices를 저장
    rewards = []
    
    obs, info = env.reset()
    waypoints = env.agent.navigation.checkpoints
    
    # 일정 거리 이상의 waypoints가 없을 경우, 환경을 재설정
    while len(waypoints) < 31:
        obs, info = env.reset()
        waypoints = env.agent.navigation.checkpoints
        
    step_count = 0
    prev_info = None
    
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
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        obs_data = {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic': semantic_data,
            'goal': ego_goal_position
        }
        observations.append(obs_data)
        
        # 행동 선택 (discrete indices와 continuous values 모두 받음)
        discrete_action, continuous_action = policy(obs_data)
        actions.append(discrete_action)  # discrete indices 저장
        
        target_angle, throttle = continuous_action

        # PD 제어를 통해 최종 steering 값 계산
        final_steering = pd_controller.get_control(target_angle, 0) 
        final_action = (final_steering, throttle)

        # 환경 스텝
        obs, env_reward, terminated, truncated, info = env.step(final_action)
        
        # 새로운 리워드 계산
        reward_dict = calculate_all_rewards(info, prev_info, continuous_action, env)
        total_reward = sum(reward_dict.values())
        rewards.append(total_reward)
        
        prev_info = info.copy()
        step_count += 1
        
        if terminated or truncated:
            break
    
    return observations, actions, rewards


def deviceof(m: nn.Module) -> torch.device:
    """모듈의 device 반환"""
    return next(m.parameters()).device


def obs_batch_to_tensor(obs_batch: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """관찰 배치를 텐서로 변환"""
    rgb_batch = []
    depth_batch = []
    semantic_batch = []
    goal_batch = []
    
    for obs in obs_batch:
        # RGB: (H, W, 3) -> (3, H, W)
        rgb = torch.tensor(obs['rgb'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        rgb_batch.append(rgb)
        
        # Depth: (H, W, 3) -> (3, H, W)
        depth = torch.tensor(obs['depth'], dtype=torch.float32).permute(2, 0, 1)
        depth_batch.append(depth)
        
        # semantic: (H, W, 3) -> (3, H, W)
        semantic = torch.tensor(obs['semantic'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        semantic_batch.append(semantic)
        
        # Goal: (2,)
        goal = torch.tensor(obs['goal'], dtype=torch.float32)
        goal_batch.append(goal)
    
    rgb_tensor = torch.stack(rgb_batch).to(device)
    depth_tensor = torch.stack(depth_batch).to(device)
    semantic_tensor = torch.stack(semantic_batch).to(device)
    goal_tensor = torch.stack(goal_batch).to(device)
    
    return rgb_tensor, depth_tensor, semantic_tensor, goal_tensor


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
    critic.eval()
    with torch.no_grad():
        rgb_tensor, depth_tensor, semantic_tensor, goal_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        obs_values = critic.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor).detach().cpu().numpy()
    critic.train()
    
    # Advantage = Reward-to-go - Value
    trajectory_advantages = np.array(rewards_to_go(trajectory_rewards, gamma)) - obs_values
    
    return list(trajectory_advantages)


@dataclass
class PPOConfig:
    ppo_eps: float
    ppo_grad_descent_steps: int


def compute_discrete_ppo_loss(
    old_steering_dist: torch.distributions.Categorical,
    old_throttle_dist: torch.distributions.Categorical,
    new_steering_dist: torch.distributions.Categorical,
    new_throttle_dist: torch.distributions.Categorical,
    steering_actions: torch.Tensor,
    throttle_actions: torch.Tensor,
    advantages: torch.Tensor,
    config: PPOConfig
) -> torch.Tensor:
    """Discrete PPO 클립 손실 계산"""
    # Calculate log probabilities
    old_steering_log_prob = old_steering_dist.log_prob(steering_actions).detach()
    old_throttle_log_prob = old_throttle_dist.log_prob(throttle_actions).detach()
    old_log_prob = old_steering_log_prob + old_throttle_log_prob
    
    new_steering_log_prob = new_steering_dist.log_prob(steering_actions)
    new_throttle_log_prob = new_throttle_dist.log_prob(throttle_actions)
    new_log_prob = new_steering_log_prob + new_throttle_log_prob
    
    # Calculate likelihood ratio
    likelihood_ratio = torch.exp(new_log_prob - old_log_prob)
    
    # PPO clipped loss
    ppo_loss_per_example = -torch.minimum(
        likelihood_ratio * advantages,
        torch.clip(likelihood_ratio, 1 - config.ppo_eps, 1 + config.ppo_eps) * advantages,
    )
    
    return ppo_loss_per_example.mean()


def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[dict],
    action_batch: list[tuple[int, int]],
    advantage_batch: list[float],
    reward_to_go_batch: list[float],
    config: PPOConfig
) -> tuple[list[float], list[float]]:
    """PPO 학습 함수 - discrete action 버전"""
    device = deviceof(critic)
    
    # 데이터를 텐서로 변환
    rgb_tensor, depth_tensor, semantic_tensor, goal_tensor = obs_batch_to_tensor(observation_batch, device)
    true_value_batch_tensor = torch.tensor(reward_to_go_batch, dtype=torch.float32, device=device)
    
    # Action tensors 생성
    steering_actions = torch.tensor([a[0] for a in action_batch], dtype=torch.long, device=device)
    throttle_actions = torch.tensor([a[1] for a in action_batch], dtype=torch.long, device=device)
    
    advantage_batch_tensor = torch.tensor(advantage_batch, dtype=torch.float32, device=device)
    
    # Normalize advantages
    advantage_batch_tensor = (advantage_batch_tensor - advantage_batch_tensor.mean()) / (advantage_batch_tensor.std() + 1e-8)
    
    # 이전 정책의 행동 확률
    with torch.no_grad():
        old_steering_dist, old_throttle_dist = actor.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
    
    # Actor and Critic 학습
    actor_losses = []
    critic_losses = []
    
    for _ in range(config.ppo_grad_descent_steps):
        # Critic Update
        critic_optimizer.zero_grad()
        pred_value_batch_tensor = critic.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
        critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))

        # Actor Update
        actor_optimizer.zero_grad()
        current_steering_dist, current_throttle_dist = actor.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
        actor_loss = compute_discrete_ppo_loss(
            old_steering_dist,
            old_throttle_dist,
            current_steering_dist,
            current_throttle_dist,
            steering_actions,
            throttle_actions,
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


def main():
    """메인 학습 함수"""
    
    config = Config()

    # episodes_per_batch 값을 16으로 변경하고 싶다면 여기서 수정할 수 있습니다.
    # config.episodes_per_batch = 16 # 현재는 8

    # 2. 생성된 config 객체를 wandb.init에 직접 전달
    wandb.init(
        project="metaurban-rl-multimodal-discrete",
        config=config  # 객체를 그대로 전달
    )
    
    # 디바이스 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 네트워크 초기화 (discrete action space)
    actor = Actor(hidden_dim=config.hidden_dim, num_steering_actions=5, num_throttle_actions=3).to(device)
    critic = Critic(hidden_dim=config.hidden_dim).to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in list(actor.parameters()) + list(critic.parameters()))
    trainable_params = sum(p.numel() for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 옵티마이저
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    
    # 정책 (discrete)
    policy = DiscreteNNPolicy(actor)
    
    # 환경 초기화
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
        act_batch = []  # discrete actions (indices)
        rtg_batch = []
        adv_batch = []
        trajectory_returns = []
        
        # 배치 수집
        for episode in range(config.episodes_per_batch):
            print(f"Epoch {epoch}/{config.train_epochs}, Collecting Episode {episode+1}/{config.episodes_per_batch}...")
            obs_traj, act_traj, rew_traj = collect_trajectory(env, policy, max_steps=config.max_steps)
            rtg_traj = rewards_to_go(rew_traj, config.gamma)
            adv_traj = compute_advantage(critic, obs_traj, rew_traj, config.gamma)
            
            obs_batch.extend(obs_traj)
            act_batch.extend(act_traj)
            rtg_batch.extend(rtg_traj)
            adv_batch.extend(adv_traj)
            trajectory_returns.append(sum(rew_traj))

        print('discrete action batch (indices):')
        print(act_batch[:10])  # 처음 10개만 출력
        print('reward batch:')
        print(rtg_batch[:10])  # 처음 10개만 출력
        
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
        wandb.log({
            "epoch": epoch,
            "avg_return": avg_return,
            "std_return": std_return,
            "median_return": median_return,
            "actor_loss": np.mean(batch_actor_losses),
            "critic_loss": np.mean(batch_critic_losses),
        })
        
        # 주기적으로 모델 저장
        if epoch % 10 == 0:
            torch.save(actor.state_dict(), f'checkpoints/metaurban_discrete_actor_epoch_{epoch}.pt')
            torch.save(critic.state_dict(), f'checkpoints/metaurban_discrete_critic_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
    
    # 최종 모델 저장
    torch.save(actor.state_dict(), 'metaurban_discrete_actor_multimodal_final.pt')
    torch.save(critic.state_dict(), 'metaurban_discrete_critic_multimodal_final.pt')
    print("Training completed! Final models saved.")
    
    # 그래프 생성 및 저장
    create_and_save_plots(returns, all_actor_losses, all_critic_losses)
    
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()