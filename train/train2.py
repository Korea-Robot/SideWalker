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
import torchvision.models as models # Added import

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 환경 설정 ---
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=False,  # 학습 시에는 렌더링 비활성화
    map='X',
    manual_control=False,
    crswalk_density=1,
    object_density=0.1,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=1000,
    
    vehicle_config=dict(enable_reverse=True),
    
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    
    random_spawn_lane_index=False,
    num_scenarios=100,
    accident_prob=0,
    max_lateral_dist=5.0,
    
    agent_type='coco',
    
    relax_out_of_road_done=False,
    
    agent_observation=ThreeSourceMixObservation,
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# --- 유틸리티 함수 ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계의 목표 지점을 에이전트 중심의 자기 좌표계로 변환"""
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])
    
    ego_vector = rotation_matrix @ vec_in_world
    return ego_vector

def extract_sensor_data(obs):
    """관찰에서 센서 데이터 추출"""
    if 'image' in obs:
        rgb_data = obs['image'][..., -1]
        rgb_data = (rgb_data * 255).astype(np.uint8)
    else:
        rgb_data = None
    
    # depth 1 => 3채널로 확장
    depth_data = obs["depth"][..., -1]
    depth_data = np.concatenate([depth_data,depth_data,depth_data], axis=-1)
    
    semantic_data = obs["semantic"][..., -1]

    return rgb_data, depth_data, semantic_data

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
        rgb = torch.tensor(obs['rgb'], dtype=torch.float32).permute(2, 0, 1) / 255.0
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

# --- PerceptNet 정의 (제공된 코드) ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PerceptNet(nn.Module):

    def __init__(self, layers, block=BasicBlock, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
                 
        super(PerceptNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# --- 네트워크 정의 (수정됨) ---
class Actor(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=2):
        super().__init__()
        
        # RGB Encoder: torchvision EfficientNet-B0
        self.rgb_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        
        # Depth Encoder: PerceptNet
        self.depth_encoder = PerceptNet(layers=[2, 2, 2, 2], norm_layer=nn.BatchNorm2d)
        
        # Use Adaptive Average Pooling to create a fixed-size feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature fusion and policy head
        # EfficientNet-B0 features: 1280, PerceptNet features: 512, Goal: 2
        fusion_input_dim = 1280 + 512 + 2 
        self.fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.distributions.MultivariateNormal:
        # RGB feature extraction
        rgb_x = self.rgb_encoder(rgb)
        rgb_x = self.avgpool(rgb_x)
        rgb_x = torch.flatten(rgb_x, 1)
        
        # Depth feature extraction
        depth_x = self.depth_encoder(depth)
        depth_x = self.avgpool(depth_x)
        depth_x = torch.flatten(depth_x, 1)
        
        # Feature fusion
        x = torch.cat([rgb_x, depth_x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        
        # Fixed standard deviation for exploration
        sigma = 0.1 * torch.ones_like(mu)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))

class Critic(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # RGB Encoder: torchvision EfficientNet-B0
        self.rgb_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        
        # Depth Encoder: PerceptNet
        self.depth_encoder = PerceptNet(layers=[2, 2, 2, 2], norm_layer=nn.BatchNorm2d)

        # Use Adaptive Average Pooling to create a fixed-size feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Value prediction head
        # EfficientNet-B0 features: 1280, PerceptNet features: 512, Goal: 2
        fusion_input_dim = 1280 + 512 + 2
        self.fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # RGB feature extraction
        rgb_x = self.rgb_encoder(rgb)
        rgb_x = self.avgpool(rgb_x)
        rgb_x = torch.flatten(rgb_x, 1)
        
        # Depth feature extraction
        depth_x = self.depth_encoder(depth)
        depth_x = self.avgpool(depth_x)
        depth_x = torch.flatten(depth_x, 1)
        
        # Feature fusion
        x = torch.cat([rgb_x, depth_x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return torch.squeeze(value, dim=1)

# --- NNPolicy 및 PPO 관련 함수들 (변경 없음) ---
class NNPolicy:
    def __init__(self, net: Actor):
        self.net = net

    def __call__(self, obs: dict) -> tuple[float, float]:
        """관찰을 받아 행동을 반환"""
        rgb_tensor, depth_tensor, goal_tensor = obs_batch_to_tensor([obs], deviceof(self.net))
        
        with torch.no_grad():
            action = self.net(rgb_tensor, depth_tensor, goal_tensor).sample()[0]
        # Actions are typically throttle/brake and steering. Clip them to valid ranges.
        throttle_brake = torch.clip(action[0], -1.0, 1.0)
        steering = torch.clip(action[1], -1.0, 1.0)
        return steering.item(), throttle_brake.item()

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
    with torch.no_grad():
        rgb_tensor, depth_tensor, goal_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        obs_values = critic.forward(rgb_tensor, depth_tensor, goal_tensor).detach().cpu().numpy()
    
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
    likelihood_ratio = torch.exp(pi_theta_given_st.log_prob(a_t) - pi_thetak_given_st.log_prob(a_t))
    
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
    # Corrected action tensor order to match (steering, throttle)
    chosen_action_tensor = torch.tensor([(s, t) for s, t in action_batch], device=device)
    advantage_batch_tensor = torch.tensor(advantage_batch, device=device)
    
    # Critic 학습
    critic_optimizer.zero_grad()
    pred_value_batch_tensor = critic.forward(rgb_tensor, depth_tensor, goal_tensor)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
    critic_loss.backward()
    critic_optimizer.step()
    
    # 이전 정책의 행동 확률
    with torch.no_grad():
        old_policy_action_probs = actor.forward(rgb_tensor, depth_tensor, goal_tensor)
    
    # Actor 학습
    actor_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
        current_policy_action_probs = actor.forward(rgb_tensor, depth_tensor, goal_tensor)
        actor_loss = compute_ppo_loss(
            old_policy_action_probs,
            current_policy_action_probs,
            chosen_action_tensor,
            advantage_batch_tensor,
            config
        )
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(float(actor_loss))
    
    return actor_losses, [float(critic_loss)] * config.ppo_grad_descent_steps

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """학습률 설정"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --- 메인 학습 코드 (변경 없음) ---
def main():
    # WandB 초기화
    wandb.init(
        project="metaurban-rl-efficientnet",
        config={
            "train_epochs": 200,
            "episodes_per_batch": 16,
            "gamma": 0.99,
            "ppo_eps": 0.2,
            "ppo_grad_descent_steps": 10,
            "actor_lr": 3e-4,
            "critic_lr": 1e-3,
            "hidden_dim": 512,
        }
    )
    
    config = wandb.config
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    
    # PPO 설정
    ppo_config = PPOConfig(
        ppo_eps=config.ppo_eps,
        ppo_grad_descent_steps=config.ppo_grad_descent_steps,
    )
    
    # 학습 통계
    returns = []
    actor_losses = []
    critic_losses = []
    
    # 학습 루프
    for epoch in range(config.train_epochs):
        obs_batch = []
        act_batch = []
        rtg_batch = []
        adv_batch = []
        trajectory_returns = []
        
        # 배치 수집
        for episode in range(config.episodes_per_batch):
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
        actor_losses.extend(batch_actor_losses)
        critic_losses.extend(batch_critic_losses)
        
        # 로깅
        avg_return = np.mean(trajectory_returns)
        std_return = np.std(trajectory_returns)
        median_return = np.median(trajectory_returns)
        
        print(f"Epoch {epoch}, Avg Returns: {avg_return:.3f} +/- {std_return:.3f}, "
              f"Median: {median_return:.3f}, Actor Loss: {batch_actor_losses[-1]:.3f}, "
              f"Critic Loss: {batch_critic_losses[-1]:.3f}")
        
        # WandB 로깅
        wandb.log({
            "epoch": epoch,
            "avg_return": avg_return,
            "std_return": std_return,
            "median_return": median_return,
            "actor_loss": batch_actor_losses[-1],
            "critic_loss": batch_critic_losses[-1],
        })
    
    # 모델 저장
    torch.save(actor.state_dict(), 'metaurban_actor_efficientnet.pt')
    torch.save(critic.state_dict(), 'metaurban_critic_efficientnet.pt')
    
    # 그래프 생성 및 저장
    create_and_save_plots(returns, actor_losses, critic_losses)
    
    env.close()
    wandb.finish()

def create_and_save_plots(returns, actor_losses, critic_losses):
    """학습 결과 그래프 생성 및 저장"""
    
    # Returns 그래프
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    return_means = [np.mean(returns[i]) for i in range(len(returns))]
    return_medians = [np.median(returns[i]) for i in range(len(returns))]
    return_stds = [np.std(returns[i]) for i in range(len(returns))]
    
    plt.plot(return_means, label="Mean", color='blue')
    plt.plot(return_medians, label="Median", color='red')
    plt.fill_between(range(len(return_means)), 
                     np.array(return_means) - np.array(return_stds), 
                     np.array(return_means) + np.array(return_stds), 
                     alpha=0.3, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.title("Training Returns")
    plt.legend()
    plt.grid(True)
    
    # Actor Loss 그래프
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses, label="Actor Loss", color='green')
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.legend()
    plt.grid(True)
    
    # Critic Loss 그래프
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses, label="Critic Loss", color='orange')
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metaurban_training_results_efficientnet.png', dpi=300, bbox_inches='tight')
    
    # WandB에 이미지 업로드
    try:
        wandb.log({"training_plots": wandb.Image(plt)})
    except Exception as e:
        print(f"Could not log plots to W&B: {e}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    xs = []
    ys = []
    for t, rets in enumerate(returns):
        for ret in rets:
            xs.append(t)
            ys.append(ret)
    plt.scatter(xs, ys, alpha=0.5, s=10)
    plt.xlabel("Epoch")
    plt.ylabel("Episode Return")
    plt.title("Episode Returns Scatter Plot")
    plt.grid(True)
    plt.savefig('metaurban_returns_scatter_efficientnet.png', dpi=300, bbox_inches='tight')
    
    try:
        wandb.log({"returns_scatter": wandb.Image(plt)})
    except Exception as e:
        print(f"Could not log scatter plot to W&B: {e}")
    
    plt.close('all')

if __name__ == "__main__":
    main()