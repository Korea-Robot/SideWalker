import numpy as np
import os
import pygame
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections import deque
import random

# --- PerceptNet Code ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
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
    def __init__(self, layers, block=BasicBlock, num_classes=1000, groups=1,
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
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done, goal):
        self.buffer.append((state, action, log_prob, reward, next_state, done, goal))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Model Architecture ---
class ActorCritic(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=2):
        super(ActorCritic, self).__init__()
        self.perceptnet = PerceptNet(layers=[2, 2, 2, 2], block=BasicBlock, num_classes=hidden_dim)
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, hidden_dim)
        self.actor_fc = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.critic_fc = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, rgb, depth, goal):
        depth_features = self.perceptnet(depth)
        rgb_features = self.efficientnet(rgb)
        x = torch.cat([rgb_features, depth_features, goal], dim=1)
        actor_x = F.relu(self.actor_fc(x))
        mu = torch.tanh(self.mu(actor_x))
        log_std = self.log_std(actor_x)
        std = torch.exp(log_std)
        critic_x = F.relu(self.critic_fc(x))
        value = self.value(critic_x)
        return mu, std, value

    def get_action(self, rgb, depth, goal):
        mu, std, _ = self.forward(rgb, depth, goal)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().cpu().numpy(), log_prob.detach()

# --- PPO-CLIP ---
def ppo_update(policy, optimizer, batch, device, clip_param=0.2, ppo_epochs=10, gamma=0.99, gae_lambda=0.95):
    states, actions, old_log_probs, rewards, next_states, dones, goals = zip(*batch)

    rgb_list = [torch.FloatTensor(s['image'][..., -1]).permute(2, 0, 1) for s in states]
    depth_list = [torch.FloatTensor(s['depth'][..., -1]).unsqueeze(0) for s in states]
    rgbs = torch.stack(rgb_list).to(device)
    depths = torch.stack(depth_list).to(device)
    goals = torch.FloatTensor(np.array(goals)).to(device)
    
    actions = torch.FloatTensor(np.array(actions)).squeeze(1).to(device)
    old_log_probs = torch.stack(list(old_log_probs)).squeeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    with torch.no_grad():
        _, _, values = policy(rgbs, depths, goals)
        next_rgb_list = [torch.FloatTensor(s['image'][..., -1]).permute(2, 0, 1) for s in next_states]
        next_depth_list = [torch.FloatTensor(s['depth'][..., -1]).unsqueeze(0) for s in next_states]
        next_rgbs = torch.stack(next_rgb_list).to(device)
        next_depths = torch.stack(next_depth_list).to(device)
        _, _, next_values = policy(next_rgbs, next_depths, goals)
        
        deltas = rewards + gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            advantages[t] = deltas[t] + gamma * gae_lambda * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        returns = advantages + values.squeeze()

    for _ in range(ppo_epochs):
        mu, std, value_pred = policy(rgbs, depths, goals)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(axis=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(value_pred.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 환경 설정
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True, map='X', manual_control=False, crswalk_density=1, object_density=0.1,
    walk_on_all_regions=False, drivable_area_extension=55, height_scale=1, horizon=1000,
    vehicle_config=dict(enable_reverse=True), show_sidewalk=True, show_crosswalk=True,
    random_lane_width=True, random_agent_model=True, random_lane_num=True, 
    random_spawn_lane_index=False, num_scenarios=100, accident_prob=0, max_lateral_dist=5.0,
    agent_type='coco', relax_out_of_road_done=False, agent_observation=ThreeSourceMixObservation,
    image_observation=True, sensors={"rgb_camera": (RGBCamera, *SENSOR_SIZE), "depth_camera": (DepthCamera, *SENSOR_SIZE), "semantic_camera": (SemanticCamera, *SENSOR_SIZE)},
    log_level=50,
)

def get_ego_goal_position(agent, k=15):
    nav = agent.navigation
    waypoints = nav.checkpoints
    if len(waypoints) > k:
        global_target = waypoints[k]
        vec_in_world = global_target - agent.position
        theta = -agent.heading_theta
        cos_h, sin_h = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return rotation_matrix @ vec_in_world
    return np.array([0.0, 0.0])

# --- 메인 실행 로직 ---
env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
policy = ActorCritic().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
replay_buffer = ReplayBuffer(2048)
batch_size = 128

running = True
try:
    for i in range(1000):
        obs, info = env.reset(seed=i + 1)
        done = False
        episode_reward = 0
        
        while not done:
            ego_goal_position = get_ego_goal_position(env.agent)

            rgb = torch.FloatTensor(obs['image'][..., -1]).permute(2, 0, 1).unsqueeze(0).to(device)
            depth = torch.FloatTensor(obs['depth'][..., -1]).unsqueeze(0).unsqueeze(0).to(device)
            goal = torch.FloatTensor(ego_goal_position).unsqueeze(0).to(device)
            
            action, log_prob = policy.get_action(rgb, depth, goal)
            
            next_obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            replay_buffer.push(obs, action, log_prob.cpu(), reward, next_obs, done, ego_goal_position)
            obs = next_obs
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                ppo_update(policy, optimizer, batch, device)

            env.render(text={"Episode": i, "Reward": f"{episode_reward:.2f}", "Ego Goal": np.round(ego_goal_position, 2)})
            if done:
                print(f"Episode {i} finished. Reward: {episode_reward}")
                break
finally:
    env.close()
    pygame.quit()
