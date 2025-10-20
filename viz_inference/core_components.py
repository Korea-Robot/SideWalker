# 파일명: core_components.py
# 설명: 환경 설정, PyTorch 모델, 공용 유틸리티 함수를 정의합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 환경 설정 ---
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
    horizon=1000,
    vehicle_config=dict(enable_reverse=True, image_source="rgb_camera"),
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
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    stack_size=1,
    log_level=50,
)

def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계의 목표 지점을 에이전트 중심의 자기 좌표계로 변환"""
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    cos_h, sin_h = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    ego_vector = rotation_matrix @ vec_in_world
    return ego_vector

def extract_sensor_data(obs):
    """관찰에서 센서 데이터 추출"""
    # image 데이터에서 RGB 추출 (마지막 프레임 사용)
    if 'image' in obs:
        rgb_data = obs['image'][..., -3:].squeeze(-1)
        rgb_data = (rgb_data * 255).astype(np.uint8)
    else:
        rgb_data = None
    
    # 현재 설정에서는 depth와 semantic이 image에 포함되지 않으므로 None으로 처리
    depth_data = None
    semantic_data = None
    
    return rgb_data, depth_data, semantic_data


class Actor(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=2):
        super().__init__()
        # RGB 처리 (3채널)
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.rgb_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.rgb_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Depth 처리 (1채널)
        self.depth_conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.depth_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.depth_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 특징 융합
        self.fc1 = nn.Linear(64 * 28 * 16 * 2 + 2, hidden_dim) # 크기 계산 확인 필요
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.distributions.MultivariateNormal:
        batch_size = rgb.shape[0]
        
        # RGB 특징 추출
        rgb_x = F.relu(self.rgb_conv1(rgb))
        rgb_x = F.relu(self.rgb_conv2(rgb_x))
        rgb_x = F.relu(self.rgb_conv3(rgb_x))
        rgb_x = rgb_x.view(batch_size, -1)
        
        # Depth 특징 추출
        depth_x = F.relu(self.depth_conv1(depth))
        depth_x = F.relu(self.depth_conv2(depth_x))
        depth_x = F.relu(self.depth_conv3(depth_x))
        depth_x = depth_x.view(batch_size, -1)
        
        # 특징 융합
        x = torch.cat([rgb_x, depth_x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        
        sigma = 0.1 * torch.ones_like(mu)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))