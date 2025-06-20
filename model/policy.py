import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


# from categoric


class Policy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.rgb_backbone  = nn.Sequential() # torchvision or timm Efficientnet-B0
        
        self.depth_backbone = nn.Sequential() # Planer for Depth image which is alpha is 1 channel.
        
        self.proj = nn.Sequential() ## Residual Block please
        self.v = nn.Sequential()
        
        self.pi= nn.Sequential()
        
        
    def forward(self,segment,depth): # 일단 goal 없이 해보자.
        
        segment = self.rgb_backbone(segment)
        depth   = self.depth_backbone(depth) 
        
        # concat to x = torch.tesnfor =>  
        
        # concat and projection to x 
        
        x 
        value = self.v(x)
        action_prob = self.pi(x)
        return value,action_prob
        
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out + identity)


class Policy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # RGB backbone: EfficientNet-B0 feature extractor
        base_rgb = models.efficientnet_b0(pretrained=True)
        self.rgb_backbone = nn.Sequential(*list(base_rgb.features.children()),
                                          nn.AdaptiveAvgPool2d(1))
        rgb_feat_dim = base_rgb.classifier[1].in_features  # typically 1280

        # Depth backbone: simple ConvNet for single-channel depth
        self.depth_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        depth_feat_dim = 64

        # Projection Residual Block
        proj_dim = 512
        concat_dim = rgb_feat_dim + depth_feat_dim
        self.proj_fc = nn.Linear(concat_dim, proj_dim)
        self.proj = ResidualBlock(proj_dim)

        # Value head
        self.v = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Policy head (discrete actions)
        self.pi = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Softmax(dim=-1)
        )

    def forward(self, segment: torch.Tensor, depth: torch.Tensor):
        # segment: [B, 3, H, W], depth: [B, 1, H, W]
        rgb_feat = self.rgb_backbone(segment).view(segment.size(0), -1)
        depth_feat = self.depth_backbone(depth).view(depth.size(0), -1)

        x = torch.cat([rgb_feat, depth_feat], dim=-1)
        x = F.relu(self.proj_fc(x))
        x = self.proj(x)

        value = self.v(x)
        action_prob = self.pi(x)
        return value, action_prob

# Example instantiation:
# model = Policy(action_dim=5)  # if you have 5 discrete actions
 