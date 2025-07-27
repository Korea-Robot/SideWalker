import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models #  --- MODIFIED: Added torchvision for EfficientNet ---
from perceptnet import PerceptNet


import torch
from PIL import Image
import requests
from torchvision import transforms

# 1. DINOv2 모델 불러오기 (ViT-Small, 14x14 패치 크기)
# 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14' 등 다양한 모델 선택 가능
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg') # 55 fps 
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg') # 40 fps 


# --- 네트워크 정의 (MODIFIED) ---
class Encoder(nn.Module):
    def __init__(self,hidden_dim=512):
        super().__init__()
        
        # --- RGB Encoder: EfficientNet-B0 (pre-trained) ---
        # We use the feature extractor part and remove the final classifier
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # The feature extractor part of efficientnet
        self.rgb_encoder = efficientnet.features

        
        # --- Dino v2 Encoder : feature
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg') # 40 fps 
        
        # The output of features is (batch, 1280, H/32, W/32). We pool it to (batch, 1280).
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Depth Encoder: PerceptNet ---
        self.depth_encoder = PerceptNet(layers=[2, 2, 2, 2]) # As per the example
        # The output of PerceptNet is (batch, 512, H/32, W/32). We pool it to (batch, 512).
        self.depth_pool = nn.AdaptiveAvgPool2d((1, 1))
        

        self.goal_mlp = nn.Sequential(
            nn.Linear(2,256),
            nn.ELU(),
            nn.Linear(256,256),
            nn.ELU(),
            nn.Linear(256,256)
        )
        # --- Feature Fusion ---
        # EfficientNet-B0 feature size = 1280
        # PerceptNet feature size = 512
        # Goal vector size = 2

        fusion_dim = 1280 + 768 + 512 + 256
        
        self.fc1 = nn.Linear(fusion_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.distributions.MultivariateNormal:
        batch_size = rgb.shape[0]
        
        # RGB Feature Extraction
        rgb_features = self.rgb_encoder(rgb)
        rgb_features = self.rgb_pool(rgb_features)
        rgb_features = rgb_features.view(batch_size, -1)
        
        dino_features = self.dino_encoder(rgb)
        dino_features = dino_features.view(batch_size,-1)
        
        # Depth Feature Extraction
        depth_features = self.depth_encoder(depth)
        depth_features = self.depth_pool(depth_features)
        depth_features = depth_features.view(batch_size, -1)
        
        goal_features = self.goal_mlp(goal)
        # Feature Fusion
        latent_z = torch.cat([rgb_features, dino_features,depth_features, goal_features], dim=1)
        latent_z = F.relu(self.fc1(latent_z))
        latent_z = F.relu(self.fc2(latent_z))
        
        return latent_z 
    
    
class Actor(nn.Module):
    def __init__(self, encoder: Encoder,hidden_dim=512, output_dim=2):
        super().__init__()

        self.encoder = encoder
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.distributions.MultivariateNormal:

        z = self.encoder(rgb,depth,goal)

        mu = self.fc3(z)
        # Fixed standard deviation
        std = F.softplus(self.fc4(z))
        sigma = 0.1 * std
        
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma)),mu,std 

class Critic(nn.Module):
    def __init__(self,encoder: Encoder, hidden_dim=512):
        super().__init__()
        
        self.encoder = encoder
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.encoder(rgb,depth,goal) 
        x = F.relu(self.fc2(z))
        value = self.fc3(z)
        
        return torch.squeeze(value, dim=1)


if __name__=="__main__":
    batch = 4 
    
    encoder = Encoder()
    rgb  = torch.randn(batch,3,224,224)
    depth = torch.randn(batch,3,224,224)
    goal = torch.randn(batch,2)
    
    actor = Actor(encoder=encoder)
    critic = Critic(encoder=encoder)
    
    y,mu,std = actor(rgb,depth,goal)
    yy = critic(rgb,depth,goal)
    
    breakpoint()
    
    