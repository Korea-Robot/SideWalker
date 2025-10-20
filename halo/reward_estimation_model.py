# reward_estimation_model.py

"""
RGB => DINOv2 => MHSA => patches embedding
Mask=> CNN => flatten => MLP => 

요소별 곱셈 (element-wise multiplication)" 또는 "Hadamard Product" 라고 하는 연산을 통해 '조절(modulation)'을 수행

=> latent Z => MLP => R(s,a)


How to define temperature


pLackett-Luce ranking likelihood & Loss

At = {a1,a2,...,an}
theta_i = R(s_t,a_t) : output MLP outputs. 
HALO의 볼츠만 기반 점수 : 이 후보들을 순서  σ(1)이 1등, σ(2)가 2등 …)로 랭킹합

\sum_i^N exp(theta_i)

# negative log likelihood

L_PL(theta,sigma) = 
"""

import torch
import torch.nn as nn
import math
from transformers import AutoModel
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- Constants based on the paper and DINOv2-base ---
EMBED_DIM = 768         # DinoV2 Base model's embedding dimension
NUM_PATCHES = 256       # (224 / 14)^2 = 16*16 = 256 patches
PATCH_GRID_SIZE = 16    # 16x16 grid of patches

class HALORewardModel(nn.Module):
    """
    HALO Reward Model Architecture as described in the paper.
    Given a visual observation (RGB image) and a candidate action (represented by its
    trajectory mask), this model outputs a scalar reward.
    """
    def __init__(self, n_attn_layers=2, n_heads=4, dropout=0.1, freeze_dino=True):
        super().__init__()
        
        # 1. Image Feature Extraction (Frozen DINOv2 + Self-Attention)
        self.dino_encoder = AutoModel.from_pretrained('facebook/dinov2-base') 
        if freeze_dino:
            for param in self.dino_encoder.parameters():
                param.requires_grad = False
        
        # Paper mentions "refined via Nsa Transformer layers"
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_attn_layers)
        ])

        # 2. Action-Conditioned Visual Feature Aggregation (Trajectory Mask CNN)
        # This lightweight CNN processes the trajectory mask to create a spatial relevance map.
        # Input: (B, 1, 32, 32) from dataset -> Output: (B, 1, 16, 16) relevance map
        self.trajectory_mask_cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),            
        )

        self.trajectory_mask_mlp = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
        )

        self.final_activation = nn.Sigmoid()
        
        # 3. Final Reward Prediction (MLP Head)
        # Maps the aggregated feature representation to a scalar reward.
        self.reward_mlp = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # # 4. additional Depth Estimation (MLP Head)
        # self.depth_mlp = nn.Sequential(
        #     nn.LayerN
        # )

        self.depth_patch_pool_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.ReLU(),
            nn.Linear(EMBED_DIM // 2, 1)
        )

        self.depth_mlp = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 16),
            nn.Sigmoid()
        )
        
        self.depth_activation = nn.Sigmoid()



    def forward(self, rgb_image: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_image (torch.Tensor): Batch of RGB images, shape (B, 3, 224, 224).
            trajectory_mask (torch.Tensor): Batch of trajectory masks, shape (B, 1, 32, 32).

        Returns:
            torch.Tensor: Scalar reward for each image-mask pair, shape (B, 1).
        """
        # --- Image Feature Extraction ---
        # Get patch embeddings from DINOv2. Output shape: (B, 257, 768) including CLS token.
        dino_outputs = self.dino_encoder(rgb_image).last_hidden_state
        
        # We only use patch embeddings, excluding the [CLS] token at index 0.
        patch_embeddings = dino_outputs[:, 1:] # Shape: (B, 256, 768)

        # Refine patch embeddings with self-attention
        refined_embeddings = patch_embeddings
        for layer in self.self_attention_layers:
            refined_embeddings = layer(refined_embeddings)

        # --- Action-Conditioned Aggregation ---
        # Generate spatial relevance map from the trajectory mask.
        # Shape: (B, 1, 32, 32)
        # 1. CNN을 통해 공간적 특징(raw logit)을 추출합니다.
        spatial_relevance_map = self.trajectory_mask_cnn(trajectory_mask) # Shape: (B, 1, 16, 16)
        
        
        # 2. 맵을 1D 벡터로 펼칩니다.
        cnn_feature_vector = spatial_relevance_map.view(-1, NUM_PATCHES) # Shape: (B, 256)

        # 3. MLP를 통해 CNN 특징을 한 번 더 정제합니다.
        mlp_feature_vector = self.trajectory_mask_mlp(cnn_feature_vector) # Shape: (B, 256)

        # 4. 잔차 연결: 원본 CNN 특징과 MLP가 정제한 특징을 더합니다.
        #    (unbounded 값) + (unbounded 값) = (unbounded 값)
        combined_vector = cnn_feature_vector + mlp_feature_vector
        
        # 5. 마지막에 Sigmoid 활성화 함수를 적용하여 최종 가중치 벡터를 0~1 사이로 정규화합니다.
        #    이것이 최종적인 '공간적 연관성 가중치'가 됩니다.
        weighting_vector = self.final_activation(combined_vector) # Shape: (B, 256)

        # 6. 요소별 곱셈으로 이미지 패치들을 '조절(modulate)'합니다.
        #    (B, 256, 768) * (B, 256, 1) -> 안정적인 연산
        modulated_embeddings = refined_embeddings * weighting_vector.unsqueeze(-1)
        
        # 7. 조절된 특징들을 합산하여 최종 특징 벡터를 생성합니다.
        aggregated_features = modulated_embeddings.sum(dim=1)
        
        # --- 최종 보상 예측 ---
        reward = self.reward_mlp(aggregated_features)
        
        
        # Depth Estimation 
        # --- Depth Estimation (action 결합 이전) ---
        # refined_embeddings: (B, 256, 768)
        attn_logits = self.depth_patch_pool_proj(refined_embeddings)   # (B,256,1)
        attn_weights = torch.softmax(attn_logits, dim=1)               # (B,256,1)
        depth_feature = (refined_embeddings * attn_weights).sum(dim=1) # (B,768)
        depth = self.depth_mlp(depth_feature)                          # (B,16)        

        return reward,depth

if __name__ == '__main__':
    # --- Test the model with dummy data ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a model instance
    model = HALORewardModel().to(device)
    model.eval()

    # Create dummy input tensors
    batch_size = 4
    dummy_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_mask = torch.rand(batch_size, 1, 32, 32).to(device)

    # Perform a forward pass
    with torch.no_grad():
        predicted_reward,predicted_depth = model(dummy_rgb, dummy_mask)

    print(f"Model instantiated on: {device}")
    print(f"Input RGB shape: {dummy_rgb.shape}")
    print(f"Input Mask shape: {dummy_mask.shape}")
    print(f"Output Reward shape: {predicted_reward.shape}")
    print(f"Sample rewards:\n{predicted_reward.squeeze()}")
    print(f"output depth shape: {predicted_depth}")
    
    # Check number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"DINOv2 is {'NOT frozen' if model.dino_encoder.training else 'FROZEN'}")


    # breakpoint()
    
    # return {
    #     "rgb": rgb_tensor,
    #     "depth": depth_tensor,
    #     "distance_target": distance_target,
    #     "local_goal_position": torch.from_numpy(local_goal_position).float(),
    #     "trajectory": torch.from_numpy(interpolated_trajectory).float(),
    #     "avg_action_to_goal": torch.from_numpy(avg_action_to_goal).float(),
    #     "trajectory_mask_2s": torch.from_numpy(trajectory_mask_2s).float().unsqueeze(0),
    #     "next_action_linear_discretized": torch.tensor(action_linear_x, dtype=torch.long),
    #     "next_action_angular_discretized": torch.tensor(action_angular_z, dtype=torch.long),
    # }
    

    DATA_DIR = '../../data/ilrl/0903_inside_night'
    DATA_DIR = '../../data/ilrl/data/20250925_1715'
    RGB_SUBFOLDER = 'images/realsense_color'
    DEPTH_SUBFOLDER = 'depth/realsense_depth'
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    from reward_estimation_dataset import NavigationDataset

    dataset = NavigationDataset(
        data_root=DATA_DIR,
        transform=image_transforms
    )
    if len(dataset) > 0:
        print(f"\n--- Dataset Initialized Successfully: {len(dataset)} samples ---")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        print("\n--- Fetching a batch of data... (This should now work without error) ---")
        batch = next(iter(dataloader))
        print("\n--- Batch Shapes ---")

        
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        mask = batch['trajectory_mask_2s'].to(device)
    
    # Perform a forward pass
    with torch.no_grad():
        predicted_reward,predicted_depth = model(rgb, mask)

    print(f"Model instantiated on: {device}")
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input Mask shape: {mask.shape}")
    print(f"Output Reward shape: {predicted_reward.shape}")
    print(f"Sample rewards:\n{predicted_reward.squeeze()}")
    print(f"output depth shape: {predicted_depth}")
    
    # Check number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"DINOv2 is {'NOT frozen' if model.dino_encoder.training else 'FROZEN'}")
