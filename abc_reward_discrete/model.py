# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation

from config import Config
from perceptnet import PerceptNet

class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that processes RGB, semantic, and depth images
    through different frozen encoders and fuses their features effectively.
    """
    def __init__(self, hidden_dim=512, fusion_strategy='attention'):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy
        
        # Initialize encoders
        self._init_rgb_encoders()
        self._init_semantic_encoder()  
        self._init_depth_encoder()
        self._init_goal_encoder()
        self._init_fusion_layers()
        
    def _init_rgb_encoders(self):
        """Initialize RGB encoders: DINOv2 and SegFormer"""
        print("Initializing RGB encoders...")
        
        # 1. DINOv2 for RGB feature extraction
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        self.dino_encoder.eval()
        
        # 2. SegFormer for RGB spatial understanding
        id2label = {0: 'background', 1: 'caution_zone', 2: 'bike_lane', 
                   3: 'alley', 4: 'roadway', 5: 'braille_guide_blocks', 6: 'sidewalk'}
        label2id = {v: k for k, v in id2label.items()}
        
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id
        )
        
        # Load pre-trained weights if available
        try:
            seg_model_path = "./models/seg_model_epoch_70.pth"
            state_dict = torch.load(seg_model_path, map_location='cpu')
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.segformer.load_state_dict(new_state_dict)
            print(f"Loaded SegFormer weights from: {seg_model_path}")
        except:
            print("Using default SegFormer weights")
            
        for param in self.segformer.parameters():
            param.requires_grad = False
        self.segformer.eval()
        
        # RGB feature projectors
        self.dino_projector = nn.Sequential(
            nn.Linear(768, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # SegFormer feature extraction from encoder
        self.segformer_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim),  # mit-b0 last hidden state channel
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_semantic_encoder(self):
        """Initialize semantic segmentation encoder: EfficientNet"""
        print("Initializing semantic encoder...")
        
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.semantic_encoder = efficientnet.features
        
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False
        self.semantic_encoder.eval()
        
        self.semantic_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_depth_encoder(self):
        """Initialize depth encoder: PerceptNet"""
        print("Initializing depth encoder...")
        
        try:
            depth_weights_path = "./models/plannernet.pt"
            net_loaded, _ = torch.load(depth_weights_path, map_location=self.device, weights_only=False)
            self.depth_encoder = net_loaded.encoder
            print(f"Loaded depth encoder from: {depth_weights_path}")
        except:
            print("Using default PerceptNet")
            self.depth_encoder = PerceptNet(layers=[2, 2, 2, 2])
            
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        self.depth_encoder.eval()
        
        self.depth_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_goal_encoder(self):
        """Initialize goal encoder"""
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_fusion_layers(self):
        """Initialize feature fusion layers"""
        if self.fusion_strategy == 'attention':
            # Multi-head attention for feature fusion
            self.feature_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Learnable query for feature aggregation
            self.query_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            
        # Final fusion layers
        if self.fusion_strategy == 'attention':
            fusion_input_dim = self.hidden_dim
        else:
            fusion_input_dim = self.hidden_dim * 4  # rgb_dino, semantic, depth, goal
            
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, rgb, semantic, depth, goal):
        batch_size = rgb.shape[0]
        
        # DINOv2 expects input dimensions to be a multiple of its patch size (14).
        # We resize the input RGB image to a compatible size like 224x224.
        rgb_resized = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract features from each modality
        # RGB features via DINOv2
        with torch.no_grad():
            # Use the resized image for the DINO encoder
            dino_features = self.dino_encoder(rgb_resized)  # [B, 768]
        dino_features = self.dino_projector(dino_features)  # [B, hidden_dim]
        
        # RGB spatial features via SegFormer (uses original image)
        with torch.no_grad():
            segformer_output = self.segformer.segformer(rgb)
            segformer_features = segformer_output['last_hidden_state']  # [B, 256, H/4, W/4]
        segformer_features = self.segformer_projector(segformer_features)  # [B, hidden_dim]
        
        # Combine RGB features (DINOv2 + SegFormer)
        rgb_combined = (dino_features + segformer_features) / 2
        
        # Semantic features via EfficientNet
        with torch.no_grad():
            semantic_features = self.semantic_encoder(semantic)  # [B, 1280, H/32, W/32]
        semantic_features = self.semantic_projector(semantic_features)  # [B, hidden_dim]
        
        # Depth features via PerceptNet
        with torch.no_grad():
            depth_features = self.depth_encoder(depth)  # [B, 512, H/32, W/32]
        depth_features = self.depth_projector(depth_features)  # [B, hidden_dim]
        
        # Goal features
        goal_features = self.goal_encoder(goal)  # [B, hidden_dim]
        
        # Feature fusion
        if self.fusion_strategy == 'attention':
            features = torch.stack([rgb_combined, semantic_features, depth_features, goal_features], dim=1)
            query = self.query_token.expand(batch_size, -1, -1)
            fused_features, _ = self.feature_attention(query, features, features)
            fused_features = fused_features.squeeze(1)
        else:  # simple concatenation
            fused_features = torch.cat([rgb_combined, semantic_features, depth_features, goal_features], dim=1)
        
        # Final fusion
        output = self.fusion_layers(fused_features)
        
        return output


class Actor(nn.Module):
    """Discrete Actor network with categorical distribution"""
    def __init__(self, hidden_dim=512, num_steering_actions=5, num_throttle_actions=3):
        super().__init__()
        
        # Create shared encoder
        self.encoder = MultiModalEncoder(hidden_dim=hidden_dim, fusion_strategy='attention')
        
        # Discrete action space definition
        self.num_steering_actions = num_steering_actions  # e.g., 5: [-1, -0.5, 0, 0.5, 1]
        self.num_throttle_actions = num_throttle_actions   # e.g., 3: [0.5, 0.75, 1.0]
        
        # Action mappings
        self.steering_actions = torch.linspace(-1.0, 1.0, num_steering_actions)
        self.throttle_actions = torch.linspace(0.5, 1.0, num_throttle_actions)
        
        # Separate networks for steering and throttle
        self.steering_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_steering_actions)
        )
        
        self.throttle_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_throttle_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
            
    def forward(self, rgb, semantic, depth, goal):
        z = self.encoder(rgb, semantic, depth, goal)
        
        # Get logits for each action dimension
        steering_logits = self.steering_net(z)
        throttle_logits = self.throttle_net(z)
        
        # Create categorical distributions
        steering_dist = torch.distributions.Categorical(logits=steering_logits)
        throttle_dist = torch.distributions.Categorical(logits=throttle_logits)
        
        return steering_dist, throttle_dist
    
    def get_action_values(self, steering_indices, throttle_indices, device):
        """Convert discrete indices to continuous action values"""
        steering_values = self.steering_actions[steering_indices].to(device)
        throttle_values = self.throttle_actions[throttle_indices].to(device)
        return steering_values, throttle_values
    
    def sample_action(self, rgb, semantic, depth, goal):
        """Sample action and return both indices and continuous values"""
        steering_dist, throttle_dist = self.forward(rgb, semantic, depth, goal)
        
        steering_idx = steering_dist.sample()
        throttle_idx = throttle_dist.sample()
        
        device = rgb.device
        steering_val, throttle_val = self.get_action_values(steering_idx, throttle_idx, device)
        
        return (steering_idx, throttle_idx), (steering_val, throttle_val)


class Critic(nn.Module):
    """Improved Critic network with better architecture"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Create shared encoder
        self.encoder = MultiModalEncoder(hidden_dim=hidden_dim, fusion_strategy='attention')
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1)
            torch.nn.init.constant_(module.bias, 0)
            
    def forward(self, rgb, semantic, depth, goal):
        z = self.encoder(rgb, semantic, depth, goal)
        value = self.value_net(z)
        
        return torch.squeeze(value, dim=-1)