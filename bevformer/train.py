import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Optional, List, Tuple
from PIL import Image
import cv2

class PositionalEncoding(nn.Module):
    """3D Positional Encoding for BEV queries"""
    def __init__(self, num_feats, temperature=10000):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature

    def forward(self, mask):
        """
        Args:
            mask: [B, H, W] mask tensor
        Returns:
            pos: [B, H, W, num_feats*2] positional encoding
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for BEV queries"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        B, N, C = query.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attn_mask is not None:
            attn.masked_fill_(attn_mask, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class BEVFormerLayer(nn.Module):
    """Single BEVFormer Transformer Layer"""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, bev_query, img_features, pos_embed=None):
        # Self-attention on BEV queries
        q1 = self.norm1(bev_query)
        bev_query = bev_query + self.self_attn(q1, q1, q1)
        
        # Cross-attention with image features
        q2 = self.norm2(bev_query)
        bev_query = bev_query + self.cross_attn(q2, img_features, img_features)
        
        # FFN
        bev_query = bev_query + self.ffn(self.norm3(bev_query))
        
        return bev_query

class ImageEncoder(nn.Module):
    """Simple CNN backbone for image encoding"""
    def __init__(self, input_channels=3, embed_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            # Stage 2
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(256, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        features = self.backbone(x)  # [B, embed_dim, H', W']
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return features

class MonoBEVFormer(nn.Module):
    """BEVFormer for mono camera to BEV segmentation"""
    def __init__(self, 
                 img_size=(224, 224),
                 bev_size=(100, 100),
                 embed_dim=256,
                 num_layers=6,
                 num_heads=8,
                 num_classes=10,
                 dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.bev_size = bev_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Image encoder
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        
        # BEV queries (learnable parameters)
        self.bev_queries = nn.Parameter(torch.randn(bev_size[0] * bev_size[1], embed_dim))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim // 2)
        
        # BEVFormer layers
        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, img):
        B = img.shape[0]
        
        # Encode image features
        img_features = self.img_encoder(img)  # [B, N_img, embed_dim]
        
        # Expand BEV queries for batch
        bev_queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, embed_dim]
        
        # Apply BEVFormer layers
        for layer in self.layers:
            bev_queries = layer(bev_queries, img_features)
        
        # Generate segmentation map
        seg_logits = self.seg_head(bev_queries)  # [B, H*W, num_classes]
        seg_logits = seg_logits.reshape(B, self.bev_size[0], self.bev_size[1], self.num_classes)
        seg_logits = seg_logits.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
        
        return seg_logits

class MonoBEVDataset(Dataset):
    """Dataset for mono camera and BEV segmentation pairs"""
    def __init__(self, 
                 img_paths: List[str], 
                 bev_paths: List[str],
                 img_size=(224, 224),
                 bev_size=(100, 100)):
        self.img_paths = img_paths
        self.bev_paths = bev_paths
        self.img_size = img_size
        self.bev_size = bev_size
        
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.bev_transform = transforms.Compose([
            transforms.Resize(bev_size),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load front view image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.img_transform(img)
        
        # Load BEV segmentation map
        bev_seg = Image.open(self.bev_paths[idx]).convert('L')  # Grayscale
        bev_seg = self.bev_transform(bev_seg)
        bev_seg = (bev_seg * 255).long().squeeze(0)  # Convert to class indices
        
        return img, bev_seg

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda'):
    """Training function"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (img, bev_seg) in enumerate(train_loader):
            img, bev_seg = img.to(device), bev_seg.to(device)
            
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, bev_seg)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, bev_seg in val_loader:
                img, bev_seg = img.to(device), bev_seg.to(device)
                pred = model(img)
                loss = criterion(pred, bev_seg)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_bevformer_mono.pth')
        
        scheduler.step()

# World Model Extension for action prediction
class WorldModelBEVFormer(nn.Module):
    """Extended BEVFormer with action conditioning for world modeling"""
    def __init__(self, 
                 base_model: MonoBEVFormer,
                 action_dim=4,  # e.g., [throttle, brake, steer, speed]
                 sequence_length=5):
        super().__init__()
        
        self.base_model = base_model
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.embed_dim = base_model.embed_dim
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim)
        )
        
        # Temporal modeling with LSTM
        self.temporal_model = nn.LSTM(
            self.embed_dim, 
            self.embed_dim, 
            batch_first=True,
            num_layers=2
        )
        
        # Future prediction head
        self.future_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, base_model.num_classes)
        )
        
    def forward(self, img, actions=None, predict_future=False):
        """
        Args:
            img: Current front view image [B, 3, H, W]
            actions: Action sequence [B, seq_len, action_dim] (optional)
            predict_future: Whether to predict future BEV maps
        """
        # Get current BEV representation
        current_bev = self.base_model(img)  # [B, num_classes, H, W]
        
        if not predict_future or actions is None:
            return current_bev
        
        B, num_classes, H, W = current_bev.shape
        
        # Get BEV features before final classification
        with torch.no_grad():
            img_features = self.base_model.img_encoder(img)
            bev_queries = self.base_model.bev_queries.unsqueeze(0).expand(B, -1, -1)
            
            for layer in self.base_model.layers:
                bev_queries = layer(bev_queries, img_features)
        
        # Encode actions
        action_features = self.action_encoder(actions)  # [B, seq_len, embed_dim]
        
        # Combine BEV features with actions for temporal modeling
        # Use mean pooled BEV features as initial state
        bev_context = bev_queries.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
        bev_context = bev_context.expand(-1, self.sequence_length, -1)  # [B, seq_len, embed_dim]
        
        # Combine with action features
        combined_features = bev_context + action_features
        
        # Temporal modeling
        future_features, _ = self.temporal_model(combined_features)  # [B, seq_len, embed_dim]
        
        # Predict future BEV maps
        future_bevs = []
        for t in range(self.sequence_length):
            # Broadcast temporal features to spatial dimensions
            temp_feat = future_features[:, t, :].unsqueeze(1).expand(-1, H*W, -1)  # [B, H*W, embed_dim]
            
            # Predict future BEV segmentation
            future_logits = self.future_head(temp_feat)  # [B, H*W, num_classes]
            future_logits = future_logits.reshape(B, H, W, num_classes).permute(0, 3, 1, 2)
            future_bevs.append(future_logits)
        
        future_bevs = torch.stack(future_bevs, dim=1)  # [B, seq_len, num_classes, H, W]
        
        return current_bev, future_bevs

def train_world_model(world_model, train_loader, val_loader, num_epochs=50, lr=1e-5, device='cuda'):
    """Training function for world model"""
    world_model = world_model.to(device)
    optimizer = torch.optim.AdamW(world_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    for epoch in range(num_epochs):
        world_model.train()
        train_loss = 0
        
        for batch_idx, (img, bev_seg, actions, future_bevs) in enumerate(train_loader):
            img = img.to(device)
            bev_seg = bev_seg.to(device)
            actions = actions.to(device)
            future_bevs = future_bevs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            current_pred, future_preds = world_model(img, actions, predict_future=True)
            
            # Current BEV loss
            current_loss = criterion(current_pred, bev_seg)
            
            # Future BEV loss
            future_loss = 0
            for t in range(future_preds.shape[1]):
                future_loss += criterion(future_preds[:, t], future_bevs[:, t])
            future_loss /= future_preds.shape[1]
            
            # Total loss
            total_loss = current_loss + 0.5 * future_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
        
        print(f'Epoch {epoch}: Average Train Loss: {train_loss/len(train_loader):.4f}')

# Example usage
if __name__ == "__main__":
    # Example data paths (replace with your actual data)
    img_paths = ['path/to/front_view_img1.jpg', 'path/to/front_view_img2.jpg']
    bev_paths = ['path/to/bev_seg1.png', 'path/to/bev_seg2.png']
    
    # Create dataset and dataloader
    dataset = MonoBEVDataset(img_paths, bev_paths)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Create model
    model = MonoBEVFormer(
        img_size=(224, 224),
        bev_size=(100, 100),
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        num_classes=10  # Adjust based on your segmentation classes
    )
    
    # Train base model
    print("Training base BEVFormer model...")
    # train_model(model, train_loader, val_loader, num_epochs=100)
    
    # Create world model
    world_model = WorldModelBEVFormer(model, action_dim=4, sequence_length=5)
    
    # Train world model (requires additional data with action sequences)
    print("World model created for future BEV prediction with actions")
    # train_world_model(world_model, world_train_loader, world_val_loader, num_epochs=50)