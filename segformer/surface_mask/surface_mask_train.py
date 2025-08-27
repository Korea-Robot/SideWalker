import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from surface_dataset import SurfaceSegmentationDataset

# --- 설정 ---
metadata_path = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"

batch_size = 32
target_size = (512, 512)
num_workers = 16

# --- 데이터 로드 ---
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

data_base_path = os.path.dirname(os.path.dirname(metadata_path))

# 데이터셋 인스턴스 생성
train_dataset = SurfaceSegmentationDataset(metadata['train_data'], target_size, is_train=True, data_base_path=data_base_path)
valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False, data_base_path=data_base_path)

# --- 클래스 가중치 계산 함수 ---
def calculate_class_weights(dataset, num_classes=7):
    """클래스 불균형을 해결하기 위한 가중치 계산"""
    class_counts = np.zeros(num_classes)
    
    print("클래스 분포 계산 중...")
    # 데이터셋의 일부를 샘플링하여 빠르게 계산
    for i in range(min(len(dataset), 1000)):
        mask = dataset[i]['labels'].numpy()
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts + 1e-6) # 0으로 나누는 것을 방지
    
    # 배경 클래스 가중치 조정 (너무 커지지 않도록 제한)
    class_weights[0] = min(class_weights[0], 0.5)
    
    return torch.FloatTensor(class_weights)

# --- 클래스 매핑 ---
class_to_idx = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
}
id2label = {int(idx): label for label, idx in class_to_idx.items()}
num_labels = len(id2label)

print(f"클래스 수: {num_labels}")
print(f"클래스 레이블: {id2label}")

# --- 데이터로더 생성 ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, pin_memory=True)

# 클래스 가중치 계산
class_weights = calculate_class_weights(train_dataset, num_labels)
print(f"계산된 클래스 가중치: {class_weights}")

# --- 모델 정의 ---
class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):    
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

# --- 학습 설정 ---
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu") # 첫번째 지정 GPU 사용
model = DirectSegFormer(num_classes=num_labels)
model.to(device)

# 하이퍼파라미터
epochs = 100
lr = 3e-5
weight_decay = 1e-4

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- WandB 초기화 ---
wandb.init(project="segmentation_project", name=f"fixed_run_0707_with_viz")

# --- 검증 및 시각화 함수 ---

def validate_model(model, valid_loader, criterion, device):
    """모델 검증 및 손실/정확도 계산"""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)
            
            val_logits = model(val_imgs)
            val_outputs = F.interpolate(
                val_logits, size=val_masks.shape[-2:], mode='bilinear', align_corners=False
            )
            
            loss = criterion(val_outputs, val_masks)
            val_loss += loss.item()
            
            preds = torch.argmax(val_outputs, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    avg_val_loss = val_loss / len(valid_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    return avg_val_loss, pixel_accuracy

def create_colormap(num_classes):
    """시각화를 위한 클래스별 고유 색상맵 생성"""
    colors = plt.cm.get_cmap('jet', num_classes)
    custom_colors = colors(np.arange(num_classes))
    custom_colors[0] = [0, 0, 0, 1] # 배경 클래스는 검은색으로 설정
    return ListedColormap(custom_colors)

def visualize_predictions(model, valid_loader, device, epoch, save_dir="validation_results", num_samples=4):
    """검증 데이터에 대한 예측 결과를 시각화하여 저장"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    colormap = create_colormap(num_labels)
    
    with torch.no_grad():
        # 검증 데이터로더에서 몇 개의 배치만 가져와 시각화
        for i, data in enumerate(valid_loader):
            if i >= num_samples:
                break
                
            images = data['pixel_values'].to(device)
            masks = data['labels'].to(device)
            
            logits = model(images)
            upsampled_logits = F.interpolate(
                logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
            )
            predictions = torch.argmax(upsampled_logits, dim=1)
            
            # 각 배치의 첫 번째 샘플만 사용
            img_np = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask_np = masks[0].cpu().numpy()
            pred_mask_np = predictions[0].cpu().numpy()
            
            # 이미지 정규화 해제 (시각화를 위해 0-1 범위로 조정)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            # 결과 이미지 생성
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(gt_mask_np, cmap=colormap, vmin=0, vmax=num_labels-1)
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            axes[2].imshow(pred_mask_np, cmap=colormap, vmin=0, vmax=num_labels-1)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

# --- 학습 루프 ---
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        images = data['pixel_values'].to(device)
        masks = data['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        upsampled_logits = F.interpolate(
            logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
        )
        
        loss = criterion(upsampled_logits, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # 에포크 종료 후 검증
    val_loss, pixel_acc = validate_model(model, valid_loader, criterion, device)
    scheduler.step()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # WandB 로깅
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "pixel_accuracy": pixel_acc,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Pixel Acc: {pixel_acc:.4f}")
    
    # ✨ 검증 결과 시각화 및 저장 ✨
    visualize_predictions(model, valid_loader, device, epoch)
    print(f"Epoch [{epoch+1}/{epochs}] - Validation visualizations saved.")
    
    # 최고 성능 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("ckpts", exist_ok=True)
        torch.save(model.state_dict(), f"ckpts/best_seg_model.pth")
        print(f"Best model saved! Val Loss: {val_loss:.4f}")
    
    # 주기적 체크포인트 저장
    if (epoch + 1) % 10 == 0:
        os.makedirs("ckpts", exist_ok=True)
        torch.save(model.state_dict(), f"ckpts/seg_model_epoch_{epoch+1}.pth")

print("학습 완료!")
wandb.finish()