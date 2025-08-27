# train_ddp.py - DDP 버전

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import SegformerForSemanticSegmentation
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

# 이전 단계에서 작성한 PolygonSegmentationDataset 클래스를 임포트합니다.
from polygon_dataset import PolygonSegmentationDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 중복 제거
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 중복 제거
os.environ["NCCL_P2P_DISABLE"] = '1'  # GPU P2P 통신 끔

def setup_ddp(rank, world_size):
    """DDP 환경 설정"""
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    # CUDA_VISIBLE_DEVICES로 설정된 GPU들은 PyTorch에서 0, 1로 매핑됨
    # 실제 GPU 0, 3 -> PyTorch에서 0, 1
    local_gpu = rank
    
    # CUDA 디바이스 설정
    torch.cuda.set_device(local_gpu)
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return local_gpu

def cleanup():
    """DDP 환경 정리"""
    dist.destroy_process_group()

def create_colormap(num_classes):
    """클래스별 고유 색상 생성"""
    colors = []
    for i in range(num_classes):
        # HSV 색공간에서 균등하게 분포된 색상 생성
        hue = i / num_classes
        saturation = 0.8
        value = 0.9
        
        # HSV를 RGB로 변환
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        colors.append([r + m, g + m, b + m])
    
    # 배경은 검은색으로
    colors[0] = [0, 0, 0]
    
    return ListedColormap(colors)

def visualize_predictions(model, valid_loader, device, epoch, save_dir="validation_results", num_samples=4, rank=0):
    """검증 결과를 시각화하고 저장 (rank 0에서만 실행)"""
    if rank != 0:
        return
        
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    colormap = create_colormap(30)  # num_labels 하드코딩
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            if batch_idx >= num_samples:  # 지정된 수만큼만 시각화
                break
                
            images = data['pixel_values'].to(device)
            masks = data['labels'].to(device)
            
            # 추론 - SegFormer 모델로 예측
            inputs = {'pixel_values': images}
            outputs = model(**inputs)
            logits = outputs.logits
            
            upsampled_logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            predictions = torch.argmax(upsampled_logits, dim=1)
            
            # 첫 번째 이미지만 시각화
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask = masks[0].cpu().numpy()
            pred_mask = predictions[0].cpu().numpy()
            
            # 이미지 정규화 해제 (0-1 범위로)
            img = (img - img.min()) / (img.max() - img.min())
            
            # 서브플롯 생성
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 원본 이미지
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground Truth
            axes[1].imshow(gt_mask, cmap=colormap, vmin=0, vmax=29)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(pred_mask, cmap=colormap, vmin=0, vmax=29)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            save_path = os.path.join(save_dir, f"loss_epoch_{epoch+1}_sample_{batch_idx+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Validation visualization saved: {save_path}")

def validate_model(model, valid_loader, device):
    """검증 함수 - SegFormer 내장 loss 사용"""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)
            
            # SegFormer 모델 직접 사용
            val_inputs = {
                'pixel_values': val_imgs,
                'labels': val_masks
            }
            val_outputs = model(**val_inputs)
            
            loss = val_outputs.loss
            val_loss += loss.item()
            
            # 예측 결과 계산
            val_logits = val_outputs.logits
            val_upsampled = F.interpolate(
                val_logits,
                size=val_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 정확도 계산
            preds = torch.argmax(val_upsampled, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    avg_val_loss = val_loss / len(valid_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    return avg_val_loss, pixel_accuracy

def main(rank, world_size):
    """메인 학습 함수"""
    
    # DDP 설정
    local_gpu = setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{local_gpu}')
    
    # 상수 정의
    ROOT_DIRECTORY = "/home/work/data/indo_walking/polygon_segmentation"
    CLASS_MAPPING_FILE = os.path.join(ROOT_DIRECTORY, 'class_mapping.json')
    
    # 클래스 정보 로드
    try:
        with open(CLASS_MAPPING_FILE, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
    except FileNotFoundError:
        if rank == 0:
            print(f"🚨 클래스 매핑 파일('{CLASS_MAPPING_FILE}')을 찾을 수 없습니다. 프로그램을 종료합니다.")
        cleanup()
        return
    
    # 클래스 매핑
    class_to_idx = {
        'background': 0,
        'barricade': 1,
        'bench': 2,
        'bicycle': 3,
        'bollard': 4,
        'bus': 5,
        'car': 6,
        'carrier': 7,
        'cat': 8,
        'chair': 9,
        'dog': 10,
        'fire_hydrant': 11,
        'kiosk': 12,
        'motorcycle': 13,
        'movable_signage': 14,
        'parking_meter': 15,
        'person': 16,
        'pole': 17,
        'potted_plant': 18,
        'power_controller': 19,
        'scooter': 20,
        'stop': 21,
        'stroller': 22,
        'table': 23,
        'traffic_light': 24,
        'traffic_light_controller': 25,
        'traffic_sign': 26,
        'tree_trunk': 27,
        'truck': 28,
        'wheelchair': 29
    }
    
    id2label = {int(idx): label for label, idx in class_to_idx.items()}
    label2id = class_to_idx
    num_labels = len(id2label)
    
    if rank == 0:
        print(f"클래스 수: {num_labels}")
        print(f"클래스 레이블: {id2label}")
    
    # 데이터셋 생성
    train_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=True)
    valid_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=False)
    
    # DDP용 샘플러 생성
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    
    # 데이터로더 생성
    batch_size = 64  # DDP에서는 총 배치 크기가 32 * 2 = 64가 됨
    num_workers = 8  # worker 수 조정
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # 모델 생성
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # DDP 래핑
    model = DDP(model, device_ids=[local_gpu], output_device=local_gpu)
    
    # 하이퍼파라미터
    epochs = 100
    lr = 5e-5
    weight_decay = 1e-4
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # WandB 초기화 (rank 0에서만)
    if rank == 0:
        wandb.init(project="segmentation_project", name=f"ddp_run_0707")
    
    # 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 샘플러 epoch 설정 (중요!)
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            images = data['pixel_values'].to(device)
            masks = data['labels'].to(device)
            
            # SegFormer 모델 입력 구성
            inputs = {
                'pixel_values': images,
                'labels': masks
            }
            
            # 순전파
            outputs = model(**inputs)
            
            # SegFormer 내장 loss 사용
            loss = outputs.loss
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0 and rank == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 검증 (모든 GPU에서 실행)
        val_loss, pixel_acc = validate_model(model, valid_loader, device)
        
        # 결과 수집 (all_reduce)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        pixel_acc_tensor = torch.tensor(pixel_acc, device=device)
        
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_acc_tensor, op=dist.ReduceOp.SUM)
        
        val_loss = val_loss_tensor.item() / world_size
        pixel_acc = pixel_acc_tensor.item() / world_size
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        # 평균 손실 계산
        avg_train_loss = running_loss / len(train_loader)
        
        # 검증 결과 시각화 (rank 0에서만)
        if rank == 0:
            visualize_predictions(model.module, valid_loader, device, epoch, 
                                 save_dir="validation_results", num_samples=4, rank=rank)
            
            # WandB 로깅
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "pixel_accuracy": pixel_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Pixel Acc: {pixel_acc:.4f}")
        
        # 모델 저장 (rank 0에서만)
        if rank == 0:
            # 체크포인트 디렉토리 생성
            os.makedirs("ckpts", exist_ok=True)
            
            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), f"ckpts/best_seg_model.pth")
                print(f"Best 모델 저장! Val Loss: {val_loss:.4f}")
            
            # 정기적으로 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), f"ckpts/seg_model_epoch_{epoch+1}.pth")
    
    if rank == 0:
        print("학습 완료!")
        wandb.finish()
    
    cleanup()

if __name__ == "__main__":
    # 사용할 GPU 수 설정 (0번, 3번 GPU)
    world_size = 3
    
    # 멀티프로세싱으로 DDP 실행
    import torch.multiprocessing as mp
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)