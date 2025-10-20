# python3 enhanced_train_ddp.py --batch_size 16
# train_ddp_improved.py - 개선된 DDP 버전

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
import logging
from datetime import datetime, timedelta

# 이전 단계에서 작성한 PolygonSegmentationDataset 클래스를 임포트합니다.
from polygon_dataset import PolygonSegmentationDataset

def setup_logging(rank):
    """로깅 설정"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

def setup_ddp(rank, world_size, master_addr='localhost', master_port='12355'):
    """DDP 환경 설정"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'  # InfiniBand 비활성화
    
    # CUDA 디바이스 설정
    torch.cuda.set_device(rank)
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800)  # 30분 타임아웃
    )
    
    return rank

def cleanup():
    """DDP 환경 정리"""
    if dist.is_initialized():
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

def visualize_predictions(model, valid_loader, device, epoch, save_dir="validation_results", num_samples=7, rank=0):
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
            save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{batch_idx+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Validation visualization saved: {save_path}")

def validate_model(model, valid_loader, device, rank, world_size):
    """검증 함수 - 분산 처리 개선"""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device, non_blocking=True)
            val_masks = val_data['labels'].to(device, non_blocking=True)
            
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
    
    # 분산 처리를 위한 텐서 생성
    val_loss_tensor = torch.tensor(val_loss, device=device)
    correct_pixels_tensor = torch.tensor(correct_pixels, device=device)
    total_pixels_tensor = torch.tensor(total_pixels, device=device)
    
    # 모든 GPU에서 결과 합계
    if world_size > 1:
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_pixels_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pixels_tensor, op=dist.ReduceOp.SUM)
    
    avg_val_loss = val_loss_tensor.item() / (len(valid_loader) * world_size)
    pixel_accuracy = correct_pixels_tensor.item() / total_pixels_tensor.item()
    
    return avg_val_loss, pixel_accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_dir="ckpts"):
    """체크포인트 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    return checkpoint

def main(rank, world_size, args):
    """메인 학습 함수"""
    
    try:
        # 로깅 설정
        setup_logging(rank)
        
        # DDP 설정
        local_gpu = setup_ddp(rank, world_size, args.master_addr, args.master_port)
        device = torch.device(f'cuda:{local_gpu}')
        
        if rank == 0:
            logging.info(f"Starting training on {world_size} GPUs")
            logging.info(f"Device: {device}")
            
    except Exception as e:
        if rank == 0:
            logging.error(f"DDP setup failed: {e}")
        return
    
    # 상수 정의
    ROOT_DIRECTORY = args.data_dir
    CLASS_MAPPING_FILE = os.path.join(ROOT_DIRECTORY, 'class_mapping.json')
    
    # 클래스 정보 로드
    try:
        with open(CLASS_MAPPING_FILE, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
    except FileNotFoundError:
        if rank == 0:
            logging.error(f"클래스 매핑 파일('{CLASS_MAPPING_FILE}')을 찾을 수 없습니다.")
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
        logging.info(f"클래스 수: {num_labels}")
        logging.info(f"클래스 레이블: {id2label}")
    
    # 데이터셋 생성
    try:
        train_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=True)
        valid_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=False)
        
        if rank == 0:
            logging.info(f"Train dataset size: {len(train_dataset)}")
            logging.info(f"Valid dataset size: {len(valid_dataset)}")
            
    except Exception as e:
        if rank == 0:
            logging.error(f"데이터셋 로딩 오류: {e}")
        cleanup()
        return
    
    # DDP용 샘플러 생성
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        sampler=valid_sampler,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if rank == 0:
        logging.info(f"Train batches per epoch: {len(train_loader)}")
        logging.info(f"Valid batches per epoch: {len(valid_loader)}")
    
    # 모델 생성
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        model.to(device)
        
        if rank == 0:
            logging.info(f"Model loaded: {args.model_name}")
            
    except Exception as e:
        if rank == 0:
            logging.error(f"모델 로딩 오류: {e}")
        cleanup()
        return
    
    # DDP 래핑
    model = DDP(
        model, 
        device_ids=[local_gpu], 
        output_device=local_gpu,
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    
    # 옵티마이저
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # WandB 초기화 (rank 0에서만)
    if rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # 학습 루프
    best_val_loss = float('inf')
    
    try:
        for epoch in range(args.epochs):
            # 샘플러 epoch 설정 (중요!)
            train_sampler.set_epoch(epoch)
            
            model.train()
            running_loss = 0.0
            
            for i, data in enumerate(train_loader):
                try:
                    images = data['pixel_values'].to(device, non_blocking=True)
                    masks = data['labels'].to(device, non_blocking=True)
                    
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
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    if (i + 1) % args.log_interval == 0 and rank == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                                   f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                        
                except Exception as e:
                    if rank == 0:
                        logging.error(f"Training step error: {e}")
                    continue
            
            # 검증
            try:
                val_loss, pixel_acc = validate_model(model, valid_loader, device, rank, world_size)
                
                # 학습률 스케줄러 업데이트
                scheduler.step()
                
                # 평균 손실 계산
                avg_train_loss = running_loss / len(train_loader)
                
                # 검증 결과 시각화 (rank 0에서만)
                if rank == 0:
                    if (epoch + 1) % args.vis_interval == 0:
                        visualize_predictions(model.module, valid_loader, device, epoch, 
                                             save_dir=args.vis_dir, num_samples=4, rank=rank)
                    
                    # WandB 로깅
                    if args.use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "pixel_accuracy": pixel_acc,
                            "learning_rate": scheduler.get_last_lr()[0]
                        })
                    
                    logging.info(f"Epoch [{epoch+1}/{args.epochs}] - "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"Pixel Acc: {pixel_acc:.4f}")
                    
                    # 모델 저장
                    try:
                        # 체크포인트 디렉토리 생성
                        os.makedirs(args.save_dir, exist_ok=True)
                        
                        # 최고 성능 모델 저장
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                            logging.info(f"Best 모델 저장! Val Loss: {val_loss:.4f}")
                        
                        # 정기적으로 체크포인트 저장
                        if (epoch + 1) % args.save_interval == 0:
                            save_checkpoint(model.module, optimizer, scheduler, epoch, val_loss, args.save_dir)
                            logging.info(f"Checkpoint saved at epoch {epoch+1}")
                            
                    except Exception as e:
                        logging.error(f"Model save error: {e}")
                        
            except Exception as e:
                if rank == 0:
                    logging.error(f"Validation error: {e}")
                continue
                
    except Exception as e:
        if rank == 0:
            logging.error(f"Training loop error: {e}")
    
    if rank == 0:
        logging.info("학습 완료!")
        if args.use_wandb:
            wandb.finish()
    
    cleanup()

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='DDP Training for Semantic Segmentation')
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, 
                       default="/home/work/data/indo_walking/polygon_segmentation",
                       help='데이터 디렉토리 경로')
    
    # 모델 관련
    parser.add_argument('--model_name', type=str, 
                       default="nvidia/mit-b0",
                       help='사용할 모델 이름')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='배치 크기 (per GPU)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='데이터 로더 워커 수')
    
    # 로깅 및 저장 관련
    parser.add_argument('--log_interval', type=int, default=50,
                       help='로그 출력 간격')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='모델 저장 간격')
    parser.add_argument('--vis_interval', type=int, default=5,
                       help='시각화 간격')
    parser.add_argument('--save_dir', type=str, default="ckpts",
                       help='모델 저장 디렉토리')
    parser.add_argument('--vis_dir', type=str, default="validation_results",
                       help='시각화 저장 디렉토리')
    
    # WandB 관련
    parser.add_argument('--use_wandb', action='store_true',
                       help='WandB 사용 여부')
    parser.add_argument('--wandb_project', type=str, default="segmentation_project",
                       help='WandB 프로젝트 이름')
    parser.add_argument('--wandb_run_name', type=str, 
                       default=f"ddp_run_{datetime.now().strftime('%m%d_%H%M')}",
                       help='WandB 실행 이름')
    
    # DDP 관련
    parser.add_argument('--world_size', type=int, default=3,
                       help='사용할 GPU 수')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='마스터 주소')
    parser.add_argument('--master_port', type=str, default='12355',
                       help='마스터 포트')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 환경 변수 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    
    # 멀티프로세싱으로 DDP 실행
    import torch.multiprocessing as mp
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)