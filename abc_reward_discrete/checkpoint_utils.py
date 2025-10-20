"""
체크포인트 로딩 유틸리티
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def load_actor_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> bool:
    """
    Actor 모델 체크포인트를 안전하게 로드
    
    Args:
        model: Actor 모델 인스턴스
        checkpoint_path: 체크포인트 파일 경로
        device: 로딩할 디바이스
        
    Returns:
        로딩 성공 여부
    """
    try:
        print(f"Loading actor checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 체크포인트가 state_dict인지 전체 모델인지 확인
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # 전체 모델이 저장된 경우
            state_dict = checkpoint.state_dict()
        
        # 불필요한 키 제거
        keys_to_remove = []
        for key in list(state_dict.keys()):
            # buffer keys (register_buffer로 등록된 것들)
            if key in ['steering_actions', 'throttle_actions']:
                keys_to_remove.append(key)
            # 모듈 prefix 제거
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
                state_dict[new_key] = state_dict.pop(key)
        
        for key in keys_to_remove:
            state_dict.pop(key, None)
            print(f"Removed key: {key}")
        
        # 모델에 로드
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        print("✅ Actor checkpoint loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load actor checkpoint: {e}")
        return False

def load_critic_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> bool:
    """
    Critic 모델 체크포인트를 안전하게 로드
    
    Args:
        model: Critic 모델 인스턴스
        checkpoint_path: 체크포인트 파일 경로
        device: 로딩할 디바이스
        
    Returns:
        로딩 성공 여부
    """
    try:
        print(f"Loading critic checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 체크포인트가 state_dict인지 전체 모델인지 확인
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # 전체 모델이 저장된 경우
            state_dict = checkpoint.state_dict()
        
        # 모듈 prefix 제거
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # 모델에 로드
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        print("✅ Critic checkpoint loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load critic checkpoint: {e}")
        return False

def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    체크포인트 파일의 내용을 검사
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        
    Returns:
        체크포인트 정보 딕셔너리
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'type': type(checkpoint).__name__,
            'keys': [],
            'tensor_shapes': {},
            'total_parameters': 0
        }
        
        if isinstance(checkpoint, dict):
            info['keys'] = list(checkpoint.keys())
            
            # state_dict 찾기
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # 전체가 state_dict인 경우
                state_dict = checkpoint
            
            if state_dict and isinstance(state_dict, dict):
                for key, value in state_dict.items():
                    if hasattr(value, 'shape'):
                        info['tensor_shapes'][key] = list(value.shape)
                        info['total_parameters'] += value.numel()
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # 체크포인트 검사 예제
    checkpoint_path = "checkpoints/metaurban_discrete_actor_epoch_90.pt"
    
    print("=== Checkpoint Inspection ===")
    info = inspect_checkpoint(checkpoint_path)
    
    if 'error' in info:
        print(f"Error: {info['error']}")
    else:
        print(f"Type: {info['type']}")
        print(f"Keys: {info['keys']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Tensor shapes:")
        for key, shape in info['tensor_shapes'].items():
            print(f"  {key}: {shape}")