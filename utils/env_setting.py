import cv2
import numpy as np 

def visualize_observations(obs: dict, window_name: str = "Observations"):
    """
    Combine RGB, Depth and Semantic images into one view and display.
    개선된 depth 이미지 시각화 포함
    """
    rgb = obs.get('rgb')
    depth = obs.get('depth')
    semantic = obs.get('semantic')

    images_to_stack = []
    
    # RGB 이미지 처리
    if rgb is not None:
        images_to_stack.append(rgb)
    
    # Depth 이미지 처리 - 더 나은 시각화를 위해 컬러맵 적용
    if depth is not None:
        # depth가 (H, W, 1) 형태인 경우 (H, W)로 변환
        if len(depth.shape) == 3 and depth.shape[2] == 1:
            depth_2d = depth.squeeze(axis=2)
        else:
            depth_2d = depth
            
        # COLORMAP_JET를 사용하여 depth를 컬러 이미지로 변환
        depth_colored = cv2.applyColorMap(depth_2d, cv2.COLORMAP_JET)
        images_to_stack.append(depth_colored)
    
    # Semantic 이미지 처리
    if semantic is not None:
        images_to_stack.append(semantic)
    
    if not images_to_stack:
        return
    
    # 모든 이미지를 같은 높이로 리사이즈
    h = images_to_stack[0].shape[0]
    resized_images = []
    
    for img in images_to_stack:
        if img.shape[0] != h:
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(h * aspect_ratio)
            img_resized = cv2.resize(img, (new_width, h))
        else:
            img_resized = img
        resized_images.append(img_resized)
    
    # 수평으로 스택
    combined = np.hstack(resized_images)
    
    # 창 크기가 너무 크면 조정
    max_width = 1920
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (max_width, new_height))
    
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)
   

import cv2
import numpy as np 

def visualize_observations(obs: dict, window_name: str = "Observations", max_range: float = 30.0):
    """
    Combine RGB, Depth and Semantic images into one view and display.
    개선된 depth 이미지 시각화 포함
    """
    rgb = obs.get('rgb')
    depth = obs.get('depth')
    semantic = obs.get('semantic')

    images_to_stack = []
    
    # RGB 이미지 처리
    if rgb is not None:
        images_to_stack.append(rgb)
    
    # Depth 이미지 처리 - 클리핑→스케일→컬러맵
    if depth is not None:
        # (H, W, 1) → (H, W)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth_2d = depth[:, :, 0].astype(np.float32)
        else:
            depth_2d = depth.astype(np.float32)
        
        # 1) 거리값을 0–max_range로 클리핑
        depth_clipped = np.clip(depth_2d, 0.0, max_range)
        
        # 2) 0–max_range → 0–255 스케일
        depth_scaled = (depth_clipped / max_range * 255.0).astype(np.uint8)
        
        # 3) JET 컬러맵 적용 (uint8 입력 필수)
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        
        images_to_stack.append(depth_colored)
    
    # Semantic 이미지 처리
    if semantic is not None:
        images_to_stack.append(semantic)
    
    if not images_to_stack:
        return
    
    # 모든 이미지를 같은 높이로 리사이즈
    h = images_to_stack[0].shape[0]
    resized_images = []
    for img in images_to_stack:
        if img.shape[0] != h:
            ar = img.shape[1] / img.shape[0]
            img = cv2.resize(img, (int(h * ar), h))
        resized_images.append(img)
    
    # 수평 스택 & 윈도우 조정
    combined = np.hstack(resized_images)
    max_width = 1920
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        combined = cv2.resize(combined, (max_width, int(combined.shape[0] * scale)))
    
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)
