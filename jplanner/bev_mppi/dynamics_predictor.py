# dynamics_predictor.py

import torch
import math

class DynamicsPredictor:
    """
    로봇의 동역학 모델(motion model)을 정의하고,
    다음 스텝의 상태(x, y, yaw)를 예측합니다.
    """
    def __init__(self, device, dt):
        """
        Args:
            device (torch.device): 'cuda' 또는 'cpu'
            dt (float): 예측 시간 간격 (control_timer와 동일)
        """
        self.device = device
        self.dt = dt
    
    # TODO : making this neural network : noisy localization data based 
    def motion_model(self, states, controls):
        """
        로봇의 다음 상태를 예측 (K개의 궤적에 대해 병렬 처리)
        
        Args:
            states: (K, 3) 텐서 [x, y, yaw]
            controls: (K, 2) 텐서 [v, w]
            
        Returns:
            (K, 3) 텐서: 다음 스텝의 [x, y, yaw]
        """
        v = controls[:, 0]
        w = controls[:, 1]
        yaw = states[:, 2]

        # sim2real gap optimization parameter 
        linear_coeff = 1.0 
        angular_coeff = 1.35 


        x_next = states[:, 0] + v * torch.cos(yaw) * self.dt * linear_coeff
        y_next = states[:, 1] + v * torch.sin(yaw) * self.dt * linear_coeff
        yaw_next = yaw + w * self.dt * angular_coeff
        
        # Yaw를 -pi ~ +pi 범위로 정규화
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        return torch.stack([x_next, y_next, yaw_next], dim=1)
