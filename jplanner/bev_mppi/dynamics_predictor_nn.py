# dynamics_predictor_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import math

class CalibrationNet(nn.Module):
    """
    (v, w)를 입력받아 (linear_coeff_adjustment, angular_coeff_adjustment)를 예측하는
    간단한 MLP (Multi-Layer Perceptron)
    """
    def __init__(self):
        super(CalibrationNet, self).__init__()
        # 입력: [v, w] (2), 출력: [linear_adjustment, angular_adjustment] (2)
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),                        
        )
        # 출력이 0에 가깝게 시작하도록 마지막 레이어의 가중치와 편향을 초기화
        # nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, controls):
        """
        Args:
            controls: (K, 2) 텐서 [v, w]
        Returns:
            (K, 2) 텐서 [linear_adjustment, angular_adjustment]
        """
        return self.network(controls)


class DynamicsPredictor:
    """
    로봇의 동역학 모델(motion model)을 정의하고,
    다음 스텝의 상태(x, y, yaw)를 예측합니다.
    (학습 가능한 보정 계수 NN 포함)
    """
    def __init__(self, device='cuda', dt=0.1, learning_rate=1e-3):
        """
        Args:
            device (torch.device): 'cuda' 또는 'cpu'
            dt (float): 예측 시간 간격 (control_timer와 동일)
            learning_rate (float): 신경망 학습률
        """
        self.device = device
        self.dt = dt

        # 신경망 모델 및 옵티마이저
        model_path = 'calibration_net_final.pth'
        self.calibration_net = CalibrationNet().to(device)
        self.calibration_net.load(model_path)

        self.optimizer = optim.Adam(self.calibration_net.parameters(), lr=learning_rate)
        
        # 위치(x, y) 오차를 위한 손실 함수
        self.pose_loss_fn = nn.MSELoss()

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

        # 신경망을 통해 (v, w)에 따른 보정값 예측
        # (K, 2) 텐서
        
        coeff_adjustments = self.calibration_net(controls)
        
        # 기본값 1.0에 신경망의 예측(조정값)을 더함
        # .unsqueeze(1)을 제거하고 텐서 연산을 맞춥니다.
        linear_coeff = 1.0 + coeff_adjustments[:, 0]
        angular_coeff = 1.0 + coeff_adjustments[:, 1]
        
        # (K,) 텐서가 (K,1) 텐서와 연산될 수 있도록 v, w, yaw의 차원을 맞춥니다.
        # [v] * [cos(yaw)] * [dt] * [coeff]
        x_next = states[:, 0] + v * torch.cos(yaw) * self.dt * linear_coeff
        y_next = states[:, 1] + v * torch.sin(yaw) * self.dt * linear_coeff
        yaw_next = yaw + w * self.dt * angular_coeff
        
        # Yaw를 -pi ~ +pi 범위로 정규화
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        return torch.stack([x_next, y_next, yaw_next], dim=1)

    def learn(self, start_states, controls, actual_next_states):
        """
        (시작 상태, 제어, 실제 결과 상태) 데이터를 기반으로 신경망을 1스텝 학습합니다.
        
        Args:
            start_states: (K, 3) 텐서 [x, y, yaw]
            controls: (K, 2) 텐서 [v, w]
            actual_next_states: (K, 3) 텐서 [x_actual, y_actual, yaw_actual]
            
        Returns:
            float: 현재 스텝의 총 손실 (loss)
        """
        # 1. 현재 신경망으로 다음 상태 예측
        # self.calibration_net.train() # (배치 정규화 등이 있다면 필요)
        predicted_next_states = self.motion_model(start_states, controls)
        
        # 2. 손실 계산 (Loss Calculation)
        
        # x, y에 대한 손실 (MSE)
        loss_xy = self.pose_loss_fn(predicted_next_states[:, :2], actual_next_states[:, :2])
        
        # Yaw에 대한 손실 (각도 차이를 정규화하여 계산)
        yaw_error = actual_next_states[:, 2] - predicted_next_states[:, 2]
        yaw_error_wrapped = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        loss_yaw = torch.mean(yaw_error_wrapped**2) # MSE
        
        # 총 손실 (Yaw 오차에 가중치 0.5 부여. 튜닝 가능)
        total_loss = loss_xy + 0.5 * loss_yaw
        
        print()
        print('xy loss : ',loss_xy.item(),'yaw loss : ', loss_yaw)

        # 3. 역전파 및 옵티마이저 스텝 (Backpropagation)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
