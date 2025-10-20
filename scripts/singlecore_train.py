# autonomous_driving_ppo.py

# multiproessing 없이 단일 core를 통해 동작을 확인해보자. 이유는 braekpoint를 통해 출력을해보면서 디버깅 하기위함.
# import multiprocessing as mp
from queue import Queue,Full  # multiprocessing 없이 단일 core를 위한 Qeueu : 안전한 큐 구조체 : thread간 안전 데이터 공유
# 여러 thread의 동시 접근 제한

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # double-ended queue : threading 무관하여 안전하지않음. 대신 리스트보다 빠른 큐작업 제공
# 단일 스레드에서 큐를 사용하는경우 deque가 좋음.

import cv2
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import random

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# mp.set_start_method("spawn", force=True)

# ============================================================================
# 1. 환경 설정 (BASE_ENV_CFG)
# ============================================================================
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=False,  # 워커에서는 렌더링 비활성화 (성능 향상) : 시각화 없음
    map='X', 
    manual_control=False, 
    crswalk_density=1, 
    object_density=0.1, 
    walk_on_all_regions=False,
    drivable_area_extension=55, 
    height_scale=1, 
    horizon=300,
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=True, 
    max_lateral_dist=5.0,
    # 이미지 관측 설정
    agent_observation=ThreeSourceMixObservation,
    image_observation=True, # image 센서 데이터 활성화
    sensors={ # 활성화시킬 센서 
        # "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,  # 로그 최소화
)

# ============================================================================
# 2. PPO 신경망 모델 (메인 프로세스에서만 사용) 
# ============================================================================
class CNNFeatureExtractor(nn.Module):
    """RGB, Depth, Semantic 이미지를 처리하는 CNN 특징 추출기"""
    def __init__(self, input_channels=6, feature_dim=512):  # Semantic(3) + Depth(3) = > RGB(3) + Depth(3) 
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Conv Block 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # 출력 크기 계산을 위한 더미 forward
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, SENSOR_SIZE[1], SENSOR_SIZE[0])
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class PPOPolicy(nn.Module):
    """PPO 정책 네트워크 (Actor)
    
    Contionous Action : Normal D를 통해 구현
    
    """
    def __init__(self, feature_dim=512, goal_vec_dim=2, action_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        
        # goal_vec과 CNN 특징을 결합
        combined_dim = feature_dim + goal_vec_dim
        
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 연속 행동 (steering, acceleration)
        )
        
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 학습 가능한 표준편차
        # 표준오차는 state에 관계없이 action dim별로 일정하다는 가정을 한다.
    
    def forward(self, images, goal_vec):
        # 이미지 특징 추출
        img_features = self.feature_extractor(images)
        
        # goal_vec과 결합
        combined = torch.cat([img_features, goal_vec], dim=1)
        
        # 평균 행동 예측
        mean = self.policy_head(combined)
        
        # 표준편차 (양수로 제한)
        std = torch.exp(self.log_std)
        
        return mean, std

class PPOValue(nn.Module):
    """PPO 가치 네트워크 (Critic)"""
    def __init__(self, feature_dim=512, goal_vec_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        
        combined_dim = feature_dim + goal_vec_dim
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 상태 가치
        )
    
    def forward(self, images, goal_vec):
        img_features = self.feature_extractor(images)
        combined = torch.cat([img_features, goal_vec], dim=1)
        value = self.value_head(combined)
        return value

# ============================================================================
# 3. PPO 알고리즘 클래스 (메인 프로세스에서만 사용) # 왜냐하면 multi process로 동작하기위해서!
# ============================================================================

# @dataclass 목적 : 데이터 저장 전용 클래스 빠르고 간결하게 만들기 위함.
# @dataclass 가 붙은 클래스는 자동으로 __init__ 생성자를 만들어줌
#               __repr__ : 객체 출력 =>  print(cfg) 시 PPOConfig(lr=1e-4, gamma=0.98, ...)
#               __eq__ : 두 객체 비교 =>  cfg1 == cfg2로 값 비교 가능

# 장점 1. 코드 간결해짐.
# 장점 2. 속성값을 명확히 표현, 타입 힌트로 가독성 좋음
# 장점 3. 자동 비교, 출력 복사 유용


@dataclass
class PPOConfig:
    """PPO 하이퍼파라미터"""
    lr: float = 3e-4
    gamma: float = 0.99         # Reward Discount
    gae_lambda: float = 0.95    # GAE lambda
    clip_epsilon: float = 0.2   # 최대 분포 변경 ratio 비율 . 논문에서 실험적으로 0.2 가 좋다고 했던것같음.
    value_coef: float = 0.5     # critic = 0 을 너무 잘맞추면 adavantage가 0이 되므로 ciritc은 천천히 맞추게 하는게 중요 
    entropy_coef: float = 0.01  # entropy : exploration
    max_grad_norm: float = 0.5  # grad maximum 제한
    ppo_epochs: int = 4         
    batch_size: int = 256
    buffer_size: int = 2048     # buffer size가 batch에 비해 큰 이유 : 
                                # 일정량의 환경 경험을 버퍼에 모았다가, 그 전체(혹은 상당 부분)를 여러 번(Epoch) 반복해서 학습합니다.
                                # 즉, buffer는 PPO 업데이트를 위한 하나의 경험 집합(rollout, trajectory)
                                # 이때, buffer는 모두 같은 data D를 가짐. ~ iid approx
                                

class PPOAgent:
    """PPO 에이전트 - 메인 프로세스에서 모델 학습 담당"""
    def __init__(self, config: PPOConfig, device='cuda'):
        self.config = config
        self.device = device
        
        # 네트워크 초기화
        self.policy = PPOPolicy().to(device)
        self.value = PPOValue().to(device)
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.lr)
        
        # 경험 버퍼 (메인 프로세스에서만 관리) : 메인 프로세서에서만 경험 버퍼를 관리할수있음. 
        self.buffer = deque(maxlen=config.buffer_size) # 이건 main process의 queue
        # mp.Queue() 는 프로세서간  데이터  전달만 담당함.
        # 각 worker가 하나씩 큐로 데이터를 던짐.
        # 메인프로세서가 큐에서 rollout(데이터 포인트트)를 받아 buffer에 저장.
        # 같은 데이터 분포에서 배치를 가져와야함
    
    def select_action(self, images, goal_vec):
        """행동 선택 (추론 모드)"""
        with torch.no_grad():
            mean, std = self.policy(images, goal_vec)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1) # 이렇게 하면 실제 prob가 달라져서 괜찮나?  왜냐하면 뒤에서 행동범위를 제약하니까. 어떻게 하는게 정확한지 모르겠어. 

            # 행동 범위 제한 [-1, 1]
            action = torch.tanh(action)
        
            # jacobian을 통해 조정 
            log_prob = log_prob - torch.log(1-action.pow(2) + 1e-7).sum(dim=1)
            
        return action, log_prob
    
    def add_experience(self, experience):
        """경험을 버퍼에 추가"""
        self.buffer.append(experience)
    
    
    # Generalized Advantage Estimator : decrease Bias
    def compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            # i 시점에서 delta : 여기서 dones[i]=1이면 끝난시점임. next_value를 반영 안함. 
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            
            # 이전의 delta를 축적해서 GAE를 구함.
            # dones[i] =1 인시점에서 누적 advantage를 보상으로 주지않음
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self):
        """PPO 학습 업데이트"""
        if len(self.buffer) < self.config.batch_size:
            return
        
        # 버퍼에서 배치 샘플링
        # batch_data = list(self.buffer) # 크기를 일정하게 잘라야지!! 어덯게?? 
        
        # random sample을 통해 미니배치 학습 함.
        batch_data = random.sample(self.buffer,self.config.batch_size)
        
        self.buffer.clear()  # 버퍼 초기화 # 
        
        # 데이터 분리 # 리스트 형식으로 저장함
        states = [exp['state'] for exp in batch_data] 
        actions = [exp['action'] for exp in batch_data]
        rewards = [exp['reward'] for exp in batch_data]
        next_states = [exp['next_state'] for exp in batch_data]
        dones = [exp['done'] for exp in batch_data]
        old_log_probs = [exp['log_prob'] for exp in batch_data] # ppo의 clip을 위해 저장해놓아야함.
        
        # 텐서 변환 및 GPU로 이동 # 인풋 텐서
        # states 안에 images,goal,action,rewards 들어있음
        images = torch.stack([s['images'] for s in states]).to(self.device)
        goal_vecs = torch.stack([s['goal_vec'] for s in states]).to(self.device)
        actions = torch.stack(actions).to(self.device) # sampling 된 action
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # next staes 안에 images, goal이 들어있어야함.
        next_images = torch.stack([s['images'] for s in next_states]).to(self.device)
        next_goal_vecs = torch.stack([s['goal_vec'] for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        
        
        # 가치 함수 계산
        with torch.no_grad():
            # value function 구할때 goal도 넣어주기
            values = self.value(images, goal_vecs).squeeze()
            next_values = self.value(next_images, next_goal_vecs).squeeze()
            # 둘다 estimator이다.
            
            # GAE 계산
            advantages = self.compute_gae(rewards, values, next_values, dones).to(self.device)
            
            # Policy part에서는 Advantage part에 detach 해줘야함.
            # Value  part에서는 target state에서 detach 해줘야함.
            returns = advantages + values #왜 values를 더하느냐?
            
            # 정규화 # 왜 정규화 해줘야할까? 왜냐하면 reward는 상대적인 값이 므로 정규화해서 너무큰 값이 나오지 않게 해주자.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트 
        for _ in range(self.config.ppo_epochs):
            # 현재 정책으로 로그 확률 계산
            mean, std = self.policy(images, goal_vecs)
            dist = torch.distributions.Normal(mean, std)
            
            # 바뀐 분포에 대한 정확한 계산
            # new_log_probs = dist.log_prob(torch.tanh(actions)).sum(dim=1)
            new_log_probs = dist.log_prob(actions).sum(dim=1) 
            new_log_probs = new_log_probs - torch.log(1-actions.pow(2)+1e-7).sum(dim=1)

            entropy = dist.entropy().mean() # entropy의 의미. 정책의 randomness : exploration
            # 엔트로피가 높으면 다양한 행동을 시도함.
            # 초반엔 탐험윽 촉진 후반엔 정책이 결정적으로 수렴함.
            
            # 비율 계산 # 확률 값 자체는 곱셈 형태라서 매우 작은 값에 빨리 0이됨. 
            #  로그를 쓰면 덧셈으로 바뀌고 계산이 더 안정적임.
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 클리핑 손실
            surr1 = ratio * advantages # 반드시 detach 해줘야함.
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages # advantage가 양수 음수인경우 모두 고려
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 함수 손실
            current_values = self.value(images, goal_vecs).squeeze()
            value_loss = nn.MSELoss()(current_values, returns)
            
            # 총 손실 # entropy term 삭제
            total_loss = policy_loss + self.config.value_coef * value_loss  - self.config.entropy_coef * entropy
            
            # 정책 업데이트
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            # policy 네트워크가 value 네트워크가 입력 및 일부 레이어를 공유할때 첫 backward에서 그래프가 지워지면 backward(value loss) 에서 오류가 남.
            #ㄷ 따라서 첫backward에서 retain_graph=True로 지정함.
            
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            # gradient expolding 을 막음 l2 norm이상으로 커지지않게 잘라줌.
            self.policy_optimizer.step()
            
            # 가치 함수 업데이트
            self.value_optimizer.zero_grad()
            value_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
        
        print(f"Updated - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")

# ============================================================================
# 4. 워커 프로세스 (환경 시뮬레이션 담당)
# ============================================================================
def preprocess_observation(obs):
    """관측 데이터 전처리"""
    # 이미지 데이터 결합 (RGB + Depth + Semantic)
    # rgb = obs["image"][..., -1]  # (H, W, 3)
    depth = obs["depth"][..., -1]  # (H, W, 1) -> (H, W, 3)로 확장
    depth = np.concatenate([depth, depth, depth], axis=-1)
    semantic = obs["semantic"][..., -1]  # (H, W, 3)
    
    # 채널 결합 (H, W, 9)
    # combined_img = np.concatenate([rgb, depth, semantic], axis=-1)
    combined_img = np.concatenate([depth, semantic], axis=-1)
    
    # 정규화 및 채널 순서 변경 (H, W, C) -> (C, H, W)
    combined_img = combined_img.astype(np.float32) / 255.0
    combined_img = np.transpose(combined_img, (2, 0, 1))
    
    # goal_vec 추출
    goal_vec = obs["goal_vec"].astype(np.float32)
    
    return {
        'images': torch.tensor(combined_img),
        'goal_vec': torch.tensor(goal_vec)
    }

def compute_reward(obs, action, next_obs, done, info):
    """보상 함수 설계"""
    reward = 0.0
    
    # 1. 목표 방향으로 이동하는 보상
    goal_vec = next_obs["goal_vec"]
    goal_distance = np.linalg.norm(goal_vec)
    reward += max(0, 1.0 - goal_distance)  # 목표에 가까울수록 높은 보상
    
    # 2. 속도 보상 (적절한 속도 유지)
    speed = info.get('speed', 0)
    reward += min(speed / 10.0, 1.0)  # 속도가 적절할 때 보상
    
    # 3. 충돌/사고 페널티
    if info.get('crash', False):
        reward -= 10.0
    
    # 4. 도로 이탈 페널티
    if info.get('out_of_road', False):
        reward -= 5.0
    
    # 5. 에피소드 완료 보상
    if info.get('arrive_dest', False):
        reward += 20.0
    
    # 6. 시간 페널티 (효율성 유도)
    reward -= 0.01
    
    return reward

# ============================================================================
# 4. 메인 프로세스 (모델 학습 담당)
# ============================================================================
def main():
    """
    메인 프로세스
    - PPO 모델 초기화 및 CUDA 설정
    - 워커들로부터 데이터 수집
    - 모델 학습 수행
    """
    print("Starting Autonomous Driving PPO Training...")
    
    # 설정
    NUM_WORKERS = 48  # 워커 프로세스 수
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # PPO 에이전트 초기화 (메인 프로세스에서만)
    config = PPOConfig()
    agent = PPOAgent(config, device=DEVICE)
    
    # 데이터 큐 생성 (워커 -> 메인)
    data_queue = Queue(maxsize=4096) # thread 안전하여 선택함.
    
    # 환경 설정 
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    
    # 학습 루프
    total_steps = 0
    update_count = 0
    
    max_steps = 1e+2
    for i,step in range(max_steps):
        obs, _ = env.reset()
        
        # goal_vec 추가
        nav = env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        state = preprocess_observation(obs)
        episode_reward = 0
        step_count = 0
        
        while True:
            # 랜덤 행동 선택 (실제로는 메인에서 정책 파라미터를 받아야 함)
            # 여기서는 단순화를 위해 랜덤 행동 사용
            # action = env.action_space.sample()
            action,log_prob = agent.select_action(state,goal_vec=obs["goal_vec"])
            
            # 환경 스텝
            next_obs, _, done, truncated, info = env.step(action)
            # custom reward 사용
            
            # goal_vec 추가
            nav = env.vehicle.navigation.get_navi_info()
            next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            
            # 보상 계산
            reward = compute_reward(obs, action, next_obs, done, info)
            
            # 다음 상태 전처리
            next_state = preprocess_observation(next_obs)
            
            
            # 경험 데이터 생성
            experience = {
                'state': state,
                'action': torch.tensor(action, dtype=torch.float32),
                'reward': reward,
                'next_state': next_state,
                'done': done or truncated,
                'log_prob': torch.tensor(log_prob,dtype=torch.float32)
            }
            
            # 메인 쓰레드 데이터 전송
            try:
                # data_queue.put(experience, timeout=1.0) # 큐에 데이터 전송 큐가 이미 가득차있으면 최대 1초동안 기다림.
                data_queue.put(experience, timeout=0) # 큐에 데이터 전송 큐가 이미 가득차있으면 바로 버림 
            except Full:
                print("Queue full, skipping data")
            
            # if len(data_queue) > 256:
            agent.update() # batch_size가 일정 이상 커졌을때 동작함.
            
            episode_reward += reward
            step_count += 1
            
            # 에피소드 종료 처리
            if done or truncated:
                print(f"Episode finished - Reward: {episode_reward:.2f}, Steps: {step_count}")
                obs, _ = env.reset()
                
                # goal_vec 추가
                nav = env.vehicle.navigation.get_navi_info()
                obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
                
                state = preprocess_observation(obs)
                episode_reward = 0
                step_count = 0
            else:
                obs = next_obs
                state = next_state
            
            # CPU 사용률 조절을 위한 짧은 대기
            time.sleep(0.001)
    

if __name__ == "__main__":
    main()

"""
자율주행 PPO 강화학습 코드 구조:

1. **멀티프로세스 구조**:
    - 메인 프로세스: PPO 모델, CUDA, 학습 담당
    - 워커 프로세스들: 환경 시뮬레이션, 데이터 수집 담당

2. **데이터 플로우**:
    워커(환경) -> mp.Queue -> 메인(버퍼) -> 배치 학습

3. **모델 구조**:
    - CNN 특징 추출기: RGB/Depth/Semantic 이미지 처리
    - PPO 정책 네트워크: 연속 행동 공간 (steering, acceleration)
    - PPO 가치 네트워크: 상태 가치 추정

4. **보상 함수**:
    - 목표 방향 이동 보상
    - 적절한 속도 유지 보상
    - 충돌/도로이탈 페널티
    - 목적지 도달 보상

5. **학습 방식**:
    - 배치 크기만큼 데이터 수집 후 PPO 업데이트
    - GAE를 통한 advantage 계산
    - 클리핑을 통한 안정적 학습

사용법:
python autonomous_driving_ppo.py



### 개선점

동 샘플링과 log_prob 연산 (워커/메인 간 불일치)
문제
현재 워커에서 랜덤하게 action을 선택하고, log_prob에 임시로 0.0을 넣어 큐로 전달하고 있습니다.

PPO의 본질은 현재 정책에서 행동을 샘플링하고, 그에 대응되는 log_prob를 반드시 같이 저장해야 합니다.

log_prob 없이 임의 action을 사용하면 policy gradient 계산이 왜곡되고, 실제 policy 개선이 안 됩니다.

해결
행동 선택과 log_prob 계산을 반드시 메인 프로세스에서 해야 합니다.

워커에서 상태만 수집해서 메인으로 보내고, 메인에서 정책 네트워크로 action, log_prob을 구해서 워커에 전달하는 Actor-Learner 구조를 사용해야 합니다.

만약 워커-메인 간 정책 파라미터 동기화가 느리거나 복잡하다면, policy 네트워크 가중치 복사본을 워커에 전달해서 동기화하도록 해야 함.

이 구조가 번거로우면, action을 워커가 policy로 직접 샘플링하도록 하되, policy 파라미터를 주기적으로 브로드캐스트해야 함.

log_prob은 항상 policy에서 직접 계산한 값으로, 경험과 함께 저장.

참고:
현재 코드는 랜덤 액션 + log_prob=0.0 이므로, PPO 업데이트의 의미가 없음.

2. 멀티프로세스에서의 네트워크 동기화
문제
만약 워커가 정책을 복제해서 행동을 샘플링한다면, 정책 동기화가 필요합니다.

현재는 메인에서만 policy를 관리, 워커는 랜덤행동 사용.

해결
mp.Queue를 통해 워커에게 최신 policy state_dict를 일정 간격마다 브로드캐스트 하거나,
매 에피소드/매 몇 스텝마다 policy.load_state_dict로 동기화.

또는, 메인에서 action을 선택해 워커로 보내는 식으로 환경 interaction loop를 설계 (속도가 떨어질 수 있음).

동 샘플링과 log_prob 연산 (워커/메인 간 불일치)
문제
현재 워커에서 랜덤하게 action을 선택하고, log_prob에 임시로 0.0을 넣어 큐로 전달하고 있습니다.

PPO의 본질은 현재 정책에서 행동을 샘플링하고, 그에 대응되는 log_prob를 반드시 같이 저장해야 합니다.

log_prob 없이 임의 action을 사용하면 policy gradient 계산이 왜곡되고, 실제 policy 개선이 안 됩니다.

해결
행동 선택과 log_prob 계산을 반드시 메인 프로세스에서 해야 합니다.

워커에서 상태만 수집해서 메인으로 보내고, 메인에서 정책 네트워크로 action, log_prob을 구해서 워커에 전달하는 Actor-Learner 구조를 사용해야 합니다.

만약 워커-메인 간 정책 파라미터 동기화가 느리거나 복잡하다면, policy 네트워크 가중치 복사본을 워커에 전달해서 동기화하도록 해야 함.

이 구조가 번거로우면, action을 워커가 policy로 직접 샘플링하도록 하되, policy 파라미터를 주기적으로 브로드캐스트해야 함.

log_prob은 항상 policy에서 직접 계산한 값으로, 경험과 함께 저장.

참고:
현재 코드는 랜덤 액션 + log_prob=0.0 이므로, PPO 업데이트의 의미가 없음.

2. 멀티프로세스에서의 네트워크 동기화
문제
만약 워커가 정책을 복제해서 행동을 샘플링한다면, 정책 동기화가 필요합니다.

현재는 메인에서만 policy를 관리, 워커는 랜덤행동 사용.

해결
mp.Queue를 통해 워커에게 최신 policy state_dict를 일정 간격마다 브로드캐스트 하거나,
매 에피소드/매 몇 스텝마다 policy.load_state_dict로 동기화.

또는, 메인에서 action을 선택해 워커로 보내는 식으로 환경 interaction loop를 설계 (속도가 떨어질 수 있음).


3. 데이터 전송 병목

문제
워커의 수(NUM_WORKERS=48)와 큐 용량이 커서,
data_queue.put(experience, timeout=1.0)에서 병목 또는 데이터 드롭이 발생할 수 있습니다.

특히 queue가 꽉 차면 experience 손실이 심할 수 있음.

해결
Queue full이 자주 발생하면 queue 용량을 조절하거나,
워커 수를 시스템에 맞게 줄여야 합니다.

또는, 경험을 local queue에 쌓고 한번에 put하는 방식도 가능.

4. 배치 샘플링 방식
문제
PPO는 배치 전체를 여러 번 epoch 반복해서 학습하는 것이 기본.

하지만 현재는 buffer를 random.sample로 1회 학습 후 buffer를 clear함.

해결
경험을 여러 epoch 동안 반복적으로 mini-batch로 학습해야 PPO 본연의 효과를 볼 수 있습니다.

for _ in range(ppo_epochs):에서
mini-batch로 데이터를 쪼개서 반복 학습하는 구조로 리팩토링 필요.

5. 학습 루프중 모델 저장, 기록(wandb)

6. randomness 조정 - torch random seed setting

7. 워커 예외 종료 시 재시작/복구 없음
문제 : 워커가 에러로 죽으면 자동 재시작되지 않고, 학습이 점점 느려질 수 있음.

해결 : 워커 헬스 체크, 자동 재시작 코드 추가 고려.


"""