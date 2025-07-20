import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# MetaUrban 환경 및 관련 구성 요소들을 가져옵니다.
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 설정 (Configurations) ---

SENSOR_SIZE = (256, 160)  # 카메라 센서의 해상도 (너비, 높이)
BASE_ENV_CFG = dict(
    use_render=False,           # 학습 시에는 렌더링 비활성화
    map='X',                    
    manual_control=False,       
    crswalk_density=1,          
    object_density=0.1,         
    walk_on_all_regions=False,  
    drivable_area_extension=55, 
    height_scale=1,             
    horizon=1000,               
    
    vehicle_config=dict(enable_reverse=True),
    
    show_sidewalk=True,         
    show_crosswalk=True,        
    random_lane_width=True,     
    random_agent_model=True,    
    random_lane_num=True,       
    
    random_spawn_lane_index=False,
    num_scenarios=100,             
    accident_prob=0,               
    max_lateral_dist=5.0,          
    
    agent_type='coco',             
    
    relax_out_of_road_done=False,  
    
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,                      
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# --- DQN 하이퍼파라미터 ---
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.0001

# --- 행동 공간 정의 ---
ACTIONS = [
    [0, 1.0],   # 직진
    [0, -1.0],  # 후진/브레이크
    [0.5, 0.5], # 좌회전하며 전진
    [-0.5, 0.5], # 우회전하며 전진
    [0, 0]      # 정지
]
N_ACTIONS = len(ACTIONS)

# --- 유틸리티 함수 ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계의 목표 지점을 에이전트 중심의 자기 좌표계로 변환"""
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])
    
    ego_vector = rotation_matrix @ vec_in_world
    return ego_vector

# --- 이미지 전처리 ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # EfficientNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_depth_image(depth_img):
    """Depth 이미지를 전처리하여 3채널로 변환"""
    # Depth 이미지를 정규화
    depth_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
    # 3채널로 복사
    depth_rgb = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=2)
    # 0-255 범위로 변환
    depth_rgb = (depth_rgb * 255).astype(np.uint8)
    return transform(depth_rgb)

def preprocess_semantic_image(semantic_img):
    """Semantic 이미지를 전처리"""
    # Semantic 이미지가 이미 3채널이라고 가정
    semantic_rgb = (semantic_img * 255).astype(np.uint8)
    return transform(semantic_rgb)

# --- 경험 리플레이 메모리 ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# --- DQN 네트워크 ---
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        # EfficientNet-B0 백본 (depth와 semantic 이미지용)
        self.depth_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.semantic_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 마지막 분류 레이어 제거하여 feature만 추출
        self.depth_backbone._fc = nn.Identity()
        self.semantic_backbone._fc = nn.Identity()
        
        # EfficientNet-B0의 feature 차원은 1280
        efficientnet_features = 1280
        
        # Goal position (2D)
        goal_dim = 2
        
        # 전체 feature 차원
        total_features = efficientnet_features * 2 + goal_dim  # depth + semantic + goal
        
        # DQN 헤드
        self.fc1 = nn.Linear(total_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, depth_img, semantic_img, goal_pos):
        # Feature 추출
        depth_features = self.depth_backbone(depth_img)
        semantic_features = self.semantic_backbone(semantic_img)
        
        # Feature 결합
        combined_features = torch.cat([
            depth_features, 
            semantic_features, 
            goal_pos
        ], dim=1)
        
        # DQN 헤드 통과
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# --- 학습 함수 ---
def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    # 배치 데이터 준비
    depth_batch = torch.stack([s[0] for s in batch[0]]).to(device)
    semantic_batch = torch.stack([s[1] for s in batch[0]]).to(device)
    goal_batch = torch.stack([s[2] for s in batch[0]]).to(device)
    
    action_batch = torch.tensor(batch[1], dtype=torch.long).to(device)
    reward_batch = torch.tensor(batch[3], dtype=torch.float).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.bool).to(device)
    
    # 다음 상태 배치 (None이 아닌 경우만)
    non_final_mask = torch.tensor([s is not None for s in batch[2]], dtype=torch.bool).to(device)
    
    if non_final_mask.any():
        non_final_next_depth = torch.stack([s[0] for s in batch[2] if s is not None]).to(device)
        non_final_next_semantic = torch.stack([s[1] for s in batch[2] if s is not None]).to(device)
        non_final_next_goal = torch.stack([s[2] for s in batch[2] if s is not None]).to(device)
    
    # 현재 Q 값 계산
    current_q_values = policy_net(depth_batch, semantic_batch, goal_batch).gather(1, action_batch.unsqueeze(1))
    
    # 다음 Q 값 계산
    next_q_values = torch.zeros(BATCH_SIZE).to(device)
    if non_final_mask.any():
        next_q_values[non_final_mask] = target_net(
            non_final_next_depth, 
            non_final_next_semantic, 
            non_final_next_goal
        ).max(1)[0].detach()
    
    # 타겟 Q 값 계산
    target_q_values = reward_batch + (GAMMA * next_q_values * ~done_batch)
    
    # 손실 계산
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
    
    # 옵티마이저 업데이트
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# --- 메인 학습 코드 ---
def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 환경 초기화
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    
    # 네트워크 초기화
    policy_net = DQN(N_ACTIONS).to(device)
    target_net = DQN(N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # 옵티마이저 및 메모리 초기화
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # 학습 통계
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # 학습 루프
    num_episodes = 1000
    steps_done = 0
    
    for episode in range(num_episodes):
        # 환경 리셋 (waypoints가 16개 이상인 환경만 사용)
        valid_env = False
        reset_attempts = 0
        max_reset_attempts = 50
        
        while not valid_env and reset_attempts < max_reset_attempts:
            obs, _ = env.reset(seed=episode + 1)
            nav = env.agent.navigation
            waypoints = nav.checkpoints
            
            if len(waypoints) >= 16:
                valid_env = True
            else:
                reset_attempts += 1
        
        if not valid_env:
            print(f"Episode {episode}: Could not find valid environment with 16+ waypoints")
            continue
        
        # 초기 상태 준비
        depth_img = obs['image']['depth_camera']
        semantic_img = obs['image']['semantic_camera']
        
        # Goal position 계산
        k = 15  # 15번째 waypoint를 목표로 설정
        global_target = waypoints[k]
        agent_pos = env.agent.position
        agent_heading = env.agent.heading_theta
        ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 이미지 전처리
        depth_tensor = preprocess_depth_image(depth_img).unsqueeze(0).to(device)
        semantic_tensor = preprocess_semantic_image(semantic_img).unsqueeze(0).to(device)
        goal_tensor = torch.tensor(ego_goal_position, dtype=torch.float).unsqueeze(0).to(device)
        
        current_state = (depth_tensor.squeeze(0), semantic_tensor.squeeze(0), goal_tensor.squeeze(0))
        
        episode_reward = 0
        episode_length = 0
        
        for t in range(1000):  # 최대 1000 스텝
            # 행동 선택 (epsilon-greedy)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    q_values = policy_net(depth_tensor, semantic_tensor, goal_tensor)
                    action_idx = q_values.max(1)[1].item()
            else:
                action_idx = random.randrange(N_ACTIONS)
            
            # 행동 실행
            action = ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 다음 상태 준비
            if not (terminated or truncated):
                next_depth_img = obs['image']['depth_camera']
                next_semantic_img = obs['image']['semantic_camera']
                
                # 다음 goal position 계산
                nav = env.agent.navigation
                waypoints = nav.checkpoints
                if len(waypoints) >= 16:
                    next_global_target = waypoints[15]
                    next_agent_pos = env.agent.position
                    next_agent_heading = env.agent.heading_theta
                    next_ego_goal_position = convert_to_egocentric(next_global_target, next_agent_pos, next_agent_heading)
                    
                    next_depth_tensor = preprocess_depth_image(next_depth_img).to(device)
                    next_semantic_tensor = preprocess_semantic_image(next_semantic_img).to(device)
                    next_goal_tensor = torch.tensor(next_ego_goal_position, dtype=torch.float).to(device)
                    
                    next_state = (next_depth_tensor, next_semantic_tensor, next_goal_tensor)
                else:
                    next_state = None
            else:
                next_state = None
            
            # 경험 저장
            memory.push(current_state, action_idx, next_state, reward, terminated or truncated)
            
            # 상태 업데이트
            if next_state is not None:
                current_state = next_state
                depth_tensor = next_depth_tensor.unsqueeze(0)
                semantic_tensor = next_semantic_tensor.unsqueeze(0)
                goal_tensor = next_goal_tensor.unsqueeze(0)
            
            # 모델 최적화
            loss = optimize_model(policy_net, target_net, optimizer, memory, device)
            if loss is not None:
                losses.append(loss)
            
            if terminated or truncated:
                break
        
        # 통계 업데이트
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 타겟 네트워크 업데이트
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 진행 상황 출력
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                  f"Avg Loss: {avg_loss:.4f}, Epsilon: {eps_threshold:.3f}")
        
        # 모델 저장
        if episode % 100 == 0 and episode > 0:
            torch.save({
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'losses': losses
            }, f'dqn_checkpoint_episode_{episode}.pth')
    
    # 최종 모델 저장
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses
    }, 'dqn_final_model.pth')
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()