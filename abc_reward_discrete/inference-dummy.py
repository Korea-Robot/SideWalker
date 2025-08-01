import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torchvision.models as models

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Import configurations and utilities
from utils import convert_to_egocentric, extract_sensor_data, PDController
from deploy_env_config import EnvConfig

# Simple replacements for complex encoders
class SimpleDINOEncoder(nn.Module):
    """Simple CNN replacement for DINOv2 - returns 768 dim features"""
    def __init__(self, output_dim=768):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Global average pooling + FC to get 768 features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleSegFormerEncoder(nn.Module):
    """Simple encoder replacement for SegFormer"""
    def __init__(self):
        super().__init__()
        # Simple encoder that outputs feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # Return dict to match SegFormer API
        features = self.encoder(x)
        return {'last_hidden_state': features}

class SimplePerceptNet(nn.Module):
    """Simple encoder replacement for PerceptNet"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.encoder(x)

class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with simple CNN replacements
    """
    def __init__(self, hidden_dim=512, fusion_strategy='attention'):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy
        
        # Initialize encoders
        self._init_rgb_encoders()
        self._init_semantic_encoder()  
        self._init_depth_encoder()
        self._init_goal_encoder()
        self._init_fusion_layers()
        
    def _init_rgb_encoders(self):
        """Initialize simple RGB encoders"""
        print("Initializing simple RGB encoders...")
        
        # 1. Simple DINO replacement
        self.dino_encoder = SimpleDINOEncoder(output_dim=768)
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        self.dino_encoder.eval()
        
        # 2. Simple SegFormer replacement  
        self.segformer = SimpleSegFormerEncoder()
        for param in self.segformer.parameters():
            param.requires_grad = False
        self.segformer.eval()
        
        # RGB feature projectors
        self.dino_projector = nn.Sequential(
            nn.Linear(768, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # SegFormer feature extraction from encoder
        self.segformer_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_semantic_encoder(self):
        """Initialize semantic segmentation encoder: EfficientNet"""
        print("Initializing semantic encoder...")
        
        try:
            # Try new style weights first
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except:
            # Fallback to old style
            efficientnet = models.efficientnet_b0(pretrained=True)
            
        self.semantic_encoder = efficientnet.features
        
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False
        self.semantic_encoder.eval()
        
        self.semantic_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_depth_encoder(self):
        """Initialize simple depth encoder"""
        print("Initializing simple depth encoder...")
        
        self.depth_encoder = SimplePerceptNet()
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        self.depth_encoder.eval()
        
        self.depth_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_goal_encoder(self):
        """Initialize goal encoder"""
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _init_fusion_layers(self):
        """Initialize feature fusion layers"""
        if self.fusion_strategy == 'attention':
            # Multi-head attention for feature fusion
            self.feature_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Learnable query for feature aggregation
            self.query_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            
        # Final fusion layers
        if self.fusion_strategy == 'attention':
            fusion_input_dim = self.hidden_dim
        else:
            fusion_input_dim = self.hidden_dim * 4  # rgb_dino, semantic, depth, goal
            
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, rgb, semantic, depth, goal):
        batch_size = rgb.shape[0]
        
        # Resize RGB to a compatible size
        rgb_resized = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract features from each modality
        # RGB features via Simple DINO
        with torch.no_grad():
            dino_features = self.dino_encoder(rgb_resized)  # [B, 768]
        dino_features = self.dino_projector(dino_features)  # [B, hidden_dim]
        
        # RGB spatial features via Simple SegFormer
        with torch.no_grad():
            segformer_output = self.segformer(rgb)
            segformer_features = segformer_output['last_hidden_state']  # [B, 256, H/4, W/4]
        segformer_features = self.segformer_projector(segformer_features)  # [B, hidden_dim]
        
        # Combine RGB features (DINO + SegFormer)
        rgb_combined = (dino_features + segformer_features) / 2
        
        # Semantic features via EfficientNet
        with torch.no_grad():
            semantic_features = self.semantic_encoder(semantic)  # [B, 1280, H/32, W/32]
        semantic_features = self.semantic_projector(semantic_features)  # [B, hidden_dim]
        
        # Depth features via Simple PerceptNet
        with torch.no_grad():
            depth_features = self.depth_encoder(depth)  # [B, 512, H/32, W/32]
        depth_features = self.depth_projector(depth_features)  # [B, hidden_dim]
        
        # Goal features
        goal_features = self.goal_encoder(goal)  # [B, hidden_dim]
        
        # Feature fusion
        if self.fusion_strategy == 'attention':
            features = torch.stack([rgb_combined, semantic_features, depth_features, goal_features], dim=1)
            query = self.query_token.expand(batch_size, -1, -1)
            fused_features, _ = self.feature_attention(query, features, features)
            fused_features = fused_features.squeeze(1)
        else:  # simple concatenation
            fused_features = torch.cat([rgb_combined, semantic_features, depth_features, goal_features], dim=1)
        
        # Final fusion
        output = self.fusion_layers(fused_features)
        
        return output


class Actor(nn.Module):
    """Discrete Actor network with categorical distribution"""
    def __init__(self, hidden_dim=512, num_steering_actions=5, num_throttle_actions=3):
        super().__init__()
        
        # Create shared encoder
        self.encoder = MultiModalEncoder(hidden_dim=hidden_dim, fusion_strategy='attention')
        
        # Discrete action space definition
        self.num_steering_actions = num_steering_actions
        self.num_throttle_actions = num_throttle_actions
        
        # Action mappings (register as buffers, not parameters)
        self.register_buffer('steering_actions', torch.linspace(-1.0, 1.0, num_steering_actions))
        self.register_buffer('throttle_actions', torch.linspace(0.5, 1.0, num_throttle_actions))
        
        # Separate networks for steering and throttle
        self.steering_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_steering_actions)
        )
        
        self.throttle_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_throttle_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
            
    def forward(self, rgb, semantic, depth, goal):
        z = self.encoder(rgb, semantic, depth, goal)
        
        # Get logits for each action dimension
        steering_logits = self.steering_net(z)
        throttle_logits = self.throttle_net(z)
        
        # Create categorical distributions
        steering_dist = torch.distributions.Categorical(logits=steering_logits)
        throttle_dist = torch.distributions.Categorical(logits=throttle_logits)
        
        return steering_dist, throttle_dist
    
    def get_action_values(self, steering_indices, throttle_indices, device):
        """Convert discrete indices to continuous action values"""
        steering_values = self.steering_actions[steering_indices].to(device)
        throttle_values = self.throttle_actions[throttle_indices].to(device)
        return steering_values, throttle_values
    
    def sample_action(self, rgb, semantic, depth, goal):
        """Sample action and return both indices and continuous values"""
        steering_dist, throttle_dist = self.forward(rgb, semantic, depth, goal)
        
        steering_idx = steering_dist.sample()
        throttle_idx = throttle_dist.sample()
        
        device = rgb.device
        steering_val, throttle_val = self.get_action_values(steering_idx, throttle_idx, device)
        
        return (steering_idx, throttle_idx), (steering_val, throttle_val)


class DiscreteNNPolicy:
    """Discrete Neural Network Policy for Inference"""
    def __init__(self, actor: Actor):
        self.actor = actor
        
    def __call__(self, obs_data: dict) -> tuple[tuple[int, int], tuple[float, float]]:
        device = next(self.actor.parameters()).device
        
        # Convert observation to tensors
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        rgb = rgb.to(device)
        depth = depth.to(device)
        semantic = semantic.to(device)
        goal = goal.to(device)
        
        with torch.no_grad():
            self.actor.eval()
            (steering_idx, throttle_idx), (steering_val, throttle_val) = self.actor.sample_action(rgb, semantic, depth, goal)
        
        return (int(steering_idx.cpu().item()), int(throttle_idx.cpu().item())), (float(steering_val.cpu().item()), float(throttle_val.cpu().item()))

def calculate_all_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """Goal-position based navigation을 위한 최적화된 리워드 함수"""
    rewards = defaultdict(float)
    
    agent_pos = env.agent.position
    agent_heading = env.agent.heading_theta
    speed = info.get('speed', 0)
    nav = env.agent.navigation
    waypoints = nav.checkpoints
    
    # 1. 목표 근접 보상 (Goal Proximity Reward)
    if prev_info:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        if current_dist < prev_dist:
            rewards['goal_proximity'] = (prev_dist - current_dist) * 10.0

    # 2. 지능적 체크포인트 진행 보상 (Smart Checkpoint Progress)
    if prev_info and 'closest_checkpoint_idx' in prev_info:
        prev_closest = prev_info.get('closest_checkpoint_idx', 0)
        current_closest = info.get('closest_checkpoint_idx', 0)
        
        if current_closest > prev_closest:
            progress_steps = current_closest - prev_closest
            # 연속적으로 여러 체크포인트를 통과하면 보너스
            bonus_multiplier = 1.0 + (progress_steps - 1) * 0.3
            rewards['checkpoint_progress'] = 8.0 * progress_steps * bonus_multiplier
    
    # 3. 방향 정렬 보상 (Directional Alignment)
    if len(waypoints) > 0:
        look_ahead = min(3, len(waypoints) - 1)
        if look_ahead > 0:
            target_pos = waypoints[look_ahead]
            direction_to_target = np.arctan2(target_pos[1] - agent_pos[1], 
                                           target_pos[0] - agent_pos[0])
            
            heading_diff = abs(agent_heading - direction_to_target)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            
            alignment_score = 1.0 - (heading_diff / np.pi)
            rewards['direction_alignment'] = max(0, alignment_score) * 2.0

    # 4. 성공 보상 (Success Reward)
    if info.get('arrive_dest', False):
        rewards['success_reward'] = 50.0

    # 5. 충돌 페널티 (Collision Penalty)
    if info.get('crash_vehicle', False) or info.get('crash_object', False):
        rewards['collision_penalty'] = -20.0
    
    # 6. 차선 이탈 페널티 (Out of Road Penalty)
    if info.get('out_of_road', False):
        rewards['out_of_road_penalty'] = -15.0

    # 7. 기본 환경 보상 (Environment's Default Reward)
    rewards['env_default_reward'] = info.get('original_reward', 0)

    return rewards

def run_single_episode(env, policy, max_steps: int = 512) -> dict:
    """단일 에피소드 실행 및 성능 측정"""
    obs, info = env.reset()
    waypoints = env.agent.navigation.checkpoints
    
    # 일정 거리 이상의 waypoints가 없을 경우, 환경을 재설정
    while len(waypoints) < 31:
        obs, info = env.reset()
        waypoints = env.agent.navigation.checkpoints
    
    total_reward = 0
    step_count = 0
    prev_info = None
    pd_controller = PDController(p_gain=0.5, d_gain=0.3)
    
    episode_stats = {
        'total_reward': 0,
        'steps': 0,
        'success': False,
        'crash': False,
        'out_of_road': False,
        'final_distance_to_goal': 0
    }
    
    while step_count < max_steps:
        # 목표 지점 계산
        ego_goal_position = np.array([0.0, 0.0])
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 관찰 데이터 준비
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        obs_data = {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic': semantic_data,
            'goal': ego_goal_position
        }
        
        # 행동 선택
        discrete_action, continuous_action = policy(obs_data)
        target_angle, throttle = continuous_action

        # PD 제어를 통해 최종 steering 값 계산
        final_steering = pd_controller.get_control(target_angle, 0) 
        final_action = (final_steering, throttle)

        # 환경 스텝
        obs, env_reward, terminated, truncated, info = env.step(final_action)
        
        # 리워드 계산
        reward_dict = calculate_all_rewards(info, prev_info, continuous_action, env)
        step_reward = sum(reward_dict.values())
        total_reward += step_reward
        
        prev_info = info.copy()
        step_count += 1
        
        # 종료 조건 체크
        if info.get('arrive_dest', False):
            episode_stats['success'] = True
            break
        if info.get('crash_vehicle', False) or info.get('crash_object', False):
            episode_stats['crash'] = True
            break
        if info.get('out_of_road', False):
            episode_stats['out_of_road'] = True
            break
        if terminated or truncated:
            break
    
    episode_stats['total_reward'] = total_reward
    episode_stats['steps'] = step_count
    episode_stats['final_distance_to_goal'] = info.get('distance_to_goal', 0)
    
    return episode_stats

def main():
    """메인 inference 함수"""
    
    # 디바이스 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 초기화 (원래 저장된 모델과 동일한 차원)
    actor = Actor(hidden_dim=512, num_steering_actions=5, num_throttle_actions=3).to(device)
    
    # 체크포인트 로드 시도
    checkpoint_path = 'metaurban_discrete_actor_multimodal_final.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 불필요한 키들 제거
        keys_to_remove = ['steering_actions', 'throttle_actions']
        for key in keys_to_remove:
            if key in checkpoint:
                del checkpoint[key]
        
        # 현재 모델과 호환되지 않는 부분은 건너뛰고 부분 로드
        model_dict = actor.state_dict()
        pretrained_dict = {}
        
        print("Attempting to load checkpoint...")
        for k, v in checkpoint.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    print(f"✓ Loaded: {k}")
                else:
                    print(f"✗ Shape mismatch for {k}: model={model_dict[k].shape}, checkpoint={v.shape}")
            else:
                print(f"✗ Key {k} not found in current model")
        
        # 호환되는 부분만 업데이트
        model_dict.update(pretrained_dict)
        actor.load_state_dict(model_dict)
        
        print(f"\nSuccessfully loaded {len(pretrained_dict)} out of {len(checkpoint)} parameters")
        if len(pretrained_dict) > 0:
            print("Using partially pre-trained model")
        else:
            print("No compatible parameters found, using randomly initialized model")
        
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        print("Using randomly initialized model...")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model...")
    
    actor.eval()
    
    # 정책 초기화
    policy = DiscreteNNPolicy(actor)
    
    # 환경 초기화
    env_config = EnvConfig()
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
    
    # 10번 주행 테스트
    num_episodes = 10
    all_stats = []
    
    print(f"\nRunning {num_episodes} episodes for evaluation...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}...")
        
        episode_stats = run_single_episode(env, policy, max_steps=512)
        all_stats.append(episode_stats)
        
        print(f"  Reward: {episode_stats['total_reward']:.2f}, "
              f"Steps: {episode_stats['steps']}, "
              f"Success: {episode_stats['success']}, "
              f"Crash: {episode_stats['crash']}, "
              f"Out of Road: {episode_stats['out_of_road']}")
    
    # 결과 분석
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    rewards = [stats['total_reward'] for stats in all_stats]
    steps = [stats['steps'] for stats in all_stats]
    successes = [stats['success'] for stats in all_stats]
    crashes = [stats['crash'] for stats in all_stats]
    out_of_roads = [stats['out_of_road'] for stats in all_stats]
    
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Median Reward: {np.median(rewards):.2f}")
    print(f"Min/Max Reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Success Rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{num_episodes})")
    print(f"Crash Rate: {np.mean(crashes)*100:.1f}% ({sum(crashes)}/{num_episodes})")
    print(f"Out of Road Rate: {np.mean(out_of_roads)*100:.1f}% ({sum(out_of_roads)}/{num_episodes})")
    
    # 개별 에피소드 상세 결과
    print(f"\nDetailed Results:")
    print(f"{'Episode':<8} {'Reward':<10} {'Steps':<6} {'Success':<8} {'Crash':<6} {'OOR':<6}")
    print("-" * 50)
    for i, stats in enumerate(all_stats):
        print(f"{i+1:<8} {stats['total_reward']:<10.2f} {stats['steps']:<6} "
              f"{'Yes' if stats['success'] else 'No':<8} "
              f"{'Yes' if stats['crash'] else 'No':<6} "
              f"{'Yes' if stats['out_of_road'] else 'No':<6}")
    
    env.close()
    print("\nInference completed!")

if __name__ == "__main__":
    main()