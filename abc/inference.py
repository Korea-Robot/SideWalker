import torch
import numpy as np
from metaurban.envs import SidewalkStaticMetaUrbanEnv

# Import your modules (make sure these are available)
from deploy_env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, PDController
from model import Actor


class InferencePolicy:
    """Inference용 정책 클래스"""
    def __init__(self, actor_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Config 로드 (학습시와 동일한 설정 사용)
        config = Config()
        
        # Actor 모델 초기화 및 가중치 로드
        self.actor = Actor(hidden_dim=config.hidden_dim).to(self.device)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()  # 평가 모드로 설정
        
        print(f"Model loaded on {self.device}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        
    def __call__(self, obs_data: dict) -> tuple[float, float]:
        """관찰을 받아 행동을 출력"""
        # 텐서로 변환
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        # GPU로 이동
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        semantic = semantic.to(self.device)
        goal = goal.to(self.device)
        
        # 추론
        with torch.no_grad():
            action_dist = self.actor(rgb, semantic, depth, goal)
            # 평균값 사용 (deterministic)
            action = action_dist.mean
            
        return tuple(action.cpu().numpy()[0])


def run_single_episode(env, policy, pd_controller, max_steps: int = 512):
    """단일 에피소드 실행"""
    obs, info = env.reset()
    waypoints = env.agent.navigation.checkpoints
    
    # waypoints가 충분하지 않으면 재시작
    while len(waypoints) < 31:
        obs, info = env.reset()
        waypoints = env.agent.navigation.checkpoints
    
    total_reward = 0
    step_count = 0
    
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
        raw_action = policy(obs_data)
        target_angle, throttle = raw_action
        
        # PD 제어를 통해 최종 steering 값 계산
        final_steering = pd_controller.get_control(target_angle, 0)
        final_action = (final_steering, throttle)
        
        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(final_action)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    return total_reward, step_count


def main():
    """메인 inference 함수"""
    # 설정
    actor_path = 'checkpoints/metaurban_actor_epoch_110.pt'  # 학습된 모델 경로
    num_episodes = 10
    max_steps = 512
    
    print(f"Starting inference with {num_episodes} episodes...")
    print(f"Actor model path: {actor_path}")
    
    # 환경 및 정책 초기화
    env_config = EnvConfig()
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
    
    policy = InferencePolicy(actor_path)
    pd_controller = PDController(p_gain=0.5, d_gain=0.3)
    
    # 여러 에피소드 실행
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}...")
        
        total_reward, steps = run_single_episode(env, policy, pd_controller, max_steps)
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        print(f"  Episode {episode + 1}: Reward = {total_reward:.3f}, Steps = {steps}")
    
    # 결과 통계
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    median_reward = np.median(episode_rewards)
    mean_steps = np.mean(episode_steps)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Median Reward: {median_reward:.3f}")
    print(f"Min Reward: {min(episode_rewards):.3f}")
    print(f"Max Reward: {max(episode_rewards):.3f}")
    print(f"Mean Steps: {mean_steps:.1f}")
    print("="*50)
    
    # 개별 에피소드 결과
    print("\nDetailed Results:")
    for i, (reward, steps) in enumerate(zip(episode_rewards, episode_steps)):
        print(f"Episode {i+1:2d}: Reward = {reward:7.3f}, Steps = {steps:3d}")
    
    env.close()
    print("\nInference completed!")


if __name__ == "__main__":
    main()