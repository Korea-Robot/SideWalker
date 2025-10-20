

import torch
import numpy as np
import typing
import math
from collections import defaultdict

from metaurban.envs import SidewalkStaticMetaUrbanEnv

# Import configurations, utilities, and models
from deploy_env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, PDController
from model import Actor

# --- Configuration ---
MODEL_PATH = 'metaurban_actor_multimodal_final.pt' # 사용할 액터 모델 경로
HORIZON = 200  # 시뮬레이션할 스텝 수
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---
def deviceof(m: torch.nn.Module) -> torch.device:
    """모듈의 device 반환"""
    return next(m.parameters()).device

class NNPolicy:
    """저장된 모델을 사용하는 Neural Network Policy"""
    def __init__(self, actor: Actor):
        self.actor = actor
        self.actor.eval()

    def __call__(self, obs_data: dict) -> tuple[float, float]:
        device = deviceof(self.actor)
        
        # Convert observation to tensors
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        rgb, depth, semantic, goal = rgb.to(device), depth.to(device), semantic.to(device), goal.to(device)
        
        with torch.no_grad():
            action_dist = action_dist = self.actor(rgb, semantic, depth, goal)
            action = action_dist.sample()
        
        return tuple(action.cpu().numpy()[0])

def calculate_all_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """
    한 스텝에서 가능한 모든 리워드 구성 요소를 계산합니다.
    """
    rewards = defaultdict(float)

    # 1. 목표 근접 보상 (Goal Proximity Reward)
    # 목표까지의 거리가 이전 스텝보다 가까워졌으면 보상
    if prev_info:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        rewards['goal_proximity'] = (prev_dist - current_dist) * 10.0 # 스케일링 팩터

    # 2. 속도 보상 (Speed Reward)
    # 목표 속도(예: 1.0 m/s)에 가까울수록 보상
    target_speed = 1.0
    speed = info.get('speed', 0)
    rewards['speed_reward'] = 1.0 - abs(speed - target_speed)

    # 3. 성공 보상 (Success Reward)
    if info.get('arrive_dest', False):
        rewards['success_reward'] = 50.0

    # 4. 충돌 페널티 (Collision Penalty)
    if info.get('crash_vehicle', False) or info.get('crash_object', False):
        rewards['collision_penalty'] = -10.0
    
    # 5. 차선 이탈 페널티 (Out of Road Penalty)
    if info.get('out_of_road', False):
        rewards['out_of_road_penalty'] = -5.0

    # 6. 과도한 조향 페널티 (Steering Penalty)
    # steering 값(action[0])의 절대값이 클수록 페널티
    steering_effort = abs(action[0])
    rewards['steering_penalty'] = -0.1 * steering_effort # 스케일링 팩터

    # 7. 기본 환경 보상 (Environment's Default Reward)
    # env.step()에서 반환된 원래 보상
    rewards['env_default_reward'] = info.get('original_reward', 0)

    return rewards

def main():
    """메인 시뮬레이션 및 리워드 분석 함수"""
    print(f"Using device: {DEVICE}")

    # --- 초기화 ---
    config = Config()
    env_config = EnvConfig()
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
    
    # 모델 불러오기
    try:
        actor = Actor(hidden_dim=config.hidden_dim).to(DEVICE)
        actor.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Actor model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    policy = NNPolicy(actor)
    pd_controller = PDController(p_gain=0.5, d_gain=0.3)

    # --- 시뮬레이션 시작 ---
    obs, info = env.reset()
    
    # Waypoint가 충분히 생성될 때까지 리셋
    while len(env.agent.navigation.checkpoints) < 31:
        obs, info = env.reset()

    reward_history = []
    prev_info = {}
    
    print(f"\n--- Running simulation for {HORIZON} steps ---")

    for step in range(HORIZON):
        # 목표 지점 계산
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
            info['distance_to_goal'] = np.linalg.norm(global_target - agent_pos)
        else:
            ego_goal_position = np.array([0.0, 0.0])
            info['distance_to_goal'] = prev_info.get('distance_to_goal', 0) # 이전 거리 유지

        # 관찰 데이터 준비
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        obs_data = {
            'rgb': rgb_data, 'depth': depth_data, 
            'semantic': semantic_data, 'goal': ego_goal_position
        }

        # 행동 선택 및 PD 제어
        raw_action = policy(obs_data)
        target_angle, throttle = raw_action
        final_steering = pd_controller.get_control(target_angle, 0)
        final_action = (final_steering, throttle)

        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(final_action)
        info['original_reward'] = reward # 원래 보상을 info에 저장

        # 모든 리워드 계산 및 저장
        step_rewards = calculate_all_rewards(info, prev_info, raw_action, env)
        reward_history.append(step_rewards)
        
        prev_info = info.copy()

        print(f"Step {step+1}/{HORIZON} | Throttle: {throttle:.2f}, Steering(Final): {final_steering:.2f} | Env Reward: {reward:.3f}")

        if terminated or truncated:
            print(f"Episode finished early at step {step+1}.")
            break
    
    env.close()

    # --- 결과 분석 및 출력 ---
    print("\n--- Reward Analysis ---")
    
    total_rewards = defaultdict(float)
    trigger_counts = defaultdict(int)

    for step_rewards in reward_history:
        for name, value in step_rewards.items():
            total_rewards[name] += value
            if value != 0:
                trigger_counts[name] += 1
    
    print(f"{'Reward Component':<25} | {'Total Value':>15} | {'Avg Value':>15} | {'Triggered Steps':>18}")
    print("-" * 80)

    for name in sorted(total_rewards.keys()):
        total = total_rewards[name]
        count = trigger_counts[name]
        avg = total / count if count > 0 else 0
        
        print(f"{name:<25} | {total:>15.3f} | {avg:>15.3f} | {count:>10} / {len(reward_history)}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()

