# 파일명: run_inference.py
# 설명: 학습된 모델을 로드하여 추론을 실행하고, 결과를 분석 및 시각화합니다.
# 실행 방법: python run_inference.py

import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from metaurban.envs import SidewalkStaticMetaUrbanEnv

# 핵심 컴포넌트 및 모델 임포트
from core_components import (
    BASE_ENV_CFG, Actor, convert_to_egocentric, extract_sensor_data
)

class MetaUrbanInference:
    """학습된 모델로 추론하는 클래스"""
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(hidden_dim=512).to(self.device)
        
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using a randomly initialized model.")
        self.actor.eval()

    def obs_to_tensor(self, obs_data):
        """관찰을 텐서로 변환"""
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        if obs_data['depth'] is not None:
            depth = torch.tensor(obs_data['depth'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else: # Depth가 없을 경우 RGB의 평균값으로 더미 데이터 생성
            depth = torch.mean(rgb, dim=1, keepdim=True)
            
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        return rgb.to(self.device), depth.to(self.device), goal.to(self.device)

    def predict_action(self, obs_data, deterministic=True):
        """관찰에서 행동 예측"""
        rgb_tensor, depth_tensor, goal_tensor = self.obs_to_tensor(obs_data)
        with torch.no_grad():
            action_dist = self.actor(rgb_tensor, depth_tensor, goal_tensor)
            action = action_dist.mean[0] if deterministic else action_dist.sample()[0]
        return action.cpu().numpy()
        
    def run(self, env, max_steps=500):
        """추론 실행"""
        obs, info = env.reset()
        total_reward, step_count = 0, 0
        trajectory = []
        
        print("=== 추론 시작 ===")
        while step_count < max_steps:
            rgb, depth, _ = extract_sensor_data(obs)
            if rgb is None: break
                
            ego_goal = np.array([0.0, 0.0])
            if hasattr(env.agent, 'navigation') and env.agent.navigation:
                waypoints = env.agent.navigation.checkpoints
                k = min(15, len(waypoints) - 1) if waypoints else 0
                if len(waypoints) > k:
                    ego_goal = convert_to_egocentric(waypoints[k], env.agent.position, env.agent.heading_theta)
            
            obs_data = {'rgb': rgb, 'depth': depth, 'goal': ego_goal}
            action = self.predict_action(obs_data)
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            trajectory.append({'position': env.agent.position, 'reward': reward})

            if (step_count % 50 == 0):
                print(f"Step {step_count}: Action={action}, Reward={reward:.3f}")

            if terminated or truncated:
                print(f"Episode finished after {step_count} steps.")
                break
        
        print(f"=== 추론 완료 | Total Reward: {total_reward:.2f} ===")
        return trajectory, total_reward

def analyze_trajectory(trajectory):
    """궤적 및 보상 그래프 시각화"""
    if not trajectory:
        print("No trajectory data to analyze.")
        return

    positions = np.array([step['position'] for step in trajectory])
    rewards = [step['reward'] for step in trajectory]

    plt.figure(figsize=(14, 6))
    
    # 궤적 플롯
    plt.subplot(1, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    plt.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start', zorder=5)
    plt.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End', zorder=5)
    plt.title('Agent Trajectory')
    plt.xlabel('X Position'); plt.ylabel('Y Position')
    plt.legend(); plt.grid(True); plt.axis('equal')

    # 보상 플롯
    plt.subplot(1, 2, 2)
    plt.plot(rewards, 'r-')
    plt.title('Reward per Step')
    plt.xlabel('Step'); plt.ylabel('Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_analysis.png')
    plt.show()

def main():
    """메인 실행 함수"""
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    inference = MetaUrbanInference(model_path='metaurban_actor.pt') # 모델 파일 경로
    
    try:
        trajectory, total_reward = inference.run(env, max_steps=200)
        analyze_trajectory(trajectory)
    finally:
        env.close()
    
    print("완료!")

if __name__ == "__main__":
    main()