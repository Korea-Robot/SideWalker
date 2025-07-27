import torch
import numpy as np
import typing
import math
from collections import defaultdict
import pygame

from metaurban.envs import SidewalkStaticMetaUrbanEnv

# Import configurations and utilities
from deploy_env_config import EnvConfig

# --- Configuration ---
HORIZON = 500  # 최대 시뮬레이션 스텝 수

ACTION_MAP = {
    pygame.K_w: [0, 1.0],   # 전진
    pygame.K_s: [0, -1.0],  # 후진/브레이크
    pygame.K_a: [0.5, 0.5], # 좌회전
    pygame.K_d: [-0.5, 0.5]  # 우회전
}

def calculate_all_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """
    한 스텝에서 가능한 모든 리워드 구성 요소를 계산합니다.
    """
    rewards = defaultdict(float)
    
    # 1. 목표 근접 보상 (Goal Proximity Reward)
    if prev_info:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        if current_dist < prev_dist:
            rewards['goal_proximity'] = (prev_dist - current_dist) * 10.0

    # 2. 속도 보상 (Speed Reward)
    target_speed = 1.0
    speed = info.get('speed', 0)
    rewards['speed_reward'] = 1.0 - abs(speed - target_speed) if speed > 0.1 else 0

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
    steering_effort = abs(action[0])
    rewards['steering_penalty'] = -0.1 * steering_effort

    # 7. 기본 환경 보상 (Environment's Default Reward)
    rewards['env_default_reward'] = info.get('original_reward', 0)

    return rewards

def main():
    """메인 시뮬레이션 및 리워드 분석 함수"""
    # --- 초기화 ---
    env_config = EnvConfig()
    env_cfg = env_config.base_env_cfg
    env_cfg['use_render'] = True
    
    env = SidewalkStaticMetaUrbanEnv(env_cfg)
    pygame.init()
    screen = pygame.display.set_mode((400, 150))
    pygame.display.set_caption("Control Agent with WASD")
    clock = pygame.time.Clock()
    
    # --- 시뮬레이션 시작 ---
    obs, info = env.reset()
    
    while len(env.agent.navigation.checkpoints) < 31:
        obs, info = env.reset()

    reward_history = []
    prev_info = {}
    
    print("\n--- Running Interactive Simulation ---")
    print("Controls: W (throttle), S (brake), A (left), D (right), Q (quit)")

    running = True
    for step in range(HORIZON):
        action = [0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in ACTION_MAP:
                action = ACTION_MAP[event.key]
        
        if not running:
            break

        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(action)
        info['original_reward'] = reward

        # 목표 지점 계산
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            info['distance_to_goal'] = np.linalg.norm(global_target - agent_pos)
        else:
            info['distance_to_goal'] = prev_info.get('distance_to_goal', 0)

        # 모든 리워드 계산 및 저장
        step_rewards = calculate_all_rewards(info, prev_info, action, env)
        reward_history.append(step_rewards)
        
        prev_info = info.copy()

        # 실시간 리워드 정보 출력
        reward_str = " | ".join([f"{k}: {v:.2f}" for k, v in step_rewards.items() if v != 0])
        print(f"Step {step+1}/{HORIZON} | {reward_str}")

        env.render(
            text={
                "Agent Position": env.agent.position,
                "Agent Heading": env.agent.heading_theta,
                "Reward": reward
            }
        )
        clock.tick(60)

        if terminated or truncated:
            print(f"Episode finished early at step {step+1}.")
            break

    env.close()
    pygame.quit()

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