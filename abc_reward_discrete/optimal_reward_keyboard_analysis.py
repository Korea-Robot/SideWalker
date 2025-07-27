import torch
import numpy as np
import typing
import math
from collections import defaultdict
import pygame
import matplotlib.pyplot as plt

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
    Goal-position based navigation을 위한 최적화된 리워드 함수
    분석 결과를 바탕으로 효과적인 네비게이션 학습을 위해 설계됨
    """
    rewards = defaultdict(float)
    
    agent_pos = env.agent.position
    agent_heading = env.agent.heading_theta
    speed = info.get('speed', 0)
    nav = env.agent.navigation
    waypoints = nav.checkpoints
    rewards = defaultdict(float)
    
    # 1. 목표 근접 보상 (Goal Proximity Reward)
    if prev_info:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        if current_dist < prev_dist:
            rewards['goal_proximity'] = (prev_dist - current_dist) * 10.0


    # 2. 지능적 체크포인트 진행 보상 (Smart Checkpoint Progress)
    # 기존: 단순 통과, 개선: 거리 기반 가중치 + 연속성 보너스
    if prev_info and 'closest_checkpoint_idx' in prev_info:
        prev_closest = prev_info.get('closest_checkpoint_idx', 0)
        current_closest = info.get('closest_checkpoint_idx', 0)
        
        if current_closest > prev_closest:
            progress_steps = current_closest - prev_closest
            # 연속적으로 여러 체크포인트를 통과하면 보너스
            bonus_multiplier = 1.0 + (progress_steps - 1) * 0.3
            rewards['checkpoint_progress'] = 8.0 * progress_steps * bonus_multiplier
    
    # 3. 방향 정렬 보상 (Directional Alignment) - 핵심 네비게이션
    if len(waypoints) > 0:
        # 다음 몇 개의 체크포인트를 고려한 방향 계산
        look_ahead = min(3, len(waypoints) - 1)
        if look_ahead > 0:
            target_pos = waypoints[look_ahead]
            direction_to_target = np.arctan2(target_pos[1] - agent_pos[1], 
                                           target_pos[0] - agent_pos[0])
            
            # 헤딩과 목표 방향의 일치도
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

def calculate_optimal_navigation_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """
    Goal-position based navigation을 위한 최적화된 리워드 함수
    분석 결과를 바탕으로 효과적인 네비게이션 학습을 위해 설계됨
    """
    rewards = defaultdict(float)
    
    agent_pos = env.agent.position
    agent_heading = env.agent.heading_theta
    speed = info.get('speed', 0)
    nav = env.agent.navigation
    waypoints = nav.checkpoints
    
    # === 핵심 목표 지향 리워드 (Goal-Oriented Rewards) ===
    
    # 1. 향상된 목표 근접 보상 (Enhanced Goal Proximity)
    # 기존: 단순 거리 차이, 개선: 속도 고려 + 스케일링
    if prev_info and len(waypoints) > 15:
        prev_dist = prev_info.get('distance_to_goal', info.get('distance_to_goal', 0))
        current_dist = info.get('distance_to_goal', 0)
        
        if current_dist < prev_dist:
            progress = prev_dist - current_dist
            speed_multiplier = min(speed / 2.0, 1.5)  # 적정 속도일 때 보너스
            rewards['goal_proximity'] = progress * 15.0 * speed_multiplier
    
    # 2. 지능적 체크포인트 진행 보상 (Smart Checkpoint Progress)
    # 기존: 단순 통과, 개선: 거리 기반 가중치 + 연속성 보너스
    if prev_info and 'closest_checkpoint_idx' in prev_info:
        prev_closest = prev_info.get('closest_checkpoint_idx', 0)
        current_closest = info.get('closest_checkpoint_idx', 0)
        
        if current_closest > prev_closest:
            progress_steps = current_closest - prev_closest
            # 연속적으로 여러 체크포인트를 통과하면 보너스
            bonus_multiplier = 1.0 + (progress_steps - 1) * 0.3
            rewards['checkpoint_progress'] = 8.0 * progress_steps * bonus_multiplier
    
    # 3. 방향 정렬 보상 (Directional Alignment) - 핵심 네비게이션
    if len(waypoints) > 0:
        # 다음 몇 개의 체크포인트를 고려한 방향 계산
        look_ahead = min(3, len(waypoints) - 1)
        if look_ahead > 0:
            target_pos = waypoints[look_ahead]
            direction_to_target = np.arctan2(target_pos[1] - agent_pos[1], 
                                           target_pos[0] - agent_pos[0])
            
            # 헤딩과 목표 방향의 일치도
            heading_diff = abs(agent_heading - direction_to_target)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            
            alignment_score = 1.0 - (heading_diff / np.pi)
            rewards['direction_alignment'] = max(0, alignment_score) * 2.0
    
    # 4. 효율적인 이동 보상 (Efficient Movement)
    if prev_info and 'agent_position' in prev_info:
        prev_pos = prev_info['agent_position']
        movement_vector = agent_pos - prev_pos
        movement_distance = np.linalg.norm(movement_vector)
        
        if movement_distance > 0.01 and len(waypoints) > 0:
            # 목표 방향으로의 실제 진전 계산
            target_direction = waypoints[0] - agent_pos
            if np.linalg.norm(target_direction) > 0:
                target_direction = target_direction / np.linalg.norm(target_direction)
                movement_direction = movement_vector / movement_distance
                
                # 목표 방향과의 dot product
                efficiency = np.dot(movement_direction[:2], target_direction[:2])
                rewards['movement_efficiency'] = max(0, efficiency) * movement_distance * 3.0
    
    # === 속도 최적화 (Speed Optimization) ===
    
    # 5. 적응적 속도 보상 (Adaptive Speed Reward)
    # 기존 문제: speed_reward가 항상 0 → 속도 범위와 목표 재조정
    if speed > 0.05:  # 정지 상태가 아닐 때만
        # 거리에 따른 적정 속도 계산
        if len(waypoints) > 0:
            distance_to_next = np.linalg.norm(waypoints[0] - agent_pos)
            
            # 거리가 가까우면 감속, 멀면 가속
            if distance_to_next < 5.0:
                target_speed = 0.5 + (distance_to_next / 10.0)  # 0.5 ~ 1.0
            else:
                target_speed = 1.0 + min((distance_to_next - 5.0) / 20.0, 0.5)  # 1.0 ~ 1.5
            
            speed_diff = abs(speed - target_speed)
            rewards['adaptive_speed'] = max(0, 1.0 - speed_diff) * 1.5
    
    # === 안전성 및 제약 조건 (Safety & Constraints) ===
    
    # 6. 통합 페널티 시스템 (Unified Penalty System)
    # 치명적 페널티
    if info.get('crash_vehicle', False) or info.get('crash_object', False):
        rewards['collision_penalty'] = -20.0
    
    if info.get('out_of_road', False):
        rewards['out_of_road_penalty'] = -8.0

    
    # === 성공 보상 (Success Reward) ===
    
    # 8. 최종 목표 달성 (Ultimate Success)
    if info.get('arrive_dest', False):
        rewards['success_reward'] = 100.0  # 기존 50 → 100으로 증가
    
    # === 추가 정보 저장 ===
    info['prev_action'] = action
    
    return rewards

def plot_reward_timeline(reward_history, title="Reward Timeline"):
    """
    리워드 발생 시점을 시각화하는 함수
    """
    plt.figure(figsize=(15, 10))
    
    # 모든 리워드 타입 수집
    all_reward_types = set()
    for step_rewards in reward_history:
        all_reward_types.update(step_rewards.keys())
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_reward_types)))
    color_map = dict(zip(all_reward_types, colors))
    
    # 각 리워드 타입별로 발생 시점과 값을 플롯
    for reward_type in all_reward_types:
        steps = []
        values = []
        
        for step, step_rewards in enumerate(reward_history):
            if reward_type in step_rewards and step_rewards[reward_type] != 0:
                steps.append(step)
                values.append(step_rewards[reward_type])
        
        if steps:  # 데이터가 있는 경우만 플롯
            plt.scatter(steps, values, label=reward_type, 
                       color=color_map[reward_type], alpha=0.7, s=30)
    
    plt.xlabel('Step')
    plt.ylabel('Reward Value')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

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

    # 두 가지 리워드 시스템으로 기록
    basic_reward_history = []
    enhanced_reward_history = []
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

        # 추가 정보 수집
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            info['distance_to_goal'] = np.linalg.norm(global_target - agent_pos)
        else:
            info['distance_to_goal'] = prev_info.get('distance_to_goal', 0)
        
        # 체크포인트 관련 정보 추가
        agent_pos = env.agent.position
        if len(waypoints) > 0:
            distances = [np.linalg.norm(wp - agent_pos) for wp in waypoints]
            info['closest_checkpoint_idx'] = np.argmin(distances)
        
        info['agent_position'] = agent_pos
        info['agent_heading'] = env.agent.heading_theta

        # 기본 리워드와 최적화된 네비게이션 리워드 계산
        basic_rewards = calculate_all_rewards(info, prev_info, action, env)
        optimal_rewards = calculate_optimal_navigation_rewards(info, prev_info, action, env)
        
        basic_reward_history.append(basic_rewards)
        enhanced_reward_history.append(optimal_rewards)
        
        prev_info = info.copy()

        # 실시간 리워드 정보 출력 (최적화된 버전)
        reward_str = " | ".join([f"{k}: {v:.2f}" for k, v in optimal_rewards.items() if v != 0])
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

    # --- 결과 분석 및 시각화 ---
    print("\n--- Creating Reward Visualizations ---")
    
    # 1. 기본 리워드 시스템 플롯
    plt1 = plot_reward_timeline(basic_reward_history, "Basic Reward System Timeline")
    plt1.savefig('basic_reward_timeline.png', dpi=300, bbox_inches='tight')
    plt1.show()
    
    # 2. 최적화된 네비게이션 리워드 시스템 플롯
    plt2 = plot_reward_timeline(enhanced_reward_history, "Optimal Goal-Position Based Navigation Rewards")
    plt2.savefig('optimal_navigation_rewards.png', dpi=300, bbox_inches='tight')
    plt2.show()

    # --- 통계 분석 출력 ---
    print("\n--- Basic Reward Analysis ---")
    print_reward_statistics(basic_reward_history)
    
    print("\n--- Optimal Navigation Reward Analysis ---")
    print_reward_statistics(enhanced_reward_history)

def print_reward_statistics(reward_history):
    """리워드 통계를 출력하는 함수"""
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


if __name__ == "__main__":
    main()

    
# 실제 실험 결과 
    """
    --- Creating Reward Visualizations ---

--- Basic Reward Analysis ---
Reward Component          |     Total Value |       Avg Value |    Triggered Steps
--------------------------------------------------------------------------------
collision_penalty         |         -20.000 |         -10.000 |          2 / 315
env_default_reward        |          67.595 |           0.322 |        210 / 315
goal_proximity            |         303.517 |           2.734 |        111 / 315
speed_reward              |           0.000 |           0.000 |          0 / 315
success_reward            |          50.000 |          50.000 |          1 / 315

--- Optimal Navigation Reward Analysis ---
Reward Component          |     Total Value |       Avg Value |    Triggered Steps
--------------------------------------------------------------------------------
checkpoint_progress       |         464.000 |           8.000 |         58 / 315
collision_penalty         |         -40.000 |         -20.000 |          2 / 315
direction_alignment       |         161.597 |           0.513 |        315 / 315
goal_proximity            |           0.000 |           0.000 |          0 / 315
movement_efficiency       |           1.832 |           0.108 |         17 / 315
success_reward            |         100.000 |         100.000 |          1 / 315
    """