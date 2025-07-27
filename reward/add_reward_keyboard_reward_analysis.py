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

def calculate_enhanced_rewards(info: dict, prev_info: dict, action: tuple, env) -> dict:
    """
    네비게이션 관련 리워드가 추가된 향상된 리워드 계산 함수
    """
    rewards = defaultdict(float)
    
    # 기존 리워드들 계산
    rewards.update(calculate_all_rewards(info, prev_info, action, env))
    
    # === 추가된 네비게이션 리워드들 ===
    
    # 8. 경로 추적 보상 (Path Following Reward)
    nav = env.agent.navigation
    waypoints = nav.checkpoints
    if len(waypoints) > 1:
        # 현재 위치에서 다음 체크포인트까지의 방향과 실제 이동 방향 비교
        agent_pos = env.agent.position
        agent_heading = env.agent.heading_theta
        
        # 가장 가까운 체크포인트 찾기
        distances = [np.linalg.norm(wp - agent_pos) for wp in waypoints[:5]]
        closest_wp_idx = np.argmin(distances)
        
        if closest_wp_idx < len(waypoints) - 1:
            target_wp = waypoints[closest_wp_idx + 1]
            direction_to_target = np.arctan2(target_wp[1] - agent_pos[1], target_wp[0] - agent_pos[0])
            heading_diff = abs(agent_heading - direction_to_target)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)  # 최소 각도 차이
            
            # 방향이 맞을수록 높은 보상
            rewards['path_following'] = max(0, 1.0 - heading_diff / np.pi) * 0.5
    
    # 9. 체크포인트 통과 보상 (Checkpoint Passing Reward)
    if prev_info and 'closest_checkpoint_idx' in prev_info:
        prev_closest = prev_info.get('closest_checkpoint_idx', 0)
        current_closest = info.get('closest_checkpoint_idx', 0)
        if current_closest > prev_closest:
            rewards['checkpoint_passed'] = 5.0 * (current_closest - prev_closest)
    
    # 10. 진행 방향 일관성 보상 (Forward Progress Reward)
    if prev_info and 'agent_position' in prev_info:
        prev_pos = prev_info['agent_position']
        current_pos = env.agent.position
        movement_vector = current_pos - prev_pos
        movement_distance = np.linalg.norm(movement_vector)
        
        if movement_distance > 0.01:  # 최소 이동 거리
            # 목표 방향과의 일치도 계산
            nav = env.agent.navigation
            if len(nav.checkpoints) > 1:
                target_direction = nav.checkpoints[1] - current_pos
                target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)
                movement_direction = movement_vector / movement_distance
                
                dot_product = np.dot(movement_direction[:2], target_direction[:2])
                rewards['forward_progress'] = max(0, dot_product) * movement_distance * 2.0
    
    # 11. 차선 중앙 유지 보상 (Lane Center Reward)
    lane_info = info.get('lane_info', {})
    lateral_distance = lane_info.get('lateral_distance', 0)  # 차선 중앙으로부터의 거리
    if lateral_distance is not None:
        # 차선 중앙에 가까울수록 높은 보상
        max_lane_width = 3.5  # 일반적인 차선 폭
        normalized_distance = min(abs(lateral_distance) / max_lane_width, 1.0)
        rewards['lane_center'] = (1.0 - normalized_distance) * 0.3
    
    # 12. 급격한 방향 변경 페널티 (Sharp Turn Penalty)
    if prev_info and 'agent_heading' in prev_info:
        prev_heading = prev_info['agent_heading']
        current_heading = env.agent.heading_theta
        
        heading_change = abs(current_heading - prev_heading)
        heading_change = min(heading_change, 2 * np.pi - heading_change)
        
        if heading_change > 0.1:  # 임계값 이상의 급격한 방향 변경
            rewards['sharp_turn_penalty'] = -heading_change * 2.0
    
    # 13. 목표 지향성 보상 (Goal Orientation Reward)
    if len(waypoints) > 10:
        long_term_target = waypoints[10]
        agent_pos = env.agent.position
        agent_heading = env.agent.heading_theta
        
        direction_to_long_term_goal = np.arctan2(
            long_term_target[1] - agent_pos[1], 
            long_term_target[0] - agent_pos[0]
        )
        
        heading_alignment = np.cos(agent_heading - direction_to_long_term_goal)
        rewards['goal_orientation'] = max(0, heading_alignment) * 0.2
    
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

        # 기본 리워드와 향상된 리워드 계산
        basic_rewards = calculate_all_rewards(info, prev_info, action, env)
        enhanced_rewards = calculate_enhanced_rewards(info, prev_info, action, env)
        
        basic_reward_history.append(basic_rewards)
        enhanced_reward_history.append(enhanced_rewards)
        
        prev_info = info.copy()

        # 실시간 리워드 정보 출력 (향상된 버전)
        reward_str = " | ".join([f"{k}: {v:.2f}" for k, v in enhanced_rewards.items() if v != 0])
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
    
    # 2. 향상된 리워드 시스템 플롯
    plt2 = plot_reward_timeline(enhanced_reward_history, "Enhanced Reward System with Navigation")
    plt2.savefig('enhanced_reward_timeline.png', dpi=300, bbox_inches='tight')
    plt2.show()

    # --- 통계 분석 출력 ---
    print("\n--- Basic Reward Analysis ---")
    print_reward_statistics(basic_reward_history)
    
    print("\n--- Enhanced Reward Analysis ---")
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