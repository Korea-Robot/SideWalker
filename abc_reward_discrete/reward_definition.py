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