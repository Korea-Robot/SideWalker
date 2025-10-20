# path_following_datagen.py
import numpy as np
import os
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math
import random 

# Import configurations and utilities
from env_config import EnvConfig
from utils import PD_Controller,convert_to_egocentric

## PD controller 생성
pd_controller = PD_Controller(kp=0.2,kd=0.0) # 제일 안정적임을 확인 

# 개선된 데이터셋 수집기 import
from improved_walle_dataset import ImitationDatasetCollector
# 데이터셋 수집기 초기화
collector = ImitationDatasetCollector("imitation_dataset")

# --- 메인 실행 로직 ---
env_config = EnvConfig()
env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)

running = True 
try:
    # 여러 에피소드 실행 : 각 에피소드 마다 데이터 저장 
    episode = 1
    for i in range(100000):
        episode+=1
        # 일정 waypoints 이상 없으면 다시 환경 구성 .
        obs,info = env.reset(seed=episode)
        
        waypoints = env.agent.navigation.checkpoints 
        print('waypoint num: ',len(waypoints))
        
        # 웨이포인트가 충분히 있는지 확인
        while len(waypoints)<30:
            episode+=1
            obs,info = env.reset(seed= episode)
            waypoints = env.agent.navigation.checkpoints 
            print('NOT sufficient waypoints ',episode,' !!!!',len(waypoints))
            
        num_waypoints = len(waypoints)
        
        # *** 새로운 에피소드 시작 - 이 부분이 누락되어 있었습니다! ***
        collector.start_new_episode(waypoints)
        
        # 5번째 웨이포인트를 목표로 설정
        k = 5
        step = 0 
        reward = 0
        # 만약에 끼어서 계속 가만히 있는경우 제거하기 위해서.
        start_position = env.agent.position
        stuck_interval = 10
        
        # 에피소드 루프
        while running:

            # --- 목표 지점 계산 (Egocentric) ---
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            
            # k 번째 waypoint의 ego coordinate 기준 좌표 
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

            action = pd_controller.update(ego_goal_position[1]) # yaw방향에 대해서만 추측함. throttle은 고정 
            
            # ----------- 목표 웨이포인트 업데이트 ---------------- 
            # 목표지점까지 직선거리 계산 
            distance_to_target = np.linalg.norm(ego_goal_position)
            
            # 목표 지점 업데이트
            if distance_to_target< 5.0:
                k +=1
                if k>= num_waypoints:
                    k = num_waypoints-1

            # 에이전트 상태 정보 수집
            agent_state = {
                "position": env.agent.position,
                "heading": env.agent.heading_theta,
                "velocity": env.agent.speed,
                "angular_velocity": getattr(env.agent, 'angular_velocity', 0.0)
            }
            
            # 데이터 수집
            collector.collect_sample(
                obs, action, agent_state, ego_goal_position, reward, step
            )                

            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)

            
            # rgb = obs['image']
            # depth = obs['depth']
            # semantic = obs['semantic']
            # breakpoint()
            
            
            # 환경 렌더링 및 정보 표시
            # env.render(
            #     text={
            #         "Agent Position": np.round(env.agent.position, 2),
            #         "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
            #         "Reward": f"{reward:.2f}",
            #         "Ego Goal Position": np.round(ego_goal_position, 2)
            #     }
            # )
            
            
            step+=1 
            

            # 에피소드 종료 조건
            if terminated or truncated or step >= 800 or reward <0:
                episode_info = {
                    "seed": episode,
                    "terminated": terminated,
                    "truncated": truncated,
                    "crashed": reward < 0,
                    "episode_length": step,
                    "success": bool(np.linalg.norm(waypoints-env.agent.position) < 1)
                }


                # 에이전트 상태 정보 수집
                agent_state = {
                    "position": env.agent.position,
                    "heading": env.agent.heading_theta,
                    "velocity": env.agent.speed,
                }
                
                # 데이터 수집
                collector.collect_sample(
                    obs, action, agent_state, ego_goal_position, reward, step
                )  
                
                break
            
        
        print(f"Episode completed with {step} steps")
        
        
        # 에피소드 하나 종료
        # try:
        collector.finish_episode(episode_info)
        
        # 각 에피소드의 길이가 64 이상이면 주기적으로 저장
        if step > 64: 
            # 주기적으로 데이터셋 정보 저장 (매 10 에피소드마다)
            if collector.episode_counter % 10 == 0:
                collector.save_dataset_info()
                collector.create_train_test_split()
                
                print(f"\nDataset update - Episode {collector.episode_counter}:")
                print(f"  Total episodes: {collector.episode_counter}")
                print(f"  Total samples: {collector.dataset_info['total_samples']}")
                print(f"  Dataset saved to: {collector.dataset_root}")
        else:
            print(f"Episode {episode} too short ({step} steps), skipping...")
            
        # except Exception as e:
        #     print(f"Error saving episode {episode}: {e}")
        #     continue
        
        
            
        print(f"Episode {episode}: completed with {step} steps")

finally:
    # 종료 시 리소스 정리
    env.close()