# 필요한 라이브러리들을 가져옵니다.
import numpy as np
import os
import pygame
import math
import json
import time
from datetime import datetime
from PIL import Image
import cv2

# MetaUrban 환경 및 관련 구성 요소들을 가져옵니다.
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- PD 제어기 클래스 ---
class PDController:
    """PD (Proportional-Derivative) 제어기를 구현한 클래스"""
    
    def __init__(self, kp_steer=1.5, kd_steer=0.3, kp_speed=0.8, kd_speed=0.1):
        """
        PD 제어기 초기화
        
        :param kp_steer: 조향각에 대한 비례 게인
        :param kd_steer: 조향각에 대한 미분 게인
        :param kp_speed: 속도에 대한 비례 게인
        :param kd_speed: 속도에 대한 미분 게인
        """
        # 조향 제어 파라미터
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        
        # 속도 제어 파라미터
        self.kp_speed = kp_speed
        self.kd_speed = kd_speed
        
        # 이전 오차값 저장 (미분항 계산용)
        self.prev_lateral_error = 0.0
        self.prev_speed_error = 0.0
        
        # 목표 속도
        self.target_speed = 0.6  # m/s
        
    def update(self, ego_goal_position, current_speed, dt=1/60):
        """
        PD 제어기를 업데이트하여 조향각과 스로틀을 계산
        
        :param ego_goal_position: 에이전트 기준 목표 위치 [x, y]
        :param current_speed: 현재 속도 (m/s)
        :param dt: 시간 간격 (초)
        :return: [steer, throttle] 액션
        """
        # 1. 조향 제어 (횡방향 오차 기반)
        lateral_error = ego_goal_position[0]  # x축 오차 (좌우)
        lateral_error_rate = (lateral_error - self.prev_lateral_error) / dt
        
        # PD 제어로 조향각 계산
        steer = -(self.kp_steer * lateral_error + self.kd_steer * lateral_error_rate)
        steer = np.clip(steer, -1.0, 1.0)  # 조향각 제한
        
        # 2. 속도 제어 (종방향 제어)
        speed_error = self.target_speed - current_speed
        speed_error_rate = (speed_error - self.prev_speed_error) / dt
        
        # PD 제어로 스로틀 계산
        throttle = self.kp_speed * speed_error + self.kd_speed * speed_error_rate
        throttle = np.clip(throttle, -1.0, 1.0)  # 스로틀 제한
        
        # 급정거 방지: 목표 지점이 너무 가까우면 속도 감소
        distance_to_goal = np.linalg.norm(ego_goal_position)
        if distance_to_goal < 3.0:
            throttle *= 0.5
        
        # 이전 오차값 업데이트
        self.prev_lateral_error = lateral_error
        self.prev_speed_error = speed_error
        
        return [steer, throttle]

# --- 데이터셋 수집기 클래스 ---
class ImitationDatasetCollector:
    """모방학습용 데이터셋을 수집하는 클래스"""
    
    def __init__(self, dataset_root="imitation_dataset"):
        self.dataset_root = dataset_root
        self.create_directory_structure()
        
        # 현재 에피소드 데이터 저장
        self.current_episode_data = []
        self.episode_counter = 0
        
        # 메타데이터
        self.dataset_info = {
            "created_at": datetime.now().isoformat(),
            "total_episodes": 0,
            "total_samples": 0,
            "image_size": None,
            "action_space": {"steer": [-1.0, 1.0], "throttle": [-1.0, 1.0]},
            "sensors": ["rgb_camera", "depth_camera", "semantic_camera"]
        }
        
    def create_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        directories = [
            self.dataset_root,
            os.path.join(self.dataset_root, "images"),
            os.path.join(self.dataset_root, "depth"),
            os.path.join(self.dataset_root, "semantic"),
            os.path.join(self.dataset_root, "episodes")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def start_new_episode(self):
        """새로운 에피소드 시작"""
        self.current_episode_data = []
        self.episode_counter += 1
        
    def collect_sample(self, obs, action, agent_state, goal_position, reward, step):
        """한 스텝의 데이터를 수집"""
        timestamp = step
        
        # 이미지 데이터 저장
        rgb_filename = f"ep{self.episode_counter:04d}_step{step:04d}_rgb.png"
        depth_filename = f"ep{self.episode_counter:04d}_step{step:04d}_depth.png"
        semantic_filename = f"ep{self.episode_counter:04d}_step{step:04d}_semantic.png"
        
        # RGB 이미지 저장
        if 'rgb_camera' in obs:
            rgb_image = (obs['rgb_camera'] * 255).astype(np.uint8)
            Image.fromarray(rgb_image).save(
                os.path.join(self.dataset_root, "images", rgb_filename)
            )
        
        # Depth 이미지 저장
        if 'depth_camera' in obs:
            depth_image = obs['depth_camera']
            # Depth를 0-255 범위로 정규화
            depth_normalized = ((depth_image - depth_image.min()) / 
                              (depth_image.max() - depth_image.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized, mode='L').save(
                os.path.join(self.dataset_root, "depth", depth_filename)
            )
        
        # Semantic 이미지 저장
        if 'semantic_camera' in obs:
            semantic_image = (obs['semantic_camera'] * 255).astype(np.uint8)
            Image.fromarray(semantic_image).save(
                os.path.join(self.dataset_root, "semantic", semantic_filename)
            )
        
        # 샘플 데이터 구성
        sample_data = {
            "timestamp": timestamp,
            "step": step,
            "images": {
                "rgb": rgb_filename,
                "depth": depth_filename,
                "semantic": semantic_filename
            },
            "action": {
                "steer": float(action[0]),
                "throttle": float(action[1])
            },
            "agent_state": {
                "position": agent_state["position"].tolist(),
                "heading": float(agent_state["heading"]),
                "velocity": float(agent_state["velocity"]),
                "angular_velocity": float(agent_state.get("angular_velocity", 0.0))
            },
            "goal_position": {
                "ego_x": float(goal_position[0]),
                "ego_y": float(goal_position[1])
            },
            "reward": float(reward)
        }
        
        self.current_episode_data.append(sample_data)
        
        # 이미지 크기 정보 저장 (첫 번째 샘플에서만)
        if self.dataset_info["image_size"] is None and 'rgb_camera' in obs:
            self.dataset_info["image_size"] = list(obs['rgb_camera'].shape[:2])
    
    def finish_episode(self, episode_info):
        """에피소드 종료 및 데이터 저장"""
        if not self.current_episode_data:
            return
        
        # 에피소드 데이터 구성
        episode_data = {
            "episode_id": self.episode_counter,
            "episode_info": episode_info,
            "samples": self.current_episode_data,
            "total_steps": len(self.current_episode_data)
        }
        
        # JSON 파일로 저장
        episode_filename = f"episode_{self.episode_counter:04d}.json"
        with open(os.path.join(self.dataset_root, "episodes", episode_filename), 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # 메타데이터 업데이트
        self.dataset_info["total_episodes"] = self.episode_counter
        self.dataset_info["total_samples"] += len(self.current_episode_data)
        
        print(f"Episode {self.episode_counter} saved with {len(self.current_episode_data)} samples")
    
    def save_dataset_info(self):
        """데이터셋 메타정보 저장"""
        with open(os.path.join(self.dataset_root, "dataset_info.json"), 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def create_train_test_split(self, test_ratio=0.2, val_ratio=0.1):
        """학습/검증/테스트 데이터 분할"""
        total_episodes = self.episode_counter
        episodes = list(range(1, total_episodes + 1))
        
        # 랜덤 셔플
        np.random.shuffle(episodes)
        
        # 분할 계산
        test_size = int(total_episodes * test_ratio)
        val_size = int(total_episodes * val_ratio)
        train_size = total_episodes - test_size - val_size
        
        # 분할
        train_episodes = episodes[:train_size]
        val_episodes = episodes[train_size:train_size + val_size]
        test_episodes = episodes[train_size + val_size:]
        
        split_info = {
            "train": sorted(train_episodes),
            "validation": sorted(val_episodes),
            "test": sorted(test_episodes),
            "split_info": {
                "total_episodes": total_episodes,
                "train_episodes": len(train_episodes),
                "val_episodes": len(val_episodes),
                "test_episodes": len(test_episodes)
            }
        }
        
        with open(os.path.join(self.dataset_root, "train_test_split.json"), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Dataset split created: Train({len(train_episodes)}), Val({len(val_episodes)}), Test({len(test_episodes)})")

# --- 유틸리티 함수 ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계를 에이전트 중심 좌표계로 변환"""
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

# --- 설정 ---
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True,
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

# --- 메인 실행 로직 ---
def main():
    # 환경 초기화
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    
    # PD 제어기 초기화
    controller = PDController(kp_steer=2.0, kd_steer=0.5, kp_speed=1.0, kd_speed=0.2)
    
    # 데이터셋 수집기 초기화
    collector = ImitationDatasetCollector("imitation_dataset")
    
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((400, 150))
    pygame.display.set_caption("Autonomous Path Following - Dataset Collection")
    clock = pygame.time.Clock()
    
    running = True
    manual_override = False  # 수동 조작 모드
    
    # 키보드 입력 매핑 (수동 조작용)
    ACTION_MAP = {
        pygame.K_w: [0, 1.0],
        pygame.K_s: [0, -1.0],
        pygame.K_a: [0.5, 0.5],
        pygame.K_d: [-0.5, 0.5]
    }
    
    try:
        # 여러 에피소드 실행
        for episode in range(20):  # 20개 에피소드 수집
            print(f"\n=== Episode {episode + 1} started ===")
            
            # 환경 리셋
            obs = env.reset(seed=episode + 42)
            
            # 새 에피소드 시작
            collector.start_new_episode()
            controller = PDController()  # 제어기 리셋
            
            step = 0
            episode_reward = 0
            start_time = time.time()
            
            while running:
                step += 1
                
                # 이벤트 처리
                manual_action = None
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:  # 스페이스바로 수동/자동 모드 전환
                            manual_override = not manual_override
                            print(f"Manual override: {manual_override}")
                        elif event.key in ACTION_MAP and manual_override:
                            manual_action = ACTION_MAP[event.key]
                
                if not running:
                    break
                
                # 목표 지점 계산
                ego_goal_position = np.array([0.0, 0.0])
                nav = env.agent.navigation
                waypoints = nav.checkpoints
                
                if len(waypoints) > 15:
                    k = min(15, len(waypoints) - 1)
                    global_target = waypoints[k]
                    agent_pos = env.agent.position
                    agent_heading = env.agent.heading_theta
                    ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
                
                # 액션 결정
                if manual_override and manual_action is not None:
                    action = manual_action
                else:
                    # PD 제어기를 사용한 자동 제어
                    current_speed = env.agent.speed
                    action = controller.update(ego_goal_position, current_speed)
                
                # 환경 스텝 실행
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
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
                
                # 렌더링
                control_mode = "MANUAL" if manual_override else "AUTO"
                env.render(
                    text={
                        "Episode": f"{episode + 1}/20",
                        "Step": step,
                        "Control Mode": control_mode,
                        "Agent Position": np.round(env.agent.position, 2),
                        "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                        "Speed": f"{env.agent.speed:.2f} m/s",
                        "Reward": f"{reward:.2f}",
                        "Total Reward": f"{episode_reward:.2f}",
                        "Ego Goal": np.round(ego_goal_position, 2),
                        "Action": f"[{action[0]:.2f}, {action[1]:.2f}]"
                    }
                )
                
                clock.tick(60)
                
                # 에피소드 종료 조건
                if terminated or truncated or step >= 800:
                    episode_time = time.time() - start_time
                    episode_info = {
                        "seed": episode + 42,
                        "terminated": terminated,
                        "truncated": truncated,
                        "total_reward": episode_reward,
                        "episode_length": step,
                        "episode_time": episode_time,
                        "success": reward > 0.5  # 성공 기준
                    }
                    
                    collector.finish_episode(episode_info)
                    print(f"Episode {episode + 1} finished - Steps: {step}, Reward: {episode_reward:.2f}")
                    break
            
            if not running:
                break
    
    finally:
        # 데이터셋 정보 저장
        collector.save_dataset_info()
        collector.create_train_test_split()
        
        print(f"\nDataset collection completed!")
        print(f"Total episodes: {collector.episode_counter}")
        print(f"Total samples: {collector.dataset_info['total_samples']}")
        print(f"Dataset saved to: {collector.dataset_root}")
        
        # 환경 및 Pygame 정리
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()