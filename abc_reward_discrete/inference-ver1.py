import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional
import time

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Import model and utilities
from model import Actor, Critic
from env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, PDController


class MetaUrbanInference:
    """학습된 모델을 사용한 MetaUrban 추론 클래스"""
    
    def __init__(self, 
                 actor_checkpoint_path: str,
                 critic_checkpoint_path: Optional[str] = None,
                 device: str = "cuda:0"):
        """
        Args:
            actor_checkpoint_path: Actor 모델 체크포인트 경로
            critic_checkpoint_path: Critic 모델 체크포인트 경로 (선택사항)
            device: 사용할 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Config 로드
        self.config = Config()
        
        # 모델 초기화
        self.actor = Actor(
            hidden_dim=self.config.hidden_dim,
            num_steering_actions=5,
            num_throttle_actions=3
        ).to(self.device)
        
        if critic_checkpoint_path:
            self.critic = Critic(hidden_dim=self.config.hidden_dim).to(self.device)
        else:
            self.critic = None
        
        # 체크포인트 로드
        self._load_checkpoints(actor_checkpoint_path, critic_checkpoint_path)
        
        # PD 컨트롤러
        self.pd_controller = PDController(p_gain=0.5, d_gain=0.3)
        
        # 환경 설정
        env_config = EnvConfig()
        self.env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
        
        # 추론 모드로 설정
        self.actor.eval()
        if self.critic:
            self.critic.eval()
            
    def _load_checkpoints(self, actor_path: str, critic_path: Optional[str] = None):
        """체크포인트 로드"""
        from checkpoint_utils import load_actor_checkpoint, load_critic_checkpoint
        
        # Actor 로드
        success = load_actor_checkpoint(self.actor, actor_path, str(self.device))
        if not success:
            raise RuntimeError(f"Failed to load actor checkpoint from {actor_path}")
        
        # Critic 로드 (선택사항)
        if critic_path and self.critic:
            success = load_critic_checkpoint(self.critic, critic_path, str(self.device))
            if not success:
                print(f"Warning: Failed to load critic checkpoint from {critic_path}")
            
    def _prepare_observation(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """관찰 데이터를 모델 입력 형태로 변환"""
        # 센서 데이터 추출
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        # 목표 지점 계산 (ego-centric)
        ego_goal_position = np.array([0.0, 0.0])
        nav = self.env.agent.navigation
        waypoints = nav.checkpoints
        
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = self.env.agent.position
            agent_heading = self.env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 텐서로 변환
        rgb_tensor = torch.tensor(rgb_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth_tensor = torch.tensor(depth_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic_tensor = torch.tensor(semantic_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal_tensor = torch.tensor(ego_goal_position, dtype=torch.float32).unsqueeze(0)
        
        return {
            'rgb': rgb_tensor.to(self.device),
            'depth': depth_tensor.to(self.device),
            'semantic': semantic_tensor.to(self.device),
            'goal': goal_tensor.to(self.device)
        }
    
    def predict_action(self, obs: Dict, deterministic: bool = False) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        주어진 관찰에 대해 행동 예측
        
        Args:
            obs: 환경 관찰 데이터
            deterministic: True면 최대 확률 행동 선택, False면 샘플링
            
        Returns:
            (discrete_action, continuous_action): discrete indices와 continuous values
        """
        obs_tensors = self._prepare_observation(obs)
        
        with torch.no_grad():
            steering_dist, throttle_dist = self.actor(
                obs_tensors['rgb'],
                obs_tensors['semantic'],
                obs_tensors['depth'],
                obs_tensors['goal']
            )
            
            if deterministic:
                # 최대 확률 행동 선택
                steering_idx = torch.argmax(steering_dist.probs, dim=-1)
                throttle_idx = torch.argmax(throttle_dist.probs, dim=-1)
            else:
                # 확률 분포에서 샘플링
                steering_idx = steering_dist.sample()
                throttle_idx = throttle_dist.sample()
            
            # continuous values 계산
            steering_val, throttle_val = self.actor.get_action_values(
                steering_idx, throttle_idx, self.device
            )
            
        discrete_action = (int(steering_idx.cpu().item()), int(throttle_idx.cpu().item()))
        continuous_action = (float(steering_val.cpu().item()), float(throttle_val.cpu().item()))
        
        return discrete_action, continuous_action
    
    def predict_value(self, obs: Dict) -> Optional[float]:
        """상태 가치 예측 (Critic이 로드된 경우만)"""
        if not self.critic:
            return None
            
        obs_tensors = self._prepare_observation(obs)
        
        with torch.no_grad():
            value = self.critic(
                obs_tensors['rgb'],
                obs_tensors['semantic'],
                obs_tensors['depth'],
                obs_tensors['goal']
            )
            
        return float(value.cpu().item())
    
    def run_episode(self, 
                   max_steps: int = 512,
                   deterministic: bool = False,
                   save_video: bool = True,
                   video_path: str = "inference_video.mp4") -> Dict:
        """
        한 에피소드 실행
        
        Args:
            max_steps: 최대 스텝 수
            deterministic: 결정적 행동 선택 여부
            save_video: 비디오 저장 여부
            video_path: 비디오 저장 경로
            
        Returns:
            에피소드 결과 딕셔너리
        """
        obs, info = self.env.reset()
        
        # 충분한 waypoints가 있는지 확인
        waypoints = self.env.agent.navigation.checkpoints
        while len(waypoints) < 31:
            obs, info = self.env.reset()
            waypoints = self.env.agent.navigation.checkpoints
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'total_reward': 0,
            'steps': 0,
            'success': False,
            'crash': False
        }
        
        # 비디오 저장용
        if save_video:
            frames = []
        
        for step in range(max_steps):
            # 행동 예측
            discrete_action, continuous_action = self.predict_action(obs, deterministic)
            
            # 가치 예측 (가능한 경우)
            value = self.predict_value(obs)
            
            # PD 컨트롤을 통한 최종 steering 계산
            target_angle, throttle = continuous_action
            final_steering = self.pd_controller.get_control(target_angle, 0)
            final_action = (final_steering, throttle)
            
            # 환경 스텝
            next_obs, reward, terminated, truncated, next_info = self.env.step(final_action)
            
            # 데이터 저장
            episode_data['observations'].append(obs)
            episode_data['actions'].append(discrete_action)
            episode_data['rewards'].append(reward)
            if value is not None:
                episode_data['values'].append(value)
            episode_data['total_reward'] += reward
            episode_data['steps'] += 1
            
            # 비디오 프레임 저장
            if save_video:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            
            # 종료 조건 확인
            if next_info.get('arrive_dest', False):
                episode_data['success'] = True
                print("✅ Successfully reached destination!")
                break
                
            if next_info.get('crash_vehicle', False) or next_info.get('crash_object', False):
                episode_data['crash'] = True
                print("❌ Crashed!")
                break
                
            if terminated or truncated:
                break
                
            obs = next_obs
            info = next_info
        
        # 비디오 저장
        if save_video and frames:
            self._save_video(frames, video_path)
            print(f"Video saved to: {video_path}")
        
        return episode_data
    
    def _save_video(self, frames: list, video_path: str):
        """프레임들을 비디오로 저장"""
        if not frames:
            return
            
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def evaluate(self, 
                num_episodes: int = 10,
                deterministic: bool = True,
                max_steps: int = 512) -> Dict:
        """
        여러 에피소드에 걸쳐 모델 평가
        
        Args:
            num_episodes: 평가할 에피소드 수
            deterministic: 결정적 행동 선택 여부
            max_steps: 에피소드당 최대 스텝 수
            
        Returns:
            평가 결과 딕셔너리
        """
        print(f"Evaluating model over {num_episodes} episodes...")
        
        results = {
            'episode_rewards': [],
            'episode_steps': [],
            'success_rate': 0,
            'crash_rate': 0,
            'avg_reward': 0,
            'avg_steps': 0
        }
        
        success_count = 0
        crash_count = 0
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            episode_data = self.run_episode(
                max_steps=max_steps,
                deterministic=deterministic
            )
            
            results['episode_rewards'].append(episode_data['total_reward'])
            results['episode_steps'].append(episode_data['steps'])
            
            if episode_data['success']:
                success_count += 1
            if episode_data['crash']:
                crash_count += 1
                
            print(f"  Reward: {episode_data['total_reward']:.2f}, "
                  f"Steps: {episode_data['steps']}, "
                  f"Success: {episode_data['success']}, "
                  f"Crash: {episode_data['crash']}")
        
        # 통계 계산
        results['success_rate'] = success_count / num_episodes
        results['crash_rate'] = crash_count / num_episodes
        results['avg_reward'] = np.mean(results['episode_rewards'])
        results['avg_steps'] = np.mean(results['episode_steps'])
        
        print(f"\n=== Evaluation Results ===")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Crash Rate: {results['crash_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        
        return results
    
    def close(self):
        """환경 종료"""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    """메인 추론 함수"""
    # 체크포인트 경로 설정
    actor_checkpoint = "checkpoints/metaurban_discrete_actor_epoch_90.pt"  # 실제 경로로 변경
    critic_checkpoint = "checkpoints/metaurban_discrete_critic_epoch_90.pt"  # 실제 경로로 변경
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(actor_checkpoint):
        print(f"Error: Actor checkpoint not found at {actor_checkpoint}")
        return
    
    if not os.path.exists(critic_checkpoint):
        print(f"Warning: Critic checkpoint not found at {critic_checkpoint}")
        critic_checkpoint = None
    
    try:
        device = 'cpu'
        
        # 추론 객체 생성
        inference = MetaUrbanInference(
            actor_checkpoint_path=actor_checkpoint,
            critic_checkpoint_path=critic_checkpoint,
            device=device  # 또는 "cpu"
        )
        
        print("Model loaded successfully!")
        
        # 단일 에피소드 실행 (비디오 저장)
        print("\n=== Running single episode with video recording ===")
        episode_data = inference.run_episode(
            max_steps=512,
            deterministic=True,
            save_video=True,
            video_path="inference_demo.mp4"
        )
        
        print(f"Episode completed!")
        print(f"Total Reward: {episode_data['total_reward']:.2f}")
        print(f"Steps: {episode_data['steps']}")
        print(f"Success: {episode_data['success']}")
        print(f"Crash: {episode_data['crash']}")
        
        # 다중 에피소드 평가
        print("\n=== Running evaluation ===")
        eval_results = inference.evaluate(
            num_episodes=5,
            deterministic=True,
            max_steps=512
        )
        
        # 환경 종료
        inference.close()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()