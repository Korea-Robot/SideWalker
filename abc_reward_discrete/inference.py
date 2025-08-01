import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from typing import List
import argparse

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Import configurations and utilities
from env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, PDController

# Import models
from model import Actor, Critic


class InferenceAgent:
    """추론용 에이전트 클래스"""
    
    def __init__(self, actor_path: str, device: torch.device):
        """
        Args:
            actor_path: 학습된 actor 모델의 경로
            device: 실행할 디바이스
        """
        self.device = device
        
        # 모델 초기화 및 로드
        config = Config()
        self.actor = Actor(
            hidden_dim=config.hidden_dim,
            num_steering_actions=5,
            num_throttle_actions=3
        ).to(device)
        
        # 학습된 가중치 로드
        print(f"Loading actor model from: {actor_path}")
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.actor.eval()
        
        # PD 컨트롤러
        self.pd_controller = PDController(p_gain=0.5, d_gain=0.3)
        
        print("Inference agent initialized successfully!")
    
    def get_action(self, obs_data: dict) -> tuple:
        """관찰로부터 행동을 선택"""
        with torch.no_grad():
            # 관찰을 텐서로 변환
            rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
            
            # 디바이스로 이동
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            semantic = semantic.to(self.device)
            goal = goal.to(self.device)
            
            # 행동 샘플링
            (steering_idx, throttle_idx), (steering_val, throttle_val) = self.actor.sample_action(
                rgb, semantic, depth, goal
            )
            
            return (
                int(steering_idx.cpu().item()),
                int(throttle_idx.cpu().item())
            ), (
                float(steering_val.cpu().item()),
                float(throttle_val.cpu().item())
            )


class VideoRecorder:
    """RGB 비디오 녹화 클래스"""
    
    def __init__(self, output_path: str, fps: int = 30):
        """
        Args:
            output_path: 저장할 비디오 파일 경로
            fps: 프레임 레이트
        """
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self.writer = None
        
    def add_frame(self, rgb_frame: np.ndarray):
        """프레임 추가"""
        # RGB에서 BGR로 변환 (OpenCV 형식)
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr_frame)
    
    def save_video(self):
        """비디오 저장"""
        if not self.frames:
            print("No frames to save!")
            return
        
        # 비디오 작성기 초기화
        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
        # 프레임 작성
        for frame in self.frames:
            self.writer.write(frame)
        
        # 리소스 정리
        self.writer.release()
        print(f"Video saved to: {self.output_path}")
        print(f"Total frames: {len(self.frames)}")
        print(f"Duration: {len(self.frames) / self.fps:.2f} seconds")


def run_inference_episode(
    agent: InferenceAgent,
    env: SidewalkStaticMetaUrbanEnv,
    video_recorder: VideoRecorder,
    max_steps: int = 1000,
    verbose: bool = True
) -> dict:
    """한 에피소드 추론 실행"""
    
    # 환경 초기화
    obs, info = env.reset()
    waypoints = env.agent.navigation.checkpoints
    
    # 충분한 waypoints가 있는 환경 찾기
    while len(waypoints) < 31:
        obs, info = env.reset()
        waypoints = env.agent.navigation.checkpoints
        if verbose:
            print(f"Resetting environment... waypoints: {len(waypoints)}")
    
    if verbose:
        print(f"Starting episode with {len(waypoints)} waypoints")
        print(f"Goal position: {waypoints[-1]}")
    
    step_count = 0
    total_reward = 0
    episode_info = {
        'success': False,
        'crash': False,
        'out_of_road': False,
        'steps': 0,
        'final_distance_to_goal': 0,
        'checkpoints_passed': 0
    }
    
    while step_count < max_steps:
        # 목표 지점 계산 (look-ahead waypoint)
        ego_goal_position = np.array([0.0, 0.0])
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        
        k = 15  # look-ahead distance
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 관찰 데이터 준비
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        # RGB 프레임을 비디오에 추가
        video_recorder.add_frame(rgb_data)
        
        obs_data = {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic': semantic_data,
            'goal': ego_goal_position
        }
        
        # 행동 선택
        discrete_action, continuous_action = agent.get_action(obs_data)
        target_angle, throttle = continuous_action
        
        # PD 제어를 통해 최종 steering 값 계산
        final_steering = agent.pd_controller.get_control(target_angle, 0)
        final_action = (final_steering, throttle)
        
        # 환경 스텝
        obs, env_reward, terminated, truncated, info = env.step(final_action)
        
        total_reward += env_reward
        step_count += 1
        
        # 에피소드 정보 업데이트
        episode_info['steps'] = step_count
        episode_info['final_distance_to_goal'] = info.get('distance_to_goal', 0)
        episode_info['checkpoints_passed'] = info.get('closest_checkpoint_idx', 0)
        
        if info.get('arrive_dest', False):
            episode_info['success'] = True
            if verbose:
                print(f"SUCCESS! Reached destination in {step_count} steps")
            break
        
        if info.get('crash_vehicle', False) or info.get('crash_object', False):
            episode_info['crash'] = True
            if verbose:
                print(f"CRASH! Episode ended at step {step_count}")
            break
        
        if info.get('out_of_road', False):
            episode_info['out_of_road'] = True
            if verbose:
                print(f"OUT OF ROAD! Episode ended at step {step_count}")
            break
        
        if terminated or truncated:
            if verbose:
                print(f"Episode terminated/truncated at step {step_count}")
            break
        
        # 진행 상황 출력
        if verbose and step_count % 100 == 0:
            print(f"Step {step_count}: Distance to goal: {info.get('distance_to_goal', 0):.2f}, "
                  f"Speed: {info.get('speed', 0):.2f}, Checkpoint: {info.get('closest_checkpoint_idx', 0)}")
    
    episode_info['total_reward'] = total_reward
    
    if verbose:
        print(f"\nEpisode Summary:")
        print(f"  Success: {episode_info['success']}")
        print(f"  Crash: {episode_info['crash']}")
        print(f"  Out of road: {episode_info['out_of_road']}")
        print(f"  Steps: {episode_info['steps']}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final distance to goal: {episode_info['final_distance_to_goal']:.2f}")
        print(f"  Checkpoints passed: {episode_info['checkpoints_passed']}")
    
    return episode_info


def main():
    """메인 추론 함수"""
    parser = argparse.ArgumentParser(description='MetaUrban RL Inference with Video Recording')
    parser.add_argument('--actor_path', type=str, required=True,
                       help='Path to the trained actor model (.pt file)')
    parser.add_argument('--output_video', type=str, default='inference_episode.mp4',
                       help='Output video file path')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_video) if os.path.dirname(args.output_video) else '.', exist_ok=True)
    
    # 환경 설정
    env_config = EnvConfig()
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
    
    # 시드 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # 추론 에이전트 초기화
        agent = InferenceAgent(args.actor_path, device)
        
        # 비디오 레코더 초기화
        video_recorder = VideoRecorder(args.output_video, fps=args.fps)
        
        print(f"\nStarting inference episode...")
        print(f"Max steps: {args.max_steps}")
        print(f"Output video: {args.output_video}")
        
        # 추론 실행
        episode_info = run_inference_episode(
            agent=agent,
            env=env,
            video_recorder=video_recorder,
            max_steps=args.max_steps,
            verbose=True
        )
        
        # 비디오 저장
        print(f"\nSaving video...")
        video_recorder.save_video()
        
        print(f"\n{'='*50}")
        print(f"Inference completed successfully!")
        print(f"Video saved to: {args.output_video}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()


if __name__ == "__main__":
    main()