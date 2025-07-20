import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
import os

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 환경 설정 ---
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=False,
    map='X',
    manual_control=False,
    crswalk_density=1,
    object_density=0.1,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=1000,
    
    vehicle_config=dict(enable_reverse=True, image_source="rgb_camera"),
    
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
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    stack_size=1,  # 스택 사이즈 추가
    log_level=50,
)

def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계의 목표 지점을 에이전트 중심의 자기 좌표계로 변환"""
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

class Actor(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=2):
        super().__init__()
        
        # RGB 처리 (3채널)
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.rgb_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.rgb_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Depth 처리 (1채널)
        self.depth_conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.depth_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.depth_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 특징 융합
        self.fc1 = nn.Linear(64 * 28 * 16 * 2 + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, goal: torch.Tensor) -> torch.distributions.MultivariateNormal:
        batch_size = rgb.shape[0]
        
        # RGB 특징 추출
        rgb_x = F.relu(self.rgb_conv1(rgb))
        rgb_x = F.relu(self.rgb_conv2(rgb_x))
        rgb_x = F.relu(self.rgb_conv3(rgb_x))
        rgb_x = rgb_x.view(batch_size, -1)
        
        # Depth 특징 추출
        depth_x = F.relu(self.depth_conv1(depth))
        depth_x = F.relu(self.depth_conv2(depth_x))
        depth_x = F.relu(self.depth_conv3(depth_x))
        depth_x = depth_x.view(batch_size, -1)
        
        # 특징 융합
        x = torch.cat([rgb_x, depth_x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        
        # 고정된 표준편차
        sigma = 0.1 * torch.ones_like(mu)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))

def extract_sensor_data(obs):
    """관찰에서 센서 데이터 추출"""
    # image 데이터에서 RGB 추출 (마지막 프레임 사용)
    if 'image' in obs:
        # image shape: (H, W, C*stack_size)
        rgb_data = obs['image'][..., -3:].squeeze(-1)  # 마지막 3채널 (RGB)
        rgb_data = (rgb_data * 255).astype(np.uint8)
    else:
        rgb_data = None
    
    # 다른 센서 데이터들은 별도로 처리해야 할 수 있음
    # MetaUrban의 센서 시스템에 따라 depth, semantic 데이터 접근 방법이 다를 수 있음
    depth_data = None
    semantic_data = None
    
    return rgb_data, depth_data, semantic_data

def visualize_sensor_data(env, num_steps=10, save_images=True):
    """센서 데이터 시각화 함수"""
    obs, info = env.reset()
    
    print("=== 센서 데이터 시각화 ===")
    print(f"환경 정보: {info}")
    print(f"관찰 키들: {obs.keys()}")
    print(f"에이전트 위치: {env.agent.position}")
    print(f"에이전트 방향: {env.agent.heading_theta}")
    
    # 관찰 데이터 구조 확인
    if 'image' in obs:
        print(f"Image shape: {obs['image'].shape}")
        print(f"Image dtype: {obs['image'].dtype}")
        print(f"Image range: [{obs['image'].min():.3f}, {obs['image'].max():.3f}]")
    
    for step in range(num_steps):
        # 센서 데이터 추출
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        print(f"\n--- Step {step} ---")
        if rgb_data is not None:
            print(f"RGB shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, range: [{rgb_data.min()}, {rgb_data.max()}]")
        print(f"Agent position: {env.agent.position}")
        print(f"Agent heading: {env.agent.heading_theta:.3f}")
        
        # 목표 지점 계산
        ego_goal_position = np.array([0.0, 0.0])
        if hasattr(env.agent, 'navigation') and env.agent.navigation:
            nav = env.agent.navigation
            waypoints = nav.checkpoints
            
            k = min(15, len(waypoints)-1) if len(waypoints) > 0 else 0
            if len(waypoints) > k:
                global_target = waypoints[k]
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
                print(f"Goal (egocentric): {ego_goal_position}")
        
        # 시각화
        if rgb_data is not None:
            if depth_data is not None and semantic_data is not None:
                # 모든 센서 데이터가 있는 경우
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'MetaUrban Sensor Data - Step {step}', fontsize=14)
                
                # RGB 이미지
                axes[0, 0].imshow(rgb_data)
                axes[0, 0].set_title('RGB Camera')
                axes[0, 0].axis('off')
                
                # Depth 이미지
                depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)
                im_depth = axes[0, 1].imshow(depth_normalized, cmap='plasma')
                axes[0, 1].set_title('Depth Camera')
                axes[0, 1].axis('off')
                plt.colorbar(im_depth, ax=axes[0, 1], shrink=0.8)
                
                # Semantic 이미지
                semantic_normalized = (semantic_data - semantic_data.min()) / (semantic_data.max() - semantic_data.min() + 1e-8)
                im_semantic = axes[1, 0].imshow(semantic_normalized, cmap='tab20')
                axes[1, 0].set_title('Semantic Camera')
                axes[1, 0].axis('off')
                plt.colorbar(im_semantic, ax=axes[1, 0], shrink=0.8)
                
                info_ax = axes[1, 1]
            else:
                # RGB만 있는 경우
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'MetaUrban RGB Camera Data - Step {step}', fontsize=14)
                
                # RGB 이미지
                axes[0].imshow(rgb_data)
                axes[0].set_title('RGB Camera')
                axes[0].axis('off')
                
                info_ax = axes[1]
            
            # 에이전트 정보 텍스트
            info_text = f"""Agent Info:
Position: ({env.agent.position[0]:.2f}, {env.agent.position[1]:.2f})
Heading: {env.agent.heading_theta:.3f} rad
Speed: {env.agent.speed:.2f} m/s
Goal (ego): ({ego_goal_position[0]:.2f}, {ego_goal_position[1]:.2f})

Sensor Info:
RGB: {rgb_data.shape if rgb_data is not None else 'None'}
Depth: {depth_data.shape if depth_data is not None else 'None'}
Semantic: {semantic_data.shape if semantic_data is not None else 'None'}

Camera Position: Front of vehicle
Camera View: First-person perspective
Resolution: {SENSOR_SIZE[0]}x{SENSOR_SIZE[1]}"""
            
            info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            info_ax.set_xlim(0, 1)
            info_ax.set_ylim(0, 1)
            info_ax.axis('off')
            
            plt.tight_layout()
            
            if save_images:
                os.makedirs('sensor_visualization', exist_ok=True)
                plt.savefig(f'sensor_visualization/step_{step:03d}.png', dpi=150, bbox_inches='tight')
                print(f"Saved: sensor_visualization/step_{step:03d}.png")
            
            plt.show()
        
        # 랜덤 액션으로 다음 스텝
        action = [np.random.uniform(-0.5, 1.0), np.random.uniform(-0.5, 0.5)]  # [throttle, steering]
        action = [0,1]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

class MetaUrbanInference:
    """학습된 모델로 추론하는 클래스"""
    
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로드
        self.actor = Actor(hidden_dim=512).to(self.device)
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
            print("Using randomly initialized model for demo")
    
    def obs_to_tensor(self, obs_data):
        """관찰을 텐서로 변환"""
        # RGB: (H, W, 3) -> (1, 3, H, W)
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Depth가 없으면 더미 데이터 생성
        if obs_data['depth'] is not None:
            depth = torch.tensor(obs_data['depth'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            # RGB에서 그레이스케일로 변환하여 더미 depth 생성
            gray = torch.mean(rgb, dim=1, keepdim=True)
            depth = gray
        
        # Goal: (2,) -> (1, 2)
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        return rgb.to(self.device), depth.to(self.device), goal.to(self.device)
    
    def predict_action(self, obs_data, deterministic=False):
        """관찰에서 행동 예측"""
        rgb_tensor, depth_tensor, goal_tensor = self.obs_to_tensor(obs_data)
        
        with torch.no_grad():
            action_dist = self.actor(rgb_tensor, depth_tensor, goal_tensor)
            
            if deterministic:
                action = action_dist.mean[0]
            else:
                action = action_dist.sample()[0]
        
        return action.cpu().numpy()
    
    def run_inference(self, env, max_steps=1000, visualize=False, save_video=False):
        """추론 실행"""
        obs, info = env.reset()
        
        step_count = 0
        total_reward = 0
        trajectory = []
        
        if save_video:
            os.makedirs('inference_video', exist_ok=True)
            frames = []
        
        print("=== 추론 시작 ===")
        print(f"관찰 키들: {obs.keys()}")

        # dict_keys(['image','state'])
        
        
        while step_count < max_steps:
            # 센서 데이터 추출
            rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
            
            if rgb_data is None:
                print("RGB 데이터를 찾을 수 없습니다.")
                break
            
            # 목표 지점 계산
            ego_goal_position = np.array([0.0, 0.0])
            if hasattr(env.agent, 'navigation') and env.agent.navigation:
                nav = env.agent.navigation
                waypoints = nav.checkpoints
                
                k = min(15, len(waypoints)-1) if len(waypoints) > 0 else 0
                if len(waypoints) > k:
                    global_target = waypoints[k]
                    agent_pos = env.agent.position
                    agent_heading = env.agent.heading_theta
                    ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
            
            # 관찰 데이터 준비
            obs_data = {
                'rgb': rgb_data,
                'depth': depth_data,
                'goal': ego_goal_position
            }
            
            # 행동 예측
            action = self.predict_action(obs_data, deterministic=True)
            throttle, steering = float(action[0]), float(action[1])
            
            # 행동을 적절한 범위로 클리핑
            throttle = np.clip(throttle, -1.0, 1.0)
            steering = np.clip(steering, -1.0, 1.0)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step([throttle, steering])
            
            total_reward += reward
            step_count += 1
            
            # 로그 출력
            if step_count % 50 == 0:
                print(f"Step {step_count}: Action=[{throttle:.3f}, {steering:.3f}], "
                      f"Reward={reward:.3f}, Total={total_reward:.3f}, "
                      f"Pos=({env.agent.position[0]:.2f}, {env.agent.position[1]:.2f})")
            
            # 시각화
            if visualize and step_count % 20 == 0:
                self.visualize_step(obs_data, [throttle, steering], reward, step_count)
            
            # 비디오 프레임 저장
            if save_video:
                frame = self.create_frame(obs_data, [throttle, steering], reward, step_count)
                if frame is not None:
                    frames.append(frame)
            
            trajectory.append({
                'obs': obs_data,
                'action': [throttle, steering],
                'reward': reward,
                'position': env.agent.position,
                'heading': env.agent.heading_theta
            })
            
            if terminated or truncated:
                print(f"Episode finished at step {step_count}")
                break
        
        print(f"=== 추론 완료 ===")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward/step_count:.3f}")
        
        if save_video and frames:
            self.save_video_frames(frames)
        
        return trajectory, total_reward
    
    def visualize_step(self, obs_data, action, reward, step):
        """단일 스텝 시각화"""
        if obs_data['rgb'] is None:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Inference Step {step} - Action: [{action[0]:.3f}, {action[1]:.3f}], Reward: {reward:.3f}')
        
        # RGB
        axes[0].imshow(obs_data['rgb'])
        axes[0].set_title('RGB Input')
        axes[0].axis('off')
        
        # Action and goal info
        info_text = f"""Prediction Results:
Throttle: {action[0]:.3f}
Steering: {action[1]:.3f}
Reward: {reward:.3f}

Goal (egocentric):
X: {obs_data['goal'][0]:.3f}
Y: {obs_data['goal'][1]:.3f}

Note: Using RGB camera only
Depth: {'Available' if obs_data['depth'] is not None else 'Simulated from RGB'}"""
        
        axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes, 
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_frame(self, obs_data, action, reward, step):
        """비디오용 프레임 생성"""
        if obs_data['rgb'] is None:
            return None
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(f'Step {step} - Throttle: {action[0]:.2f}, Steering: {action[1]:.2f}, Reward: {reward:.2f}')
        
        ax.imshow(obs_data['rgb'])
        ax.set_title('RGB Camera View')
        ax.axis('off')
        
        plt.tight_layout()
        
        # 이미지를 numpy array로 변환
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return frame
    
    def save_video_frames(self, frames):
        """프레임들을 이미지로 저장"""
        for i, frame in enumerate(frames):
            cv2.imwrite(f'inference_video/frame_{i:04d}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {len(frames)} frames to inference_video/")

def test_basic_visualization():
    """기본 시각화 테스트 (동작 확인된 코드 기반)"""
    print("=== 기본 RGB 카메라 테스트 ===")
    
    sensor_size = (200, 100)  # 작은 크기로 테스트
    cfg = dict(
        object_density=0.1,
        image_observation=True, 
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=1,
    )
    
    env = SidewalkStaticMetaUrbanEnv(cfg)
    frames = []
    
    try:
        obs, info = env.reset()
        print(f"초기 관찰 키들: {obs.keys()}")
        if 'image' in obs:
            print(f"Image shape: {obs['image'].shape}")
            print(f"Image range: [{obs['image'].min():.3f}, {obs['image'].max():.3f}]")
        
        for i in range(10):
            # RGB 데이터 추출
            if 'image' in obs:
                # ret = obs["image"] * 255  # [0., 1.] to [0, 255]
                ret = obs["image"].squeeze(-1) * 255  # [0., 1.] to [0, 255]
                ret = ret.astype(np.uint8)
                frames.append(ret)
                
                # 시각화
                plt.figure(figsize=(8, 6))
                plt.imshow(ret)
                plt.title(f'RGB Camera - Step {i}\nAgent Pos: ({env.agent.position[0]:.1f}, {env.agent.position[1]:.1f})')
                plt.axis('off')
                
                # 저장
                os.makedirs('basic_test', exist_ok=True)
                plt.savefig(f'basic_test/step_{i:03d}.png', dpi=100, bbox_inches='tight')
                plt.show()
            
            # 랜덤 액션
            action = [np.random.uniform(-0.5, 1.0), np.random.uniform(-0.3, 0.3)]
            action = [0,1]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
        print(f"총 {len(frames)} 프레임 수집 완료")
        
    finally:
        env.close()

def main():
    """메인 함수"""
    print("MetaUrban 센서 데이터 시각화 및 추론 테스트")
    
    # 1. 기본 동작 테스트
    print("\n1. 기본 RGB 카메라 테스트...")
    test_basic_visualization()
    
    # 2. 고급 시각화 테스트
    print("\n2. 고급 센서 데이터 시각화...")
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    
    try:
        visualize_sensor_data(env, num_steps=5, save_images=True)
        
        # 3. 추론 테스트
        print("\n3. 학습된 모델로 추론 시작...")
        inference = MetaUrbanInference('metaurban_actor.pt')
        
        trajectory, total_reward = inference.run_inference(
            env, 
            max_steps=100, 
            visualize=True, 
            save_video=True
        )
        
        # 4. 궤적 분석
        print("\n4. 궤적 분석...")
        if trajectory:
            positions = [step['position'] for step in trajectory]
            rewards = [step['reward'] for step in trajectory]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
            plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5)
            plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Agent Trajectory')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.subplot(1, 2, 2)
            plt.plot(rewards, 'r-', linewidth=1.5, alpha=0.8)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('Reward Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('trajectory_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
        
    finally:
        env.close()
    
    print("완료!")

if __name__ == "__main__":
    main()