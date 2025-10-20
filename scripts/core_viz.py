# validation_test.py
"""
주요 기능

모델 로드: 저장된 .pth 파일에서 학습된 정책 로드
시각적 검증: 환경 렌더링으로 실제 주행 확인
비디오 저장: 주행 과정을 동영상으로 저장
통계 분석: 여러 에피소드의 성공률 등 분석

🎯 사용법
bashpython validation_test.py
📊 출력 결과

센서 시각화: RGB, Depth, Semantic 카메라 이미지
주행 비디오: validation_run.avi 파일로 저장
상세 통계: 성공률, 충돌률, 평균 보상 등

⚙️ 설정 변경 포인트

model_path: 실제 모델 파일 경로로 변경
num_episodes: 테스트할 에피소드 수
max_steps: 에피소드당 최대 스텝 수
use_render=True: 시각화 활성화

이 코드로 학습된 에이전트가 실제로 얼마나 잘 주행하는지 직관적으로 확인 가능!

"""
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# 최상위 경로에서 import
# from scripts.autonomous_driving_ppo_modular import PPOAgent, PPOConfig, ObservationProcessor, RewardCalculator

# 상대 경로로 import
from scripts.core import PPOAgent, PPOConfig, ObservationProcessor, RewardCalculator

# ============================================================================
# 환경 설정 (동일)
# ============================================================================
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True,  # 시각화를 위해 True로 변경
    map='X', 
    manual_control=False, 
    crswalk_density=1, 
    object_density=0.1, 
    walk_on_all_regions=False,
    drivable_area_extension=55, 
    height_scale=1, 
    horizon=300,
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=True, 
    max_lateral_dist=5.0,
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

class ValidationTester:
    """학습된 모델 검증 클래스"""
    
    def __init__(self, model_path: str, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.agent = PPOAgent(PPOConfig(), self.device)
        # self.agent.load_model(model_path,map_location=torch.device('cpu'))
        self.device = torch.device('cpu')
        
    # def load_model(self, filepath):
        """모델 로드"""
        # checkpoint = torch.load(model_path,map_location= self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.value.load_state_dict(checkpoint['value_state_dict'])
        self.agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.agent.stats = checkpoint['stats']
        self.agent.policy.eval()
        
        # 환경 초기화 (렌더링 활성화)
        self.env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
        
        # 유틸리티
        self.obs_processor = ObservationProcessor()
        self.reward_calculator = RewardCalculator()
        
    def run_episode(self, max_steps=1000, save_video=False):
        """단일 에피소드 실행"""
        # 환경 리셋
        obs, _ = self.env.reset()
        nav = self.env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        state = self.obs_processor.preprocess_observation(obs)
        
        episode_reward = 0
        step = 0
        trajectory = []
        
        # 비디오 저장 설정
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('validation_run.avi', fourcc, 20.0, (800, 600))
        
        print(f"Starting validation - Initial goal distance: {np.linalg.norm(obs['goal_vec']):.2f}")
        
        while step < max_steps:
            # 행동 선택 (탐험 없이)
            with torch.no_grad():
                action, _, _ = self.agent.select_action(state)
            
            # 환경 스텝
            next_obs, _, done, truncated, info = self.env.step(action.squeeze().numpy())
            
            # goal_vec 업데이트
            nav = self.env.vehicle.navigation.get_navi_info()
            next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            
            # 보상 계산
            reward = self.reward_calculator.compute_reward(obs, action, next_obs, done, info)
            
            # 상태 정보 저장
            trajectory.append({
                'step': step,
                'action': action.squeeze().numpy(),
                'reward': reward,
                'goal_distance': np.linalg.norm(next_obs["goal_vec"]),
                'speed': info.get('speed', 0),
                'crash': info.get('crash', False),
                'out_of_road': info.get('out_of_road', False),
                'arrive_dest': info.get('arrive_dest', False)
            })
            
            # 렌더링 및 비디오 저장
            if save_video:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(frame, (800, 600))
                    out.write(frame)
            
            episode_reward += reward
            step += 1
            
            # 상태 출력
            if step % 50 == 0:
                goal_dist = np.linalg.norm(next_obs["goal_vec"])
                speed = info.get('speed', 0)
                print(f"Step {step}: Reward={reward:.2f}, Goal_dist={goal_dist:.2f}, Speed={speed:.1f}")
            
            # 종료 조건
            if done or truncated:
                if info.get('arrive_dest', False):
                    # print(f"SUCCESS! Arrived at destination in {step} steps!")
                    continue
                elif info.get('crash', False):
                    print(f"CRASH! Episode ended at step {step}")
                elif info.get('out_of_road', False):
                    print(f"OUT OF ROAD! Episode ended at step {step}")
                break
            
            # 상태 업데이트
            obs = next_obs
            state = self.obs_processor.preprocess_observation(obs)
        
        if save_video:
            out.release()
            print("Video saved as 'validation_run.avi'")
        
        print(f"Episode completed - Steps: {step}, Total Reward: {episode_reward:.2f}")
        return trajectory, episode_reward, step
    
    def run_multiple_episodes(self, num_episodes=5):
        """여러 에피소드 실행 및 통계"""
        results = []
        
        for i in range(num_episodes):
            print(f"\n=== Episode {i+1}/{num_episodes} ===")
            trajectory, reward, steps = self.run_episode()
            
            # 성공 여부 판단
            success = any(t['arrive_dest'] for t in trajectory)
            crash = any(t['crash'] for t in trajectory)
            
            results.append({
                'episode': i+1,
                'reward': reward,
                'steps': steps,
                'success': success,
                'crash': crash,
                'final_goal_distance': trajectory[-1]['goal_distance'] if trajectory else float('inf')
            })
        
        # 통계 출력
        self.print_statistics(results)
        return results
    
    def print_statistics(self, results):
        """통계 출력"""
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        
        total_episodes = len(results)
        successes = sum(1 for r in results if r['success'])
        crashes = sum(1 for r in results if r['crash'])
        
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_goal_distance = np.mean([r['final_goal_distance'] for r in results])
        
        print(f"Total Episodes: {total_episodes}")
        print(f"Success Rate: {successes/total_episodes*100:.1f}% ({successes}/{total_episodes})")
        print(f"Crash Rate: {crashes/total_episodes*100:.1f}% ({crashes}/{total_episodes})")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Final Goal Distance: {avg_goal_distance:.2f}")
        
        print("\nIndividual Results:")
        for r in results:
            status = "SUCCESS" if r['success'] else ("CRASH" if r['crash'] else "TIMEOUT")
            print(f"Episode {r['episode']}: {status} - Reward: {r['reward']:.2f}, Steps: {r['steps']}, Goal Dist: {r['final_goal_distance']:.2f}")
    
    def visualize_sensors(self):
        """센서 데이터 시각화"""
        obs, _ = self.env.reset()
        
        # 센서 데이터 추출 : visualize image observation
        rgb = obs["image"][..., -1]  # RGB 이미지
        depth = obs["depth"][..., -1]  # Depth 이미지  
        depth = np.concatenate([depth, depth, depth], axis=-1) # align channel
        semantic = obs["semantic"][..., -1]  # Semantic 이미지
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Camera')
        axes[0].axis('off')
        
        axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title('Depth Camera')
        axes[1].axis('off')
        
        axes[2].imshow(semantic)
        axes[2].set_title('Semantic Camera')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sensor_visualization.png')
        plt.show()
        
        print("Sensor visualization saved as 'sensor_visualization.png'")

def main():
    """메인 검증 함수"""
    # 모델 경로 설정 (실제 경로로 변경하세요)
    model_path = "checkpoints/final_model.pth"  # 또는 "checkpoints/model_episode_xxx.pth"
    # model_path = "final_model.pth"  # 또는 "checkpoints/model_episode_xxx.pth"
    
    if not Path(model_path).exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("사용 가능한 모델 파일들:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob("*.pth"):
                print(f"  {file}")
        return
    
    # 검증 테스터 초기화
    tester = ValidationTester(model_path)
    
    # 센서 시각화
    print("Visualizing sensors...")
    tester.visualize_sensors()
    
    # 단일 에피소드 실행 (비디오 저장)
    print("\nRunning single episode with video recording...")
    tester.run_episode(save_video=True)
    
    # 여러 에피소드 통계
    print("\nRunning multiple episodes for statistics...")
    tester.run_multiple_episodes(num_episodes=5)

if __name__ == "__main__":
    main()