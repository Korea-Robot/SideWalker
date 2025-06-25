# validation_test_enhanced.py
"""
주요 기능 (개선된 버전)

모델 로드: 저장된 .pth 파일에서 학습된 정책 로드
시각적 검증: 환경 렌더링으로 실제 주행 확인
Goal Vector 시각화: 목표 방향을 화살표로 표시
직접 목표 추적: Goal Vector를 향한 액션 생성
비디오 저장: 주행 과정을 동영상으로 저장
통계 분석: 여러 에피소드의 성공률 등 분석

🎯 사용법
bashpython validation_test_enhanced.py
📊 출력 결과

센서 시각화: RGB, Depth, Semantic 카메라 이미지 + Goal Vector 표시
주행 비디오: validation_run.avi 파일로 저장 (Goal Vector 오버레이 포함)
상세 통계: 성공률, 충돌률, 평균 보상 등

⚙️ 설정 변경 포인트

model_path: 실제 모델 파일 경로로 변경
num_episodes: 테스트할 에피소드 수
max_steps: 에피소드당 최대 스텝 수
use_render=True: 시각화 활성화
use_goal_tracking=True: Goal Vector 기반 액션 사용

"""
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math

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
    map='XSOS', 
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

class GoalVectorController:
    """Goal Vector 기반 제어기"""
    
    def __init__(self, max_steer=0.5, max_throttle=0.8, goal_threshold=2.0):
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        self.goal_threshold = goal_threshold
        
    def get_action_from_goal_vec(self, goal_vec, current_speed=0):
        """Goal Vector를 기반으로 액션 생성"""
        goal_distance = np.linalg.norm(goal_vec)
        
        if goal_distance < 0.1:  # 매우 가까운 경우
            return np.array([0.0, 0.0])  # 정지
        
        # 목표 각도 계산 (차량 앞쪽이 x축)
        goal_angle = math.atan2(goal_vec[1], goal_vec[0])
        
        # 조향각 계산 (-1 ~ 1로 정규화)
        steering = np.clip(goal_angle / math.pi, -1.0, 1.0) * self.max_steer
        
        # 스로틀 계산 (목표까지의 거리와 현재 속도 고려)
        desired_speed = min(goal_distance * 0.5, 15.0)  # 최대 15m/s
        speed_diff = desired_speed - current_speed
        
        if speed_diff > 0:
            throttle = np.clip(speed_diff * 0.1, 0.1, self.max_throttle)
        else:
            throttle = np.clip(speed_diff * 0.05, -0.5, 0.0)  # 브레이크
        
        return np.array([steering, throttle])

class ValidationTester:
    """학습된 모델 검증 클래스 (Goal Vector 시각화 포함)"""
    
    def __init__(self, model_path: str, device='cuda', use_goal_tracking=False):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_goal_tracking = use_goal_tracking
        print(f"Using device: {self.device}")
        print(f"Goal tracking mode: {'ON' if use_goal_tracking else 'OFF'}")
        
        # Goal Vector 제어기
        self.goal_controller = GoalVectorController()
        
        # 모델 로드 (Goal tracking이 아닌 경우에만)
        if not use_goal_tracking:
            self.agent = PPOAgent(PPOConfig(), self.device)
            self.device = torch.device('cpu')
            
            # 모델 로드
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
        
    def draw_goal_vector_on_frame(self, frame, goal_vec, vehicle_pos=None):
        """프레임에 Goal Vector 시각화"""
        if frame is None:
            return frame
            
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Goal Vector 크기 정규화 (화면 크기에 맞게)
        goal_distance = np.linalg.norm(goal_vec)
        if goal_distance > 0.1:
            # 벡터 방향을 화면 좌표로 변환
            scale = min(w, h) * 0.2  # 화살표 길이
            arrow_end_x = int(center_x + goal_vec[0] * scale / goal_distance)
            arrow_end_y = int(center_y - goal_vec[1] * scale / goal_distance)  # y축 반전
            
            # 화살표 그리기
            cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y), 
                          (0, 255, 0), 3, tipLength=0.3)
            
            # 목표 거리 텍스트
            cv2.putText(frame, f"Goal: {goal_distance:.1f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 목표 각도 텍스트
            goal_angle = math.degrees(math.atan2(goal_vec[1], goal_vec[0]))
            cv2.putText(frame, f"Angle: {goal_angle:.1f}°", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 중심점 표시
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        
        return frame
        
    def run_episode(self, max_steps=1000, save_video=False):
        """단일 에피소드 실행 (Goal Vector 시각화 포함)"""
        # 환경 리셋
        obs, _ = self.env.reset()
        nav = self.env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        
        if not self.use_goal_tracking:
            state = self.obs_processor.preprocess_observation(obs)
        
        episode_reward = 0
        step = 0
        trajectory = []
        
        # 비디오 저장 설정
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('validation_run_with_goal.avi', fourcc, 20.0, (800, 600))
        
        print(f"Starting validation - Initial goal distance: {np.linalg.norm(obs['goal_vec']):.2f}")
        print(f"Control mode: {'Goal Vector Tracking' if self.use_goal_tracking else 'Trained Policy'}")
        
        while step < max_steps:
            # 행동 선택
            if self.use_goal_tracking:
                # Goal Vector 기반 직접 제어
                current_speed = self.env.vehicle.speed if hasattr(self.env.vehicle, 'speed') else 0
                action = self.goal_controller.get_action_from_goal_vec(obs["goal_vec"], current_speed)
                action = torch.tensor(action, dtype=torch.float32)
            else:
                # 학습된 정책 사용
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
                'goal_vec': next_obs["goal_vec"].copy(),
                'speed': info.get('speed', 0),
                'crash': info.get('crash', False),
                'out_of_road': info.get('out_of_road', False),
                'arrive_dest': info.get('arrive_dest', False)
            })
            
            # 렌더링 및 비디오 저장 (Goal Vector 오버레이 포함)
            if save_video:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(frame, (800, 600))
                    
                    # Goal Vector 시각화 추가
                    frame = self.draw_goal_vector_on_frame(frame, next_obs["goal_vec"])
                    
                    # 제어 모드 표시
                    mode_text = "Goal Tracking" if self.use_goal_tracking else "Trained Policy"
                    cv2.putText(frame, f"Mode: {mode_text}", 
                               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    out.write(frame)
            
            episode_reward += reward
            step += 1
            
            # 상태 출력
            if step % 50 == 0:
                goal_dist = np.linalg.norm(next_obs["goal_vec"])
                speed = info.get('speed', 0)
                action_str = f"[{action.squeeze().numpy()[0]:.2f}, {action.squeeze().numpy()[1]:.2f}]"
                print(f"Step {step}: Action={action_str}, Reward={reward:.2f}, Goal_dist={goal_dist:.2f}, Speed={speed:.1f}")
            
            # 종료 조건
            if done or truncated:
                if info.get('arrive_dest', False):
                    print(f"SUCCESS! Arrived at destination in {step} steps!")
                elif info.get('crash', False):
                    print(f"CRASH! Episode ended at step {step}")
                elif info.get('out_of_road', False):
                    print(f"OUT OF ROAD! Episode ended at step {step}")
                break
            
            # 상태 업데이트
            obs = next_obs
            if not self.use_goal_tracking:
                state = self.obs_processor.preprocess_observation(obs)
        
        if save_video:
            out.release()
            filename = 'validation_run_with_goal.avi'
            print(f"Video saved as '{filename}'")
        
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
        
        return results
    
def main():
    """메인 검증 함수"""
    # 설정
    model_path = "checkpoints/final_model.pth"
    use_goal_tracking = True  # True: Goal Vector 기반 제어, False: 학습된 모델 사용
    
    if not use_goal_tracking and not Path(model_path).exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("Goal tracking 모드로 전환합니다...")
        use_goal_tracking = True
    
    # 검증 테스터 초기화
    tester = ValidationTester(model_path, use_goal_tracking=use_goal_tracking)
    
    
    # 단일 에피소드 실행 (비디오 저장)
    print(f"\nRunning single episode with video recording...")
    tester.run_episode(save_video=True)
    
    # 여러 에피소드 통계
    print(f"\nRunning multiple episodes for statistics...")
    tester.run_multiple_episodes(num_episodes=3)
    
if __name__ == "__main__":
    main()