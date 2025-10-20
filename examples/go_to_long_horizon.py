# validation_test_enhanced.py
"""
개선된 Goal Vector 기반 검증 시스템
- 더 긴 경로 설정
- 연속적인 웨이포인트 시스템
- 최소 스텝 보장
"""
import numpy as np
import torch
import cv2
import math
from pathlib import Path

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# 상대 경로로 import
from scripts.core import PPOAgent, PPOConfig, ObservationProcessor, RewardCalculator

# ============================================================================
# 환경 설정 (더 긴 맵과 복잡한 경로)
# ============================================================================
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=False,
    map='X',  # 더 큰 맵 사용
    manual_control=False, 
    crswalk_density=0.8, 
    object_density=0.2, 
    walk_on_all_regions=False,
    drivable_area_extension=100,  # 더 넓은 도로 영역
    height_scale=1, 
    horizon=2000,  # 더 긴 호라이즌
    vehicle_config=dict(enable_reverse=True),
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    relax_out_of_road_done=True, 
    max_lateral_dist=8.0,  # 더 관대한 도로 이탈 조건
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

class WaypointNavigator:
    """연속적인 웨이포인트 기반 네비게이션"""
    
    def __init__(self, min_waypoint_distance=20.0, waypoint_reach_threshold=5.0):
        self.min_waypoint_distance = min_waypoint_distance
        self.waypoint_reach_threshold = waypoint_reach_threshold
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.total_waypoints_reached = 0
        
    def generate_waypoints(self, env):
        """환경 기반 웨이포인트 생성"""
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.total_waypoints_reached = 0
        
        # 차량 현재 위치
        vehicle_pos = env.vehicle.position
        
        # 기본 네비게이션 목표
        nav_info = env.vehicle.navigation.get_navi_info()
        final_target = np.array([nav_info[0], nav_info[1]]) + vehicle_pos[:2]

        breakpoint()
        
        # 최종 목표까지의 거리 계산
        total_distance = np.linalg.norm(final_target - vehicle_pos[:2])
        
        if total_distance < self.min_waypoint_distance:
            # 거리가 짧으면 인위적으로 웨이포인트 생성
            self._generate_artificial_waypoints(vehicle_pos[:2], final_target)
        else:
            # 기존 목표까지 중간 웨이포인트 생성
            self._generate_intermediate_waypoints(vehicle_pos[:2], final_target)
        
        print(f"Generated {len(self.waypoints)} waypoints, total distance: {total_distance:.1f}m")
        return len(self.waypoints) > 0
    
    def _generate_artificial_waypoints(self, start_pos, final_target):
        """인위적으로 더 긴 경로 생성"""
        # 현재 위치에서 여러 방향으로 웨이포인트 생성
        angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]  # 60도씩
        
        # breakpoint()
        current_pos = list(start_pos).copy()
        
        for i, angle in enumerate(angles):
            # 각 방향으로 일정 거리만큼 웨이포인트 생성
            waypoint_distance = self.min_waypoint_distance + i * 10
            waypoint = current_pos + waypoint_distance * np.array([np.cos(angle), np.sin(angle)])
            self.waypoints.append(waypoint)
            current_pos = waypoint
        
        # 마지막에 최종 목표 추가
        self.waypoints.append(final_target)
    
    def _generate_intermediate_waypoints(self, start_pos, final_target):
        """시작점과 끝점 사이에 중간 웨이포인트 생성"""
        direction = final_target - start_pos
        total_distance = np.linalg.norm(direction)
        direction_normalized = direction / total_distance
        
        # 웨이포인트 개수 결정
        num_waypoints = max(3, int(total_distance / self.min_waypoint_distance))
        
        for i in range(1, num_waypoints + 1):
            # 직선 경로상의 웨이포인트
            progress = i / num_waypoints
            waypoint = start_pos + progress * direction
            
            # 약간의 랜덤 편차 추가 (더 흥미로운 경로)
            if i < num_waypoints:  # 마지막 웨이포인트(최종 목표)는 편차 없음
                perpendicular = np.array([-direction_normalized[1], direction_normalized[0]])
                random_offset = perpendicular * np.random.uniform(-10, 10)
                waypoint += random_offset
            
            self.waypoints.append(waypoint)
    
    def get_current_goal_vector(self, vehicle_pos):
        """현재 목표 웨이포인트에 대한 goal vector 반환"""
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return np.array([0.0, 0.0])
        
        current_target = self.waypoints[self.current_waypoint_idx]
        goal_vec = current_target - vehicle_pos[:2]
        
        # 웨이포인트 도달 체크
        if np.linalg.norm(goal_vec) < self.waypoint_reach_threshold:
            self.total_waypoints_reached += 1
            self.current_waypoint_idx += 1
            print(f"Waypoint {self.total_waypoints_reached} reached! "
                  f"Moving to waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}")
            
            # 다음 웨이포인트로
            if self.current_waypoint_idx < len(self.waypoints):
                current_target = self.waypoints[self.current_waypoint_idx]
                goal_vec = current_target - vehicle_pos[:2]
        
        return goal_vec
    
    def is_mission_complete(self):
        """모든 웨이포인트 도달 여부"""
        return self.current_waypoint_idx >= len(self.waypoints)
    
    def get_progress_info(self):
        """진행 상황 정보"""
        return {
            'current_waypoint': self.current_waypoint_idx + 1,
            'total_waypoints': len(self.waypoints),
            'waypoints_reached': self.total_waypoints_reached,
            'progress_percent': (self.current_waypoint_idx / max(1, len(self.waypoints))) * 100
        }

class GoalVectorController:
    """Goal Vector 기반 제어기"""
    
    def __init__(self, max_steer=0.4, max_throttle=0.7):
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        
    def get_action_from_goal_vec(self, goal_vec, current_speed=0):
        """Goal Vector를 기반으로 액션 생성"""
        goal_distance = np.linalg.norm(goal_vec)
        
        if goal_distance < 0.1:
            return np.array([0.0, 0.0])
        
        # 목표 각도 계산
        goal_angle = math.atan2(goal_vec[1], goal_vec[0])
        
        # 조향각 계산 (더 부드러운 조향)
        steering = np.clip(goal_angle / (math.pi/2), -1.0, 1.0) * self.max_steer
        
        # 스로틀 계산 (거리 기반)
        desired_speed = min(goal_distance * 0.3, 12.0 if goal_distance > 10 else 8.0)
        speed_diff = desired_speed - current_speed
        
        if speed_diff > 0:
            throttle = np.clip(speed_diff * 0.2, 0.1, self.max_throttle)
        else:
            throttle = np.clip(speed_diff * 0.1, -0.3, 0.0)
        
        return np.array([steering, throttle])

class ValidationTester:
    """학습된 모델 검증 클래스"""
    
    def __init__(self, model_path: str = None, device='cuda', use_goal_tracking=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_goal_tracking = use_goal_tracking
        
        print(f"Using device: {self.device}")
        print(f"Goal tracking mode: {'ON' if use_goal_tracking else 'OFF'}")
        
        # 컨트롤러 및 네비게이터 초기화
        self.goal_controller = GoalVectorController()
        self.waypoint_navigator = WaypointNavigator()
        
        # 모델 로드 (Goal tracking이 아닌 경우에만)
        if not use_goal_tracking and model_path:
            self.agent = PPOAgent(PPOConfig(), self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.value.load_state_dict(checkpoint['value_state_dict'])
            self.agent.policy.eval()
        
        # 환경 초기화
        self.env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
        
        # 유틸리티
        self.obs_processor = ObservationProcessor()
        self.reward_calculator = RewardCalculator()
        
    def run_episode(self, max_steps=2000, min_steps=100, save_video=False):
        """단일 에피소드 실행"""
        # 환경 리셋
        obs, _ = self.env.reset()
        
        # 웨이포인트 생성
        if not self.waypoint_navigator.generate_waypoints(self.env):
            print("Failed to generate waypoints!")
            return [], 0, 0
        
        if not self.use_goal_tracking:
            state = self.obs_processor.preprocess_observation(obs)
        
        episode_reward = 0
        step = 0
        trajectory = []
        
        # 비디오 저장 설정
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('validation_run.avi', fourcc, 20.0, (800, 600))
        
        progress_info = self.waypoint_navigator.get_progress_info()
        print(f"Starting episode - Total waypoints: {progress_info['total_waypoints']}")
        
        while step < max_steps:
            # 현재 goal vector 가져오기
            vehicle_pos = self.env.vehicle.position
            goal_vec = self.waypoint_navigator.get_current_goal_vector(vehicle_pos)
            
            # obs에 goal_vec 추가
            obs["goal_vec"] = goal_vec.astype(np.float32)
            
            # 행동 선택
            if self.use_goal_tracking:
                current_speed = getattr(self.env.vehicle, 'speed', 0)
                action = self.goal_controller.get_action_from_goal_vec(goal_vec, current_speed)
                action = torch.tensor(action, dtype=torch.float32)
            else:
                with torch.no_grad():
                    action, _, _ = self.agent.select_action(state)
            
            # 환경 스텝
            next_obs, _, done, truncated, info = self.env.step(action.squeeze().numpy())
            
            # 보상 계산
            # reward = self.reward_calculator.compute_reward(obs, action, next_obs, done, info)
            reward = 0
            # 진행 상황 체크
            progress_info = self.waypoint_navigator.get_progress_info()
            mission_complete = self.waypoint_navigator.is_mission_complete()
            
            # 상태 정보 저장
            trajectory.append({
                'step': step,
                'action': action.squeeze().numpy(),
                'reward': reward,
                'goal_distance': np.linalg.norm(goal_vec),
                'goal_vec': goal_vec.copy(),
                'speed': info.get('speed', 0),
                'crash': info.get('crash', False),
                'out_of_road': info.get('out_of_road', False),
                'mission_complete': mission_complete,
                'waypoint_progress': progress_info
            })
            
            # 비디오 저장
            if save_video:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(frame, (800, 600))
                    
                    # 정보 오버레이
                    cv2.putText(frame, f"Waypoint: {progress_info['current_waypoint']}/{progress_info['total_waypoints']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Progress: {progress_info['progress_percent']:.1f}%", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Goal: {np.linalg.norm(goal_vec):.1f}m", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    out.write(frame)
            
            episode_reward += reward
            step += 1
            
            # 상태 출력
            if step % 100 == 0:
                goal_dist = np.linalg.norm(goal_vec)
                speed = info.get('speed', 0)
                print(f"Step {step}: Waypoint {progress_info['current_waypoint']}/{progress_info['total_waypoints']}, "
                      f"Goal_dist={goal_dist:.1f}m, Speed={speed:.1f}m/s, Progress={progress_info['progress_percent']:.1f}%")
            
            # 종료 조건 체크
            early_termination = False
            if done or truncated:
                if info.get('crash', False):
                    print(f"CRASH! Episode ended at step {step}")
                    early_termination = True
                elif info.get('out_of_road', False):
                    print(f"OUT OF ROAD! Episode ended at step {step}")
                    early_termination = True
            
            # 미션 완료 체크
            if mission_complete:
                print(f"MISSION COMPLETE! All waypoints reached in {step} steps!")
                if step < min_steps:
                    print(f"Continuing to reach minimum steps ({min_steps})...")
                else:
                    break
            
            # 최소 스텝 미달시 계속 진행
            if early_termination and step < min_steps:
                print(f"Early termination but minimum steps not reached. Resetting...")
                obs, _ = self.env.reset()
                self.waypoint_navigator.generate_waypoints(self.env)
                step += 1
                continue
            elif early_termination:
                break
            
            # 상태 업데이트
            obs = next_obs
            if not self.use_goal_tracking:
                state = self.obs_processor.preprocess_observation(obs)
        
        if save_video:
            out.release()
            print(f"Video saved as 'validation_run.avi'")
        
        final_progress = self.waypoint_navigator.get_progress_info()
        print(f"Episode completed - Steps: {step}, Total Reward: {episode_reward:.2f}")
        print(f"Final progress: {final_progress['waypoints_reached']}/{final_progress['total_waypoints']} waypoints reached")
        
        return trajectory, episode_reward, step
    
    def run_multiple_episodes(self, num_episodes=3):
        """여러 에피소드 실행 및 통계"""
        results = []
        
        for i in range(num_episodes):
            print(f"\n=== Episode {i+1}/{num_episodes} ===")
            trajectory, reward, steps = self.run_episode()
            
            if not trajectory:
                continue
                
            # 성공 여부 판단
            mission_complete = trajectory[-1]['mission_complete']
            crash = any(t['crash'] for t in trajectory)
            waypoints_reached = trajectory[-1]['waypoint_progress']['waypoints_reached']
            
            results.append({
                'episode': i+1,
                'reward': reward,
                'steps': steps,
                'mission_complete': mission_complete,
                'crash': crash,
                'waypoints_reached': waypoints_reached,
                'final_goal_distance': trajectory[-1]['goal_distance']
            })
            
            print(f"Episode {i+1} summary:")
            print(f"  - Mission complete: {mission_complete}")
            print(f"  - Waypoints reached: {waypoints_reached}")
            print(f"  - Steps: {steps}")
            print(f"  - Reward: {reward:.2f}")
        
        # 전체 통계
        if results:
            success_rate = sum(r['mission_complete'] for r in results) / len(results) * 100
            avg_reward = sum(r['reward'] for r in results) / len(results)
            avg_steps = sum(r['steps'] for r in results) / len(results)
            avg_waypoints = sum(r['waypoints_reached'] for r in results) / len(results)
            
            print(f"\n=== Overall Statistics ===")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.1f}")
            print(f"Average waypoints reached: {avg_waypoints:.1f}")
        
        return results
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'env'):
            self.env.close()

def main():
    """메인 함수"""
    # 설정
    model_path = "checkpoints/final_model.pth"
    use_goal_tracking = True
    
    # 모델 파일 체크
    if not use_goal_tracking and not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Switching to goal tracking mode...")
        use_goal_tracking = True
    
    # 검증 테스터 초기화
    tester = ValidationTester(model_path, use_goal_tracking=use_goal_tracking)
    
    try:
        # 단일 에피소드 실행 (비디오 저장)
        print("Running single episode with video recording...")
        tester.run_episode(save_video=True, min_steps=200)
        
        # 여러 에피소드 통계
        print("\nRunning multiple episodes for statistics...")
        tester.run_multiple_episodes(num_episodes=3)
        
    finally:
        # 리소스 정리
        tester.cleanup()

if __name__ == "__main__":
    main()