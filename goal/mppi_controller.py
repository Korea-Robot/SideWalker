import numpy as np
import time

class MPPI:
    def __init__(self, alpha=1.0, beta=1.0, dt=0.1, lambda_=1.0):
        """
        MPPI 컨트롤러 초기화
        
        Args:
            alpha: 가속도 제어 게인
            beta: 조향 제어 게인  
            dt: 시간 스텝
            lambda_: 온도 매개변수 (작을수록 더 탐욕적)
        """
        self.alpha = alpha
        self.beta = beta 
        self.dt = dt
        self.lambda_ = lambda_
        
        # MPPI 파라미터
        self.N = 1000  # 샘플링할 궤적 수
        self.H = 15    # 예측 지평선 (시간 스텝)
        
        # 제어 입력 제한
        self.u_min = np.array([-1.0, -1.0])  # [steering, throttle] 최소값
        self.u_max = np.array([1.0, 1.0])    # [steering, throttle] 최대값
        
        # 노이즈 공분산
        self.sigma = np.array([[0.3, 0.0], [0.0, 0.2]])  # [steering_noise, throttle_noise]
        
        # 이전 제어 시퀀스 저장 (warm start)
        self.prev_u_seq = np.zeros((self.H, 2))
    
    def dynamics_model(self, state, action):
        """
        차량 동역학 모델 (bicycle model 근사)
        
        Args:
            state: [x, y, theta, velocity, angular_velocity]
            action: [steering, throttle]
            
        Returns:
            next_state: 다음 상태
        """
        x, y, theta, velocity, angular_vel = state
        steering, throttle = action
        
        # 차량 동역학 업데이트
        next_x = x + np.cos(theta) * velocity * self.dt
        next_y = y + np.sin(theta) * velocity * self.dt
        next_theta = theta + angular_vel * self.dt
        next_velocity = velocity + self.alpha * throttle * self.dt
        next_angular_vel = angular_vel + self.beta * steering * self.dt
        
        # 속도와 각속도 제한
        next_velocity = np.clip(next_velocity, -5.0, 10.0)
        next_angular_vel = np.clip(next_angular_vel, -2.0, 2.0)
        
        return np.array([next_x, next_y, next_theta, next_velocity, next_angular_vel])
    
    def rollout_dynamics(self, initial_state, u_sequences):
        """
        N개의 제어 시퀀스에 대해 H 스텝 동안 궤적을 시뮬레이션
        
        Args:
            initial_state: 초기 상태 [x, y, theta, velocity, angular_velocity]
            u_sequences: (N, H, 2) 제어 시퀀스들
            
        Returns:
            trajectories: (N, H+1, 5) 궤적들
        """
        N = u_sequences.shape[0]
        trajectories = np.zeros((N, self.H + 1, 5))
        
        # 모든 궤적의 초기 상태 설정
        trajectories[:, 0, :] = initial_state
        
        # 각 시간 스텝에 대해 동역학 시뮬레이션
        for t in range(self.H):
            for n in range(N):
                trajectories[n, t+1, :] = self.dynamics_model(
                    trajectories[n, t, :], 
                    u_sequences[n, t, :]
                )
        
        return trajectories
    
    def cost_function(self, trajectories, lidar_data, goal_position):
        """
        궤적들에 대한 비용 계산
        
        Args:
            trajectories: (N, H+1, 5) 궤적들
            lidar_data: (240,) 라이다 데이터
            goal_position: [x, y] 목표 위치 (ego 좌표계)
            
        Returns:
            costs: (N,) 각 궤적의 비용
        """
        N = trajectories.shape[0]
        costs = np.zeros(N)
        
        for n in range(N):
            traj = trajectories[n]
            cost = 0.0
            
            for t in range(1, self.H + 1):
                x, y, theta, vel, angular_vel = traj[t]
                
                # 1. 목표 추적 비용 (가장 중요)
                goal_distance = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)
                cost += 10.0 * goal_distance
                
                # 2. 전진 장려 비용
                cost -= 2.0 * vel if vel > 0 else 0.0
                
                # 3. 장애물 회피 비용 (간단한 근사)
                # 라이다 데이터를 이용한 충돌 위험도 계산
                collision_risk = self.calculate_collision_risk(x, y, theta, lidar_data)
                cost += 50.0 * collision_risk
                
                # 4. 제어 입력 페널티 (부드러운 주행)
                if t < self.H:
                    steering = trajectories[n, t, 4] if t > 0 else 0  # angular_vel을 steering 근사로 사용
                    cost += 0.1 * steering**2
                
                # 5. 차선 중앙 유지 비용 (y=0 근처 유지)
                cost += 5.0 * y**2
            
            costs[n] = cost
        
        return costs
    
    def calculate_collision_risk(self, x, y, theta, lidar_data):
        """
        라이다 데이터를 이용한 충돌 위험도 계산
        
        Args:
            x, y, theta: 차량 위치와 방향
            lidar_data: (240,) 라이다 거리 데이터
            
        Returns:
            risk: 충돌 위험도 (0~1)
        """
        # 차량 전방과 측면의 라이다 빔들만 고려
        front_indices = list(range(110, 130))  # 전방 20도 범위
        left_indices = list(range(90, 110))    # 좌측
        right_indices = list(range(130, 150))  # 우측
        
        # 예상 차량 위치에서의 장애물까지 거리 추정
        min_safe_distance = 3.0
        risk = 0.0
        
        # 전방 위험도 (가중치 높음)
        front_distances = lidar_data[front_indices]
        front_risk = np.sum(front_distances < min_safe_distance) / len(front_indices)
        risk += 3.0 * front_risk
        
        # 측면 위험도
        left_distances = lidar_data[left_indices]
        right_distances = lidar_data[right_indices]
        
        left_risk = np.sum(left_distances < min_safe_distance/2) / len(left_indices)
        right_risk = np.sum(right_distances < min_safe_distance/2) / len(right_indices)
        
        risk += 1.0 * (left_risk + right_risk)
        
        return np.clip(risk, 0.0, 1.0)
    
    def sample_control_sequences(self):
        """
        제어 시퀀스들을 샘플링 (가우시안 노이즈 + warm start)
        
        Returns:
            u_sequences: (N, H, 2) 제어 시퀀스들
        """
        u_sequences = np.zeros((self.N, self.H, 2))
        
        for n in range(self.N):
            for t in range(self.H):
                # 이전 해에 노이즈를 추가 (warm start)
                if t < len(self.prev_u_seq):
                    base_u = self.prev_u_seq[t]
                else:
                    base_u = np.array([0.0, 0.3])  # 기본값: 직진, 적당한 속도
                
                # 가우시안 노이즈 추가
                noise = np.random.multivariate_normal([0, 0], self.sigma)
                u_sequences[n, t] = np.clip(base_u + noise, self.u_min, self.u_max)
        
        return u_sequences
    
    def path_integral_update(self, costs, u_sequences):
        """
        Path Integral을 이용한 제어 시퀀스 업데이트
        
        Args:
            costs: (N,) 각 궤적의 비용
            u_sequences: (N, H, 2) 제어 시퀀스들
            
        Returns:
            optimal_u_seq: (H, 2) 최적 제어 시퀀스
        """
        # 비용을 음수로 변환하고 정규화 (낮은 비용 = 높은 가중치)
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        weights = weights / np.sum(weights)
        
        # 가중 평균으로 최적 제어 시퀀스 계산
        optimal_u_seq = np.zeros((self.H, 2))
        for t in range(self.H):
            optimal_u_seq[t] = np.sum(weights[:, np.newaxis] * u_sequences[:, t, :], axis=0)
        
        return optimal_u_seq
    
    def optimize(self, current_state, lidar_data, goal_position):
        """
        MPPI 최적화 수행
        
        Args:
            current_state: [x, y, theta, velocity, angular_velocity]
            lidar_data: (240,) 라이다 데이터
            goal_position: [x, y] 목표 위치 (ego 좌표계)
            
        Returns:
            optimal_action: [steering, throttle] 최적 제어 입력
            proximal_goal: [x, y] 단기 목표 위치
        """
        # 1. 제어 시퀀스 샘플링
        u_sequences = self.sample_control_sequences()
        
        # 2. 동역학 시뮬레이션
        trajectories = self.rollout_dynamics(current_state, u_sequences)
        
        # 3. 비용 계산
        costs = self.cost_function(trajectories, lidar_data, goal_position)
        
        # 4. Path Integral 업데이트
        optimal_u_seq = self.path_integral_update(costs, u_sequences)
        
        # 5. 다음 반복을 위해 최적 시퀀스 저장 (shift + padding)
        self.prev_u_seq[:-1] = optimal_u_seq[1:]
        self.prev_u_seq[-1] = optimal_u_seq[-1]  # 마지막 제어 입력으로 패딩
        
        # 6. 최적 궤적에서 단기 목표 위치 추출 (k번째 시점)
        best_traj_idx = np.argmin(costs)
        best_trajectory = trajectories[best_traj_idx]
        
        k = min(5, self.H)  # 5 스텝 앞 또는 지평선 끝
        proximal_goal = best_trajectory[k, :2]  # [x, y]만 추출
        
        return optimal_u_seq[0], proximal_goal