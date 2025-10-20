import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 시뮬레이션 환경 설정 ---
DT = 0.1  # 시간 간격 (델타 t, Δt)
SIM_TIME = 100.0  # 총 시뮬레이션 시간
FIELD_SIZE = 15.0 # 시뮬레이션 필드 크기

# --- MPPI 컨트롤러 파라미터 ---
H = 25  # 예측 호라이즌 (Horizon): 미래를 몇 스텝까지 예측할지 결정
N = 1000  # 샘플링 개수: 매 스텝마다 생성할 무작위 궤적의 수
# Monte Carlo를 위해 생성할 궤적의 수 

LAMBDA = 3.0  # 온도 파라미터 (λ): 가중치 계산 시 사용. 클수록 탐험적, 작을수록 이용적.

# 제어 입력의 노이즈 공분산 (Σ): 무작위 제어 입력을 생성할 때 사용되는 정규분포의 공분산
SIGMA = np.array([[0.2, 0.0], [0.0, 0.2]])

# --- 로봇 및 환경 모델 ---
class OmniRobot:
    """ 전방향 로봇의 상태와 동역학 모델을 정의하는 클래스 """
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta]) # 로봇의 상태 [px, py, theta]
        # 2차원에서 x,y, heading

    def update_state(self, u):
        """
        로봇의 동역학 모델. 제어 입력 u를 받아 다음 상태를 계산.
        수학식: x_{t+1} = x_t + u_t * Δt
        """
        self.state[0] += u[0] * DT
        self.state[1] += u[1] * DT
        return self.state

class DynamicObstacle:
    """ 움직이는 장애물의 상태와 모델을 정의하는 클래스 """
    def __init__(self, x, y, vx, vy, size):
        self.pos = np.array([x, y]) # 장애물 위치
        self.vel = np.array([vx, vy]) # 장애물 속도
        self.size = size  # 장애물 크기 (반지름)

    def update_position(self):
        """ 장애물 위치를 속도와 시간 간격에 따라 업데이트 """
        self.pos += self.vel * DT
        # 간단한 경계 처리: 필드 밖으로 나가면 방향 전환
        if self.pos[0] < 0 or self.pos[0] > FIELD_SIZE: self.vel[0] *= -1
        if self.pos[1] < 0 or self.pos[1] > FIELD_SIZE: self.vel[1] *= -1
        return self.pos

# --- 비용 함수 정의 ---
def cost_function(predicted_trajectory, goal, obstacles):
    """
    예측된 궤적에 대한 비용을 계산하는 함수.
    수학식: c(x) = c_goal(x) + w_obs * c_obs(x)
    """
    # 1. 목표 비용: 목표 지점까지의 유클리드 거리
    # c_goal(x) = || p - p_goal ||
    # H개의 horizon의 개수만큼 계산되는것인가?
    goal_cost = np.linalg.norm(predicted_trajectory[:, :2] - goal, axis=1)

    # 2. 장애물 비용: 각 장애물과의 거리가 가까워질수록 비용이 기하급수적으로 증가
    # 장애물의 반지름 안에 있으면 cost가 지수적으로 커짐. 반대로 밖에 있으면 작아짐.
    # c_obs(x) = Σ exp(-β * (||p - p_obs,i|| - r_i))
    obstacle_cost = np.zeros(predicted_trajectory.shape[0])
    for obs in obstacles:
        dist_to_obs_edge = np.linalg.norm(predicted_trajectory[:, :2] - obs.pos, axis=1) - obs.size
        obstacle_cost += np.exp(-2.0 * dist_to_obs_edge) # β=2.0으로 설정

    # 총 비용 반환 (장애물 비용에 가중치 5.0 부여)
    return goal_cost + 5.0 * obstacle_cost

# --- MPPI 컨트롤러 ---
def mppi_controller(robot_state, goal, obstacles, prev_optimal_u):
    """ MPPI 알고리즘의 메인 로직 """
    # 1. 무작위 제어 시퀀스 생성 (Perturbation)
    # 이전 최적 시퀀스에 가우시안 노이즈를 추가하여 N개의 후보 시퀀스 생성
    # V_i = U_{t-1}^* + ε_i,  ε_i ~ N(0, Σ)
    
    # N : sampling 개수
    # H : horizon 개수 
    
    # input이 들어갔을때 환경이나 상황에 따라 생기는 노이즈
    noise = np.random.multivariate_normal(np.zeros(2), SIGMA, size=(N, H))
    perturbed_u = prev_optimal_u + noise # (N, H, 2) 크기의 텐서

    # 2. 궤적 예측 (Rollout)
    # 생성된 N개의 제어 시퀀스를 각각 적용하여 미래 궤적들을 시뮬레이션
    predicted_trajectories = np.zeros((N, H, 3))
    current_state = np.tile(robot_state, (N, 1))
    for t in range(H):
        
        # 실제 물리세계에 적용되는 input 
        u_t = perturbed_u[:, t, :]
        
        # 실제 환경에서 일어나는 state transition 
        current_state[:, 0] += u_t[:, 0] * DT
        current_state[:, 1] += u_t[:, 1] * DT
        
        # 예측된 trajectory.
        predicted_trajectories[:, t, :] = current_state

    
    # 3. 비용 계산 및 가중치 부여 (Evaluation & Weighting)
    # N 개의 샘플에 대한 cost 계산.
    # 각 궤적의 총비용 S(V_i) 계산
    total_costs = np.zeros(N)
    for t in range(H):
        total_costs += cost_function(predicted_trajectories[:, t, :], goal, obstacles)

    # 각 궤적의 비용을 바탕으로 가중치 ω_i 계산
    # ω_i = exp(-1/λ * S(V_i)) / Σ exp(-1/λ * S(V_j))
    weights = np.exp(-1.0 / LAMBDA * (total_costs - np.min(total_costs)))
    weights /= (np.sum(weights) + 1e-10) # 정규화 (Normalization)

    # 4. 최적 제어 입력 계산 (Update)
    # 모든 후보 시퀀스를 가중 평균하여 최종 최적 제어 시퀀스 U_t^* 계산
    # U_t^* = Σ ω_i * V_i
    optimal_u_sequence = np.sum(perturbed_u * weights[:, np.newaxis, np.newaxis], axis=0)

    # 다음 스텝에서 사용할 이전 최적 시퀀스를 업데이트 (Shift)
    next_optimal_u = np.roll(optimal_u_sequence, -1, axis=0)
    next_optimal_u[-1] = np.zeros(2)

    # 로봇에 적용할 첫 번째 제어 입력(u_0^*)과 다음 계산에 사용할 정보들을 반환
    return optimal_u_sequence[0], next_optimal_u, predicted_trajectories, optimal_u_sequence

# --- 메인 시뮬레이션 루프 ---
def main():
    # ... (시뮬레이션 초기 설정 코드는 이전과 동일) ...
    robot = OmniRobot(x=FIELD_SIZE/2, y=FIELD_SIZE/2, theta=0.0)
    goal = np.random.uniform(0, FIELD_SIZE, 2)
    obstacles = [DynamicObstacle(x=np.random.uniform(0, FIELD_SIZE), y=np.random.uniform(0, FIELD_SIZE),
                                 vx=np.random.uniform(-0.5, 0.5), vy=np.random.uniform(-0.5, 0.5),
                                 size=np.random.uniform(0.5, 2.5)) for _ in range(8)]
    prev_optimal_u = np.zeros((H, 2))
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))

    time = 0.0
    while time < SIM_TIME:
        ax.cla()

        for obs in obstacles:
            obs.update_position()

        optimal_u, prev_optimal_u, sampled_trajs, optimal_u_seq = mppi_controller(
            robot.state, goal, obstacles, prev_optimal_u
        )
        robot.update_state(optimal_u)

        if np.linalg.norm(robot.state[:2] - goal) < 1.5:
            goal = np.random.uniform(0, FIELD_SIZE, 2)
        
        # --- 시각화 ---
        # 1. 샘플링된 궤적들 (후보 경로)
        for i in range(0, N, 20):
            ax.plot(sampled_trajs[i, :, 0], sampled_trajs[i, :, 1], color='gray', alpha=0.25, zorder=1)
        
        # 2. 최종 선택된 최적 궤적
        optimal_path = np.zeros((H + 1, 2))
        optimal_path[0, :] = robot.state[:2]
        temp_state = robot.state.copy()
        for t in range(H):
            temp_state[0] += optimal_u_seq[t, 0] * DT
            temp_state[1] += optimal_u_seq[t, 1] * DT
            optimal_path[t+1, :] = temp_state[:2]
        ax.plot(optimal_path[:, 0], optimal_path[:, 1], color='r', linewidth=3.0, label='Optimal Trajectory', zorder=3)
        
        # 로봇, 목표, 장애물
        ax.plot(robot.state[0], robot.state[1], "ob", markersize=12, label="Robot", zorder=4)
        ax.plot(goal[0], goal[1], "xg", markersize=15, markeredgewidth=3, label="Goal", zorder=4)
        for i, obs in enumerate(obstacles):
            obstacle_circle = patches.Circle(obs.pos, obs.size, color='k', fill=True, alpha=0.8, zorder=2)
            ax.add_patch(obstacle_circle)
        
        # 그래프 설정
        ax.set_title(f"MPPI Controller with Visualization (Time: {time:.1f}s)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_xlim(0, FIELD_SIZE)
        ax.set_ylim(0, FIELD_SIZE)
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        plt.pause(0.01)
        time += DT

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
