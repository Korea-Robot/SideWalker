import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 시뮬레이션 환경 설정 ---
DT = 0.1  # 시간 간격
SIM_TIME = 100.0  # 시뮬레이션 시간 (충분히 길게 설정)
FIELD_SIZE = 20.0 # 시뮬레이션 필드 크기

# --- MPPI 컨트롤러 파라미터 ---
H = 20  # 예측 호라이즌 (Horizon)
N = 1000  # 샘플링 개수
LAMBDA = 1.0  # 가중치 계산 파라미터
SIGMA = np.array([[0.2, 0.0], [0.0, 0.2]])  # 제어 입력의 노이즈 공분산 (조금 더 탐색적으로)

# --- 로봇 및 환경 모델 ---
class OmniRobot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta])

    def update_state(self, u):
        """ 전방향 로봇의 동역학 모델 """
        self.state[0] += u[0] * DT
        self.state[1] += u[1] * DT
        return self.state

class DynamicObstacle:
    def __init__(self, x, y, vx, vy, size):
        self.pos = np.array([x, y])
        self.vel = np.array([vx, vy])
        self.size = size  # 장애물 크기 속성 추가

    def update_position(self):
        """ 장애물 위치 업데이트 및 필드 경계 처리 """
        self.pos += self.vel * DT
        # 필드 밖으로 나가면 방향을 반대로 바꿈
        if self.pos[0] < 0 or self.pos[0] > FIELD_SIZE:
            self.vel[0] *= -1
        if self.pos[1] < 0 or self.pos[1] > FIELD_SIZE:
            self.vel[1] *= -1
        return self.pos

# --- 비용 함수 정의 ---
def cost_function(predicted_trajectory, goal, obstacles):
    # 1. 목표 지점까지의 거리 비용
    goal_cost = np.linalg.norm(predicted_trajectory[:, :2] - goal, axis=1)

    # 2. 장애물과의 충돌 비용 (장애물 크기 고려)
    obstacle_cost = np.zeros(predicted_trajectory.shape[0])
    for obs in obstacles:
        # 장애물 중심까지의 거리에서 장애물 반경을 빼서 가장자리까지의 거리를 계산
        dist_to_obs_edge = np.linalg.norm(predicted_trajectory[:, :2] - obs.pos, axis=1) - obs.size
        # 장애물에 가까워질수록 비용 급증
        obstacle_cost += np.exp(-2.0 * dist_to_obs_edge)

    # 3. 제어 입력 비용 (너무 급격한 움직임 방지)
    # 이 예제에서는 생략하지만, 실제로는 제어 입력의 크기에 대한 비용을 추가하여
    # 더 부드러운 움직임을 유도할 수 있습니다.

    return goal_cost + 5.0 * obstacle_cost # 장애물 회피 가중치 증가

# --- MPPI 컨트롤러 ---
def mppi_controller(robot_state, goal, obstacles, prev_optimal_u):
    # 1. 무작위 제어 시퀀스 생성
    noise = np.random.multivariate_normal(np.zeros(2), SIGMA, size=(N, H))
    perturbed_u = prev_optimal_u + noise

    # 2. 궤적 예측
    predicted_trajectories = np.zeros((N, H, 3))
    current_state = np.tile(robot_state, (N, 1))

    for t in range(H):
        u_t = perturbed_u[:, t, :]
        current_state[:, 0] += u_t[:, 0] * DT
        current_state[:, 1] += u_t[:, 1] * DT
        predicted_trajectories[:, t, :] = current_state

    # 3. 비용 계산
    total_costs = np.zeros(N)
    for t in range(H):
        total_costs += cost_function(predicted_trajectories[:, t, :], goal, obstacles)

    # 4. 최적 제어 입력 계산 (가중 평균)
    weights = np.exp(-1.0 / LAMBDA * (total_costs - np.min(total_costs)))
    weights /= (np.sum(weights) + 1e-10) # 분모가 0이 되는 것을 방지

    optimal_u_sequence = np.sum(perturbed_u * weights[:, np.newaxis, np.newaxis], axis=0)

    # 다음 스텝을 위해 최적 제어 시퀀스 업데이트
    next_optimal_u = np.roll(optimal_u_sequence, -1, axis=0)
    next_optimal_u[-1] = np.zeros(2)

    return optimal_u_sequence[0], next_optimal_u

# --- 메인 시뮬레이션 루프 ---
def main():
    robot = OmniRobot(x=FIELD_SIZE/2, y=FIELD_SIZE/2, theta=0.0)
    goal = np.random.uniform(0, FIELD_SIZE, 2)
    
    # 다양한 크기와 속도를 가진 동적 장애물 생성
    obstacles = []
    for _ in range(8): # 장애물 개수 증가
        obstacles.append(DynamicObstacle(
            x=np.random.uniform(0, FIELD_SIZE),
            y=np.random.uniform(0, FIELD_SIZE),
            vx=np.random.uniform(-0.5, 0.5),
            vy=np.random.uniform(-0.5, 0.5),
            size=np.random.uniform(0.5, 2.5)  # 0.5 ~ 2.5 사이의 랜덤 크기
        ))

    # 이전 최적 제어 입력을 저장할 변수
    prev_optimal_u = np.zeros((H, 2))

    plt.ion()  # 대화형 모드 켜기
    fig, ax = plt.subplots(figsize=(10, 10))

    time = 0.0
    while time < SIM_TIME:
        ax.cla() # 이전 프레임 지우기

        # 동적 장애물 위치 업데이트
        for obs in obstacles:
            obs.update_position()

        # MPPI 컨트롤러로 최적 제어 입력 계산
        optimal_u, prev_optimal_u = mppi_controller(robot.state, goal, obstacles, prev_optimal_u)

        # 로봇 상태 업데이트
        robot.update_state(optimal_u)

        # 목표 도달 시 새로운 목표 설정
        if np.linalg.norm(robot.state[:2] - goal) < 1.5:
            print(f"시간 {time:.1f}s: 목표 도달! 새로운 목표를 설정합니다.")
            goal = np.random.uniform(0, FIELD_SIZE, 2)
            print(f"새로운 목표: [{goal[0]:.1f}, {goal[1]:.1f}]")


        # --- 결과 시각화 ---
        # 로봇 그리기
        ax.plot(robot.state[0], robot.state[1], "ob", markersize=10, label="Robot")
        # 목표 지점 그리기
        ax.plot(goal[0], goal[1], "xg", markersize=12, label="Goal")
        
        # 장애물 그리기 (크기 반영)
        for i, obs in enumerate(obstacles):
            obstacle_circle = patches.Circle(obs.pos, obs.size, color='k', fill=True, alpha=0.7)
            ax.add_patch(obstacle_circle)
            if i == 0: # 라벨은 한 번만 추가
                 ax.plot(obs.pos[0], obs.pos[1], 'sk', alpha=0.0, label="Obstacles")


        ax.set_title(f"MPPI Controller Simulation (Time: {time:.1f}s)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_xlim(0, FIELD_SIZE)
        ax.set_ylim(0, FIELD_SIZE)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        plt.pause(0.01)
        time += DT

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
