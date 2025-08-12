# import numpy as np
# import matplotlib.pyplot as plt
# from metaurban import SidewalkStaticMetaUrbanEnv

# class KinematicBicycleModel:
#     """
#     차량의 움직임을 예측하기 위한 운동학적 자전거 모델입니다.
#     상태: [x, y, heading_theta]
#     액션: [steering, throttle] - 이 모델에서는 throttle을 무시하고 고정 속도를 사용합니다.
#     """
#     def __init__(self, wheelbase, max_steering_angle, velocity, dt):
#         """
#         모델 파라미터를 초기화합니다.
#         :param wheelbase: 차량의 축거 (m)
#         :param max_steering_angle: 최대 조향각 (라디안)
#         :param velocity: 차량의 고정 속도 (m/s)
#         :param dt: 시뮬레이션 시간 간격 (초)
#         """
#         self.L = wheelbase
#         self.delta_max = max_steering_angle
#         self.v = velocity
#         self.dt = dt
#         self.state = np.zeros(3)

#     def set_state(self, x, y, theta):
#         """모델의 현재 상태를 설정합니다."""
#         self.state = np.array([x, y, theta])

#     def step(self, action):
#         """
#         주어진 액션으로 다음 상태를 예측합니다.
#         :param action: [steering, throttle]
#         :return: 예측된 다음 상태 [x, y, theta]
#         """
#         # 액션 [-1, 1]을 실제 조향각(라디안)으로 변환
#         steer_action = action[0]
#         delta = steer_action * self.delta_max

#         # 현재 상태
#         x, y, theta = self.state

#         # 운동학 모델 방정식
#         x_next = x + self.v * np.cos(theta) * self.dt
#         y_next = y + self.v * np.sin(theta) * self.dt
#         theta_next = theta + (self.v / self.L) * np.tan(delta) * self.dt

#         # 모델 내부 상태 업데이트
#         self.state = np.array([x_next, y_next, theta_next])

#         return self.state

# def main():
#     # === 1. 모델 및 환경 설정 ===
#     # 모델 파라미터 (위에서 보정한 값 사용)
#     WHEELBASE = 2.7  # 축거 (m)
#     MAX_STEERING_ANGLE = 0.707  # 최대 조향각 (radians, 약 40.5도)
#     VELOCITY = 10.0  # 고정 속도 (m/s)
#     DT = 0.025  # 시간 간격 (초)

#     # MetaUrban 환경 설정
#     config = dict(
#         use_render=True, # 시각화 창을 띄우려면 True
#         manual_control=False,
#         map='X',
#         num_scenarios=1,
#         horizon=1000,
#         vehicle_config=dict(show_navi_mark=False),
#         window_size=(1200, 900),
#         object_density = 0.01
#     )
#     env = SidewalkStaticMetaUrbanEnv(config)

#     # 테스트할 조향 값 리스트 (우회전)
#     steering_values_to_test = [-1.0, -0.8, -0.6, -0.4, -0.2]
    
#     # 시뮬레이션 스텝 수 (4초간 시뮬레이션)
#     simulation_steps = 160 # 160 steps * 0.025 s/step = 4 seconds

#     # 결과를 저장할 딕셔너리
#     results = {}

#     # === 2. 각 조향 값에 대해 시뮬레이션 및 예측 수행 ===
#     try:
#         for steer_val in steering_values_to_test:
#             print(f"--- Testing Steering Value: {steer_val:.2f} ---")
            
#             # 환경 및 모델 리셋
#             o, _ = env.reset(seed=0)
#             model = KinematicBicycleModel(WHEELBASE, MAX_STEERING_ANGLE, VELOCITY, DT)
            
#             # 초기 상태를 동일하게 설정
#             initial_pos = env.agent.position
#             initial_theta = env.agent.heading_theta
#             model.set_state(initial_pos[0], initial_pos[1], initial_theta)
            
#             # 궤적을 저장할 리스트
#             ground_truth_traj = [initial_pos]
#             predicted_traj = [initial_pos]
            
#             # 고정된 액션
#             action = [steer_val, 1.0] # throttle=1.0

#             for _ in range(simulation_steps):
#                 # 실제 환경 스텝
#                 o, r, tm, tc, info = env.step(action)
#                 current_pos = env.agent.position
#                 ground_truth_traj.append(current_pos)
                
#                 # 모델 예측 스텝
#                 predicted_state = model.step(action)
#                 predicted_traj.append(predicted_state[:2]) # x, y 좌표만 저장
                
#                 env.render(text={"Steering": steer_val, "Model": "Active"})

#             results[steer_val] = {
#                 'ground_truth': np.array(ground_truth_traj),
#                 'predicted': np.array(predicted_traj)
#             }

#     finally:
#         env.close()

#     # === 3. 결과 시각화 ===
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 12))

#     colors = plt.cm.viridis(np.linspace(0, 1, len(steering_values_to_test)))

#     for i, steer_val in enumerate(steering_values_to_test):
#         gt = results[steer_val]['ground_truth']
#         pred = results[steer_val]['predicted']
        
#         # 실제 시뮬레이션 궤적 (실선)
#         ax.plot(gt[:, 0], gt[:, 1], color=colors[i], lw=3, 
#                 label=f'실제 궤적 (조향: {steer_val})')
        
#         # 모델 예측 궤적 (점선)
#         ax.plot(pred[:, 0], pred[:, 1], color=colors[i], lw=2, linestyle='--', 
#                 label=f'모델 예측 (조향: {steer_val})')

#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title("동역학 모델 검증: 실제 환경 vs. 모델 예측", fontsize=16)
#     ax.set_xlabel("X-좌표 (m)")
#     ax.set_ylabel("Y-좌표 (m)")
#     ax.legend(loc='best')
#     plt.savefig("dynamics_model_validation.png", dpi=300)
#     print("\nVerification plot saved to 'dynamics_model_validation.png'")

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from metaurban import SidewalkStaticMetaUrbanEnv

class KinematicBicycleModel:
    """
    차량의 움직임을 예측하기 위한 운동학적 자전거 모델입니다.
    상태: [x, y, heading_theta]
    액션: [steering, throttle] - 이 모델에서는 throttle을 무시하고 고정 속도를 사용합니다.
    """
    def __init__(self, wheelbase, max_steering_angle, velocity, dt):
        """
        모델 파라미터를 초기화합니다.
        :param wheelbase: 차량의 축거 (m)
        :param max_steering_angle: 최대 조향각 (라디안)
        :param velocity: 차량의 고정 속도 (m/s)
        :param dt: 시뮬레이션 시간 간격 (초)
        """
        self.L = wheelbase
        self.delta_max = max_steering_angle
        self.v = velocity
        self.dt = dt
        self.state = np.zeros(3)

    def set_state(self, x, y, theta):
        """모델의 현재 상태를 설정합니다."""
        self.state = np.array([x, y, theta])

    def step(self, action):
        """
        주어진 액션으로 다음 상태를 예측합니다.
        :param action: [steering, throttle]
        :return: 예측된 다음 상태 [x, y, theta]
        """
        # 액션 [-1, 1]을 실제 조향각(라디안)으로 변환
        steer_action = action[0]
        delta = steer_action * self.delta_max

        # 현재 상태
        x, y, theta = self.state

        # 운동학 모델 방정식
        x_next = x + self.v * np.cos(theta) * self.dt
        y_next = y + self.v * np.sin(theta) * self.dt
        theta_next = theta + (self.v / self.L) * np.tan(delta) * self.dt

        # 모델 내부 상태 업데이트
        self.state = np.array([x_next, y_next, theta_next])

        return self.state

def main():
    # === 1. 모델 및 환경 설정 ===
    # 모델 파라미터 (이전 분석에서 보정한 값 사용)
    WHEELBASE = 2.7  # 축거 (m)
    MAX_STEERING_ANGLE = 0.707  # 최대 조향각 (radians, 약 40.5도)
    VELOCITY = 10.0  # 고정 속도 (m/s)
    DT = 0.025  # 시간 간격 (초)

    # MetaUrban 환경 설정
    config = dict(
        use_render=True, # 시각화 창을 띄우려면 True
        manual_control=False,
        map='X',
        num_scenarios=1,
        horizon=1000,
        vehicle_config=dict(show_navi_mark=False),
        window_size=(1200, 900),
        object_density=0.05  # <<< 수정된 부분: 필수 설정값 추가
    )
    env = SidewalkStaticMetaUrbanEnv(config)

    # 테스트할 조향 값 리스트 (우회전)
    steering_values_to_test = [-1.0, -0.8, -0.6, -0.4, -0.2]
    
    # 시뮬레이션 스텝 수 (4초간 시뮬레이션)
    simulation_steps = 160 # 160 steps * 0.025 s/step = 4 seconds

    # 결과를 저장할 딕셔너리
    results = {}

    # === 2. 각 조향 값에 대해 시뮬레이션 및 예측 수행 ===
    try:
        for steer_val in steering_values_to_test:
            print(f"--- Testing Steering Value: {steer_val:.2f} ---")
            
            # 환경 및 모델 리셋
            o, _ = env.reset(seed=0)
            model = KinematicBicycleModel(WHEELBASE, MAX_STEERING_ANGLE, VELOCITY, DT)
            
            # 초기 상태를 동일하게 설정
            initial_pos = env.agent.position
            initial_theta = env.agent.heading_theta
            model.set_state(initial_pos[0], initial_pos[1], initial_theta)
            
            # 궤적을 저장할 리스트
            ground_truth_traj = [initial_pos]
            predicted_traj = [initial_pos]
            
            # 고정된 액션
            action = [steer_val, 1.0] # throttle=1.0

            for _ in range(simulation_steps):
                # 실제 환경 스텝
                o, r, tm, tc, info = env.step(action)
                current_pos = env.agent.position
                ground_truth_traj.append(current_pos)
                
                # 모델 예측 스텝
                predicted_state = model.step(action)
                predicted_traj.append(predicted_state[:2]) # x, y 좌표만 저장
                
                if env.config["use_render"]:
                    env.render(text={"Steering": steer_val, "Model": "Active"})

            results[steer_val] = {
                'ground_truth': np.array(ground_truth_traj),
                'predicted': np.array(predicted_traj)
            }

    finally:
        env.close()

    # === 3. 결과 시각화 ===
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = plt.cm.viridis(np.linspace(0, 1, len(steering_values_to_test)))

    for i, steer_val in enumerate(steering_values_to_test):
        gt = results[steer_val]['ground_truth']
        pred = results[steer_val]['predicted']
        
        # 실제 시뮬레이션 궤적 (실선)
        ax.plot(gt[:, 0], gt[:, 1], color=colors[i], lw=3, 
                label=f'실제 궤적 (조향: {steer_val})')
        
        # 모델 예측 궤적 (점선)
        ax.plot(pred[:, 0], pred[:, 1], color=colors[i], lw=2, linestyle='--', 
                label=f'모델 예측 (조향: {steer_val})')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("동역학 모델 검증: 실제 환경 vs. 모델 예측", fontsize=16)
    ax.set_xlabel("X-좌표 (m)")
    ax.set_ylabel("Y-좌표 (m)")
    ax.legend(loc='best')
    plt.savefig("dynamics_model_validation.png", dpi=300)
    print("\nVerification plot saved to 'dynamics_model_validation.png'")

if __name__ == "__main__":
    main()