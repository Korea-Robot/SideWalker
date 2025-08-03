import numpy as np
import os
import math
import time

from metaurban.envs import SidewalkStaticMetaUrbanEnv
# Change 1: Directly use LidarStateObservation instead of the mixed observation
from metaurban.obs.state_obs import LidarStateObservation


"""
LiDAR data

획득: obs[-mppi_controller.num_lidar_rays:] 코드를 통해 환경 관측값(obs) 벡터의 마지막 240개 요소를 가져옵니다.

구조: 이 240개의 숫자 배열은 자동차를 중심으로 360도를 240개의 부채꼴로 나눈 것입니다. 각 요소는 해당 방향으로의 장애물까지의 거리를 나타냅니다.

1.0: 해당 방향으로 최대 탐지 거리(50m)까지 장애물이 없다는 의미입니다 (안전).

0.0: 장애물이 차에 바로 붙어 있다는 의미입니다 (매우 위험).

0.5: 최대 거리의 절반(25m) 지점에 장애물이 있다는 의미입니다.



## 코드에서의 활용법: '위험도' 계산
코드는 이 240개의 거리 값을 '위험도'로 변환하여 사용합니다. 핵심은 _compute_cost 함수에 있습니다.

경로-라이다 매핑: 컨트롤러가 예측한 수천 개의 가상 경로(trajectories) 위를 지나는 각 지점이 어떤 방향에 해당하는지 계산합니다. 그리고 그 방향에 해당하는 라이다 값(거리 정보)을 가져옵니다.

위험도 변환 및 증폭: 아래 코드가 가장 중요한 부분입니다.


obstacle_proximity_cost = (1.0 - lidar_scan[indices]) ** 2
1.0 - lidar_scan: 이 연산을 통해 '거리' 정보를 '근접도' 또는 '위험도' 정보로 바꿉니다.

라이다 값이 1.0(안전)이면 위험도는 0.0이 됩니다.

라이다 값이 0.0(위험)이면 위험도는 1.0이 됩니다.

** 2 (제곱): 이 '위험도'를 제곱하는 것이 핵심입니다. 이 연산은 가까운 장애물에 대한 페널티를 기하급수적으로 증폭시키는 효과를 가져옵니다.

멀리 있는 장애물 (위험도 0.1) → 비용 0.01 (거의 무시)

가까이 있는 장애물 (위험도 0.9) → 비용 0.81 (매우 큰 페널티)

비용 누적: 이 계산된 위험도를 경로 전체에 대해 합산하여, 특정 경로가 얼마나 위험한지를 나타내는 최종 '장애물 비용'을 산출합니다.


"""
# --- MPPI Controller ---
class MPPIController:
    def __init__(self, env, horizon=15, num_samples=512, temperature=0.5):
        """
        MPPI Controller Initialization.
    
        :param env: The MetaUrban environment instance.
        :param horizon: int, The prediction horizon (number of steps to look ahead).
        :param num_samples: int, The number of trajectories to sample at each step.
        :param temperature: float, A parameter to control the "softness" of the weighted average.
        """
        self.agent = env.agent
        self.H = horizon
        self.K = num_samples
        self.lambda_ = temperature

        # Vehicle dynamics parameters
        self.L = self.agent.LENGTH  # Vehicle wheelbase
        # NEW CORRECTED LINE
        self.dt = env.config["decision_repeat"] * env.config["physics_world_step_size"] # Timestep duration

        # Action limits
        self.max_steer = 1
        self.max_accel = 1
        self.max_brake = 1

        # Initialize nominal control sequence (e.g., go straight with no acceleration)
        self.nominal_actions = np.zeros((self.H, 2))  # [steering, acceleration]

        # Noise for sampling
        self.noise_std = np.array([0.5, 1.0]) # Std dev for steering and acceleration noise

        # Cost function weights
        self.W_GOAL = 30.0      # Weight for reaching the goal
        self.W_OBSTACLE = 1.0  # Weight for avoiding obstacles
        self.W_CONTROL = 0.1    # Weight for smooth control

        self.lidar_max_dist = self.agent.config["lidar"]["distance"]
        self.num_lidar_rays = self.agent.config["lidar"]["num_lasers"]


    def _dynamics_model(self, state, action):
        """
        A simple kinematic bicycle model for predicting the next state.
        
        :param state: [x, y, heading_theta, speed_ms]
        :param action: [steering, acceleration_m/s^2]
        :return: next_state
        """
        x, y, theta, speed_ms = state
        steer, accel = action

        # Unnormalize action
        steer = steer * self.max_steer
        
        # Apply physics
        next_speed_ms = speed_ms + accel * self.dt
        next_speed_ms = np.clip(next_speed_ms, 0, self.agent.max_speed_km_h / 3.6)
        
        # Distance traveled
        dist = next_speed_ms * self.dt
        
        # --- THIS IS THE CORRECTED PART ---
        # Use np.where for vectorized conditional logic.
        # This avoids the "ambiguous truth value" error.
        # Condition: Check where speed is non-negligible.
        condition = abs(next_speed_ms) >= 1e-3
        
        # If condition is True, calculate heading change.
        delta_theta = (next_speed_ms / self.L) * np.tan(steer) * self.dt
        
        # If condition is False, heading change is 0.
        # np.where selects from delta_theta or 0.0 based on the condition for each element.
        next_theta = theta + np.where(condition, delta_theta, 0.0)
        # --- END OF CORRECTION ---
        
        next_x = x + dist * np.cos(next_theta)
        next_y = y + dist * np.sin(next_theta)
        
        return np.array([next_x, next_y, next_theta, next_speed_ms])

    def _compute_cost(self, trajectories, target_pos, lidar_scan):
        """
        Computes the cost for all sampled trajectories.
        
        :param trajectories: A tensor of shape (K, H, 4) containing K simulated state sequences.
        :param target_pos: The egocentric coordinates [x, y] of the target waypoint.
        :param lidar_scan: The 240-dimensional Lidar scan data.
        :return: An array of costs for each trajectory.
        """
        K, H, _ = trajectories.shape
        costs = np.zeros(K)

        # --- 1. Goal Cost ---
        # Penalize distance from the final predicted position to the target
        final_positions = trajectories[:, -1, :2] # Shape (K, 2)
        dist_to_goal = np.linalg.norm(final_positions - target_pos, axis=1)
        costs += self.W_GOAL * dist_to_goal

        # --- 2. Obstacle Cost ---
        # Penalize trajectories that get close to Lidar-detected obstacles
        traj_positions = trajectories[:, :, :2] # Shape (K, H, 2)
        
        # Lidar rays are clockwise from front (0 deg).
        # We need to map trajectory points (in egocentric x,y) to lidar ray indices.
        angles = np.arctan2(traj_positions[:, :, 1], traj_positions[:, :, 0]) # Y is forward, X is right
        angles_deg = np.rad2deg(angles)
        
        # Convert angle (-180 to 180) to lidar index (0 to 239)
        # Lidar Index 0 is front, 60 is right (-90 deg), 180 is left (90 deg)
        lidar_indices = (240 - (angles_deg + 360) % 360) / 1.5
        lidar_indices = np.floor(lidar_indices).astype(int) % self.num_lidar_rays

        # Get the Lidar distances for the corresponding angles
        obstacle_distances = lidar_scan[lidar_indices] * self.lidar_max_dist
        
        # Calculate distance of each point in the trajectory from the origin (ego vehicle)
        point_distances = np.linalg.norm(traj_positions, axis=2)
        
        # A collision is imminent if a predicted point is further than the obstacle detected in that direction
        # Add a cost that is high for close obstacles
        is_collision = (point_distances > obstacle_distances)
        
        # We can make the cost proportional to how close the obstacle is
        # (1.0 - lidar_scan) is 0 for no obstacle, 1 for obstacle at point blank
        obstacle_cost_factor = (1.0 - lidar_scan[lidar_indices])
        
        # Sum cost over the horizon for each sample
        total_obstacle_cost = np.sum(is_collision * obstacle_cost_factor, axis=1)
        costs += self.W_OBSTACLE * total_obstacle_cost
        
        return costs

    def update(self, target_pos, lidar_scan):
        """
        The main control loop for the MPPI controller.
        
        :param target_pos: The egocentric coordinates [x, y] of the target waypoint.
        :param lidar_scan: The current Lidar observation.
        :return: The optimal action [steering, throttle/brake].
        """
        K = self.K
        H = self.H

        # 1. Sample random action sequences
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=(K, H, 2))
        sampled_actions = self.nominal_actions + noise
        
        # Clip actions to be within valid range [-1, 1]
        sampled_actions[:, :, 0] = np.clip(sampled_actions[:, :, 0], -1.0, 1.0) # Steering
        sampled_actions[:, :, 1] = np.clip(sampled_actions[:, :, 1], -1.0, 1.0) # Acceleration/Brake

        # 2. Roll out trajectories using the dynamics model
        # All trajectories are simulated from the vehicle's current state in an egocentric frame
        # So the initial state for all simulations is [0, 0, 0, current_speed]
        current_speed_ms = self.agent.speed_km_h / 3.6
        initial_state = np.array([0, 0, 0, current_speed_ms])
        
        trajectories = np.zeros((K, H, 4)) # (x, y, theta, speed)
        current_states = np.tile(initial_state, (K, 1))

        # Convert throttle/brake action to raw acceleration
        accel_actions = np.zeros_like(sampled_actions[:, :, 1])
        accel_actions[sampled_actions[:, :, 1] > 0] = sampled_actions[:, :, 1][sampled_actions[:, :, 1] > 0] * self.max_accel
        accel_actions[sampled_actions[:, :, 1] < 0] = sampled_actions[:, :, 1][sampled_actions[:, :, 1] < 0] * self.max_brake
        
        sim_actions = np.stack([sampled_actions[:, :, 0], accel_actions], axis=-1)

        for t in range(H):
            current_states = self._dynamics_model(current_states.T, sim_actions[:, t, :].T).T
            trajectories[:, t, :] = current_states

        # 3. Compute costs for all trajectories
        costs = self._compute_cost(trajectories, target_pos, lidar_scan)

        # 4. Compute weights and find the optimal action
        weights = np.exp(-1.0 / self.lambda_ * (costs - np.min(costs)))
        weights /= np.sum(weights)

        # Weighted average of the first action of each sequence
        optimal_action = np.sum(weights[:, np.newaxis] * sampled_actions[:, 0, :], axis=0)

        # 5. Update nominal actions for the next step (warm start)
        self.nominal_actions = np.roll(self.nominal_actions, -1, axis=0)
        self.nominal_actions[-1] = optimal_action # Use the new best action as the last nominal action
        
        return optimal_action

# --- 설정 ---

# Change 2: Simplified environment configuration to focus on state/Lidar data
BASE_ENV_CFG = dict(
    use_render=True,
    map='X',
    manual_control=False, # We are using the MPPI controller
    crswalk_density=0.2,
    object_density=0.2, # Added some objects for the Lidar to see
    drivable_area_extension=55,
    horizon=1000,
    vehicle_config=dict(
        enable_reverse=False,
        # Use LidarStateObservation directly
        show_lidar=True, # Visualize the lidar
        lidar=dict(num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0),
    ),
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    num_scenarios=100000,
    accident_prob=0.0,
    # Change 3: Set the observation type to LidarStateObservation
    agent_observation=LidarStateObservation,
    image_observation=False, # Disable image observation for performance
    log_level=50,
)

# --- 유틸리티 함수 ---

def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    vec_in_world = global_target_pos - agent_pos
    theta = agent_heading # In MPPI we want to align with the agent's forward direction (y)
    
    # We want to rotate so the agent's heading is aligned with the new Y-axis
    # Standard rotation is counter-clockwise. agent_heading is also CCW from positive X-axis.
    # To align world to ego, we must rotate clockwise by agent_heading.
    # sin(-theta) = -sin(theta), cos(-theta) = cos(theta)
    cos_h = np.cos(-theta)
    sin_h = np.sin(-theta)
    
    # Ego frame: Y is forward, X is right
    ego_y = vec_in_world[0] * cos_h - vec_in_world[1] * sin_h
    ego_x = vec_in_world[0] * sin_h + vec_in_world[1] * cos_h
    return np.array([ego_x, ego_y])


# --- 메인 실행 로직 ---

env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)

import random 

running = True
try:
    for i in range(10):
        # Change 4: obs is now a numpy array, not a dict
        obs, info = env.reset(seed=i + 50)
        
        # Instantiate the MPPI controller after reset to link it to the new agent
        mppi_controller = MPPIController(env)
        
        waypoints = env.agent.navigation.checkpoints
        num_waypoints = len(waypoints)
        k = 5 # Target the 5th waypoint ahead


        waypoints = env.agent.navigation.checkpoints 
        print('wayppoint num: ',len(waypoints))
        
        reset = i+2000 
        while len(waypoints)<30:
            obs,info = env.reset(seed= reset)
            reset = random.randint(1,40000)
            waypoints = env.agent.navigation.checkpoints 
            print('i do not have sufficient waypoints ',i,' th')
            print(len(waypoints))
            
        num_waypoints = len(waypoints)

        
        
        while running:
            # Get target waypoint
            global_target = waypoints[min(k, num_waypoints - 1)]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            
            # Convert target to egocentric coordinates for the controller
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
            
            # Extract Lidar data from the observation vector
            # As per LidarStateObservation, the last 240 elements are the lidar points
            lidar_scan = obs[-mppi_controller.num_lidar_rays:]
            
            # Get action from MPPI controller
            action = mppi_controller.update(ego_goal_position, lidar_scan)
            
            # Update target waypoint if we get close
            distance_to_target = np.linalg.norm(ego_goal_position)
            if distance_to_target < 5.0 and k < num_waypoints - 1:
                k += 1

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Environment rendering
            env.render(
                text={
                    "Ego Goal (local)": np.round(ego_goal_position, 2),
                    "Action": np.round(action, 2),
                    "Target Waypoint": f"{k}/{num_waypoints}",
                    "Reward": f"{reward:.2f}",
                }
            )


            # if terminated or truncated:
            #     print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
            #     break
                

finally:
    env.close()