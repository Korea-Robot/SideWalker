import numpy as np
import torch

class RewardCalculator:
    @staticmethod
    def compute_reward(obs, action, next_obs, done, info, env):
        reward = 0.0
        
        # Get navigation object and goal position
        nav = env.agent.navigation
        goal_position = nav.checkpoints[-1]  # [x, y] coordinates
        
        # Get current vehicle position
        vehicle_pos = env.agent.position  # [x, y] coordinates
        
        # Calculate distance to goal
        goal_distance = np.linalg.norm(vehicle_pos - goal_position)
        
        # Distance-based reward (exponential decay)
        if goal_distance > 0:
            reward += 5.0 * np.exp(-goal_distance / 20.0)
        
        # Route completion reward (using navigation progress)
        route_progress = nav.route_completion
        reward += route_progress * 2.0  # Reward for making progress along route
        
        # Speed reward (encourage appropriate speed)
        speed = info.get('speed', 0)
        if speed > 0:
            # Reward optimal speed range (10-20 m/s)
            if 10 <= speed <= 20:
                reward += 0.8
            else:
                reward += max(0, 0.8 - abs(speed - 15) * 0.05)
        
        # Directional reward (encourage moving towards goal)
        if goal_distance > 2.0:  # Only when not too close
            velocity = info.get('velocity', np.array([0, 0]))
            if np.linalg.norm(velocity) > 0.1:
                # Calculate direction to goal
                goal_direction = (goal_position - vehicle_pos) / goal_distance
                velocity_norm = velocity / np.linalg.norm(velocity)
                
                # Reward alignment with goal direction
                alignment = np.dot(velocity_norm, goal_direction)
                reward += alignment * 0.6
        
        # Route following reward (staying on planned path)
        # Use reference trajectory to check if agent is following the path
        try:
            ref_trajectory = nav.reference_trajectory
            if hasattr(ref_trajectory, 'get_polyline'):
                # Find closest point on reference trajectory
                polyline = ref_trajectory.get_polyline()
                if len(polyline) > 0:
                    distances_to_path = [np.linalg.norm(vehicle_pos - np.array(pt)) for pt in polyline]
                    min_distance_to_path = min(distances_to_path)
                    
                    # Reward staying close to planned path
                    if min_distance_to_path < 5.0:
                        reward += (5.0 - min_distance_to_path) * 0.2
                    else:
                        reward -= min_distance_to_path * 0.1  # Penalty for deviating
        except:
            pass  # Fallback if reference trajectory not available
        
        # Penalty for crashes and violations
        if info.get('crash', False):
            reward -= 50.0
        if info.get('out_of_road', False):
            reward -= 25.0
        
        # Big reward for reaching destination
        if info.get('arrive_dest', False) or goal_distance < 3.0:
            reward += 200.0
        
        # Progress reward (compare with previous distance)
        if hasattr(env, '_prev_goal_distance'):
            prev_distance = env._prev_goal_distance
            if prev_distance > goal_distance:
                progress = prev_distance - goal_distance
                reward += progress * 8.0  # Reward for getting closer to goal
            elif prev_distance < goal_distance:
                # Small penalty for moving away from goal
                reward -= (goal_distance - prev_distance) * 2.0
        
        # Store current distance for next iteration
        env._prev_goal_distance = goal_distance
        
        # Efficiency reward (encourage reaching goal quickly)
        if goal_distance < 10.0:
            reward += (10.0 - goal_distance) * 0.5
        
        # Small time penalty to encourage efficiency
        reward -= 0.02
        
        # Smooth driving reward (penalize abrupt actions)
        if hasattr(env, '_prev_action'):
            prev_action = env._prev_action
            action_diff = np.linalg.norm(np.array(action) - np.array(prev_action))
            if action_diff > 0.5:
                reward -= action_diff * 0.3
        
        env._prev_action = action
        
        return reward

    @staticmethod
    def reset_tracking(env):
        """Call this method when environment is reset to clear tracking variables"""
        if hasattr(env, '_prev_goal_distance'):
            delattr(env, '_prev_goal_distance')
        if hasattr(env, '_prev_action'):
            delattr(env, '_prev_action')