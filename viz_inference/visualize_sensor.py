# 파일명: visualize_sensors.py
# 설명: MetaUrban 환경의 센서(RGB, Depth 등) 데이터를 시각화합니다.
# 실행 방법: python visualize_sensors.py

import matplotlib.pyplot as plt
import numpy as np
import os
from metaurban.envs import SidewalkStaticMetaUrbanEnv

# 핵심 컴포넌트 임포트
from core_components import (
    BASE_ENV_CFG, SENSOR_SIZE, convert_to_egocentric, extract_sensor_data
)

def visualize_sensor_data(env, num_steps=10, save_images=True):
    """센서 데이터 시각화 함수"""
    obs, info = env.reset()
    print("=== 센서 데이터 시각화 시작 ===")
    
    for step in range(num_steps):
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        print(f"\n--- Step {step} ---")
        if rgb_data is not None:
            print(f"RGB shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
        
        # 목표 지점 계산
        ego_goal_position = np.array([0.0, 0.0])
        if hasattr(env.agent, 'navigation') and env.agent.navigation:
            waypoints = env.agent.navigation.checkpoints
            k = min(15, len(waypoints) - 1) if waypoints else 0
            if len(waypoints) > k:
                ego_goal_position = convert_to_egocentric(waypoints[k], env.agent.position, env.agent.heading_theta)
        
        # 시각화
        if rgb_data is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'MetaUrban Sensor Data - Step {step}', fontsize=14)
            
            axes[0].imshow(rgb_data)
            axes[0].set_title('RGB Camera')
            axes[0].axis('off')
            
            info_text = f"""Agent Info:
Position: ({env.agent.position[0]:.2f}, {env.agent.position[1]:.2f})
Heading: {env.agent.heading_theta:.3f} rad
Goal (ego): ({ego_goal_position[0]:.2f}, {ego_goal_position[1]:.2f})

Sensor Info:
RGB: {rgb_data.shape}"""
            
            axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            axes[1].axis('off')
            
            plt.tight_layout()
            
            if save_images:
                os.makedirs('sensor_visualization', exist_ok=True)
                plt.savefig(f'sensor_visualization/step_{step:03d}.png')
                print(f"Saved: sensor_visualization/step_{step:03d}.png")
            plt.show()

        action = [0.5, 0.1] # 약간 직진
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
    try:
        visualize_sensor_data(env, num_steps=5)
    finally:
        env.close()