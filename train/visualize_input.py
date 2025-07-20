import matplotlib.pyplot as plt
import numpy as np
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# --- 기본 설정 ---
SENSOR_SIZE = (256, 160)
ENV_CFG = dict(
    use_render=False,
    map='S',
    # 'object_density' 키 추가 (오류 해결)
    object_density=0.1, 
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# --- 시각화 함수 ---
def visualize_stacked_obs(obs: dict):
    """
    스태킹된 관찰(obs) 데이터를 시각화합니다.
    RGB, Depth, Semantic 각 센서에 대해 3개의 프레임(t, t-1, t-2)을 보여줍니다.
    """
    # 데이터 추출 및 정규화
    rgb_images = [(obs['image'][..., i] * 255).astype(np.uint8) for i in range(3)]
    depth_images = [obs['depth'][..., 0, i] for i in range(3)]
    semantic_images = [(obs['semantic'][..., i] * 255).astype(np.uint8) for i in range(3)]
    
    # 3x3 그리드에 이미지 플로팅
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Stacked Sensor Observations (Initial State)", fontsize=16)
    
    frame_labels = ["Frame (t-2)", "Frame (t-1)", "Frame (t)"]
    
    for i in range(3):
        # RGB 이미지
        axes[0, i].imshow(rgb_images[i])
        axes[0, i].set_title(f"RGB - {frame_labels[i]}")
        axes[0, i].axis('off')
        
        # Depth 이미지
        axes[1, i].imshow(depth_images[i], cmap='gray')
        axes[1, i].set_title(f"Depth - {frame_labels[i]}")
        axes[1, i].axis('off')

        # Semantic 이미지
        axes[2, i].imshow(semantic_images[i])
        axes[2, i].set_title(f"Semantic - {frame_labels[i]}")
        axes[2, i].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig('visualize_input')

# --- 메인 실행 코드 ---
if __name__ == "__main__":
    # 1. 환경 초기화
    env = SidewalkStaticMetaUrbanEnv(ENV_CFG)
    print("Environment created. Resetting to get the first observation...")

    # 2. 환경 리셋 및 첫 관찰 데이터 획득
    # reset()을 여러 번 호출하면 스태킹된 프레임이 모두 동일한 초기 이미지로 채워집니다.
    # 의미있는 (움직임이 반영된) 프레임 스택을 보려면 env.step()을 몇 번 실행해야 합니다.
    obs, info = env.reset()
    
    # 3. 시각화 함수 호출
    print("Observation received. Visualizing...")
    visualize_stacked_obs(obs)
    
    # 4. 환경 종료
    env.close()
    print("Done.")