# autonomous_driving_data_gen.py
import multiprocessing as mp
import numpy as np
import torch
import time
import cv2
from typing import Dict, Any
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

mp.set_start_method("spawn", force=True)

SENSOR_SIZE = (256, 160)

BASE_ENV_CFG = dict(
    use_render=False,
    map='X',
    manual_control=False,
    crswalk_density=1,
    object_density=0.1,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=300,
    vehicle_config=dict(enable_reverse=True, image_source="rgb_camera"), # rgb_camera setting
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    relax_out_of_road_done=True,
    max_lateral_dist=5.0,
    agent_observation=ThreeSourceMixObservation,
    image_observation=True,
    # image_on_cuda=True,  # ✅ 추가: GPU에 직접 이미지 보관
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

def preprocess_observation(obs):
    depth = obs["depth"][..., -1]
    depth = np.concatenate([depth, depth, depth], axis=-1)
    semantic = obs["semantic"][..., -1]
    combined_img = np.concatenate([depth, semantic], axis=-1)
    combined_img = combined_img.astype(np.float32) / 255.0
    combined_img = np.transpose(combined_img, (2, 0, 1))
    goal_vec = obs["goal_vec"].astype(np.float32)
    return {
        'images': torch.tensor(combined_img),
        'goal_vec': torch.tensor(goal_vec)
    }

def compute_reward(obs, action, next_obs, done, info):
    reward = 0.0
    goal_vec = next_obs["goal_vec"]
    goal_distance = np.linalg.norm(goal_vec)
    reward += max(0, 1.0 - goal_distance)
    speed = info.get('speed', 0)
    reward += min(speed / 10.0, 1.0)
    if info.get('crash', False):
        reward -= 10.0
    if info.get('out_of_road', False):
        reward -= 5.0
    if info.get('arrive_dest', False):
        reward += 20.0
    reward -= 0.01
    return reward

def env_worker(worker_id: int, data_queue: mp.Queue, is_training: bool = True):
    print(f"Worker {worker_id} started")
    env_config = BASE_ENV_CFG.copy()
    env_config.update({
        'num_scenarios': 1000 if is_training else 200,
        'start_seed': 1000 + worker_id if is_training else worker_id,
        'training': is_training,
        'seed': 1000 + worker_id if is_training else worker_id
    })
    env = SidewalkStaticMetaUrbanEnv(env_config)
    try:
        obs, _ = env.reset()
        nav = env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        state = preprocess_observation(obs)
        episode_reward = 0
        step_count = 0
        while True:
            action = env.action_space.sample()
            next_obs, _, done, truncated, info = env.step(action)
            nav = env.vehicle.navigation.get_navi_info()
            next_obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
            reward = compute_reward(obs, action, next_obs, done, info)
            next_state = preprocess_observation(next_obs)
            experience = {
                'worker_id': worker_id,
                'state': state,
                'action': torch.tensor(action, dtype=torch.float32),
                'reward': reward,
                'next_state': next_state,
                'done': done or truncated,
                'log_prob': torch.tensor(0.0)
            }
            try:
                data_queue.put(experience, timeout=1.0)
            except:
                print(f"Worker {worker_id}: Queue full, skipping data")
            episode_reward += reward
            step_count += 1
            if done or truncated:
                print(f"Worker {worker_id}: Episode finished - Reward: {episode_reward:.2f}, Steps: {step_count}")
                obs, _ = env.reset()
                nav = env.vehicle.navigation.get_navi_info()
                obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
                state = preprocess_observation(obs)
                episode_reward = 0
                step_count = 0
            else:
                obs = next_obs
                state = next_state
            time.sleep(0.001)
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        env.close()
        print(f"Worker {worker_id} finished")

def main():
    NUM_WORKERS = 4
    data_queue = mp.Queue(maxsize=4096)
    workers = []
    for i in range(NUM_WORKERS):
        worker = mp.Process(target=env_worker, args=(i, data_queue, True), name=f"EnvWorker-{i}")
        worker.start()
        workers.append(worker)
    print(f"{NUM_WORKERS} data collection workers started.")
    try:
        while True:
            data = data_queue.get(timeout=10.0)
            # 필요시 저장 또는 처리 코드 삽입
            print(f"Received data from worker {data['worker_id']}")
    except KeyboardInterrupt:
        print("Stopping data collection...")
    finally:
        for worker in workers:
            worker.terminate()
            worker.join(timeout=5.0)
        print("All workers terminated.")

if __name__ == "__main__":
    main()

