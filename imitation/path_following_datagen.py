import numpy as np
import os
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import math
import random 

# Import configurations and utilities
from env_config import EnvConfig
from utils import PD_Controller,convert_to_egocentric

## PD controller 생성
pd_controller = PD_Controller(kp=0.3,kd=0.1)

# 데이터셋 수집기 초기화
from walle_dataset import ImitationDatasetCollector
collector = ImitationDatasetCollector("imitation_dataset")

# --- 메인 실행 로직 ---
# env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)
env_config = EnvConfig()
env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)

running = True 
try:
    # 여러 에피소드 실행 : 각 에피소드 마다 데잍 저장 
    for i in range(100000):
        
        # 일정 waypoints 이상 없으면 다시 환경 구성 .
        obs,info = env.reset(seed=i + 2)
        
        waypoints = env.agent.navigation.checkpoints 
        print('wayppoint num: ',len(waypoints))
        
        episode = random.randint(1,40000)
        while len(waypoints)<30:
            obs,info = env.reset(seed= episode)
            episode = random.randint(1,40000)
            waypoints = env.agent.navigation.checkpoints 
            print('NOT sufficient waypoints ',episode,' !!!!',len(waypoints))
        num_waypoints = len(waypoints)
        
        # 5번째 웨이포인트를 목표로 설정
        k = 5
        step = 0 

        # 만약에 끼어서 계속 가만히 있는경우 제거하기 위해서.
        start_position = env.agent.position
        stuck_interval = 10
        
        # 에피소드 루프
        while running:

            # --- 목표 지점 계산 (Egocentric) ---
            ego_goal_position = np.array([0.0, 0.0]) # 기본값 초기화
            nav = env.agent.navigation
            waypoints = nav.checkpoints
            
            # 웨이포인트가 충분히 있는지 확인

            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            
            # k 번째 waypoint의 ego coordinate 기준 좌표 
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)

            # action [steering, throttle] : steeering : -1~1 : throttle : -1~1 : 마이너스 값은 브레이크임.
            # action = [1,1] # 왼쪽 주행. 
            # action = [-1,1] # 오른쪽 주행. 
            
            action = pd_controller.update(ego_goal_position[1]) # 
            
            # ----------- 목표 웨이포인트 업데이트 ---------------- 
            # 목표지점까지 직선거리 계산 
            distance_to_target = np.linalg.norm(ego_goal_position)
            
            # 목표 지점 업데이트
            if distance_to_target< 5.0:
                k +=1
                if k>= num_waypoints:
                    k = num_waypoints-1

            # 에이전트 상태 정보 수집
            agent_state = {
                "position": env.agent.position,
                "heading": env.agent.heading_theta,
                "velocity": env.agent.speed,
                "angular_velocity": getattr(env.agent, 'angular_velocity', 0.0)
            }
            
            # 데이터 수집
            collector.collect_sample(
                obs, action, agent_state, ego_goal_position, reward, step
            )                

            # 선택된 액션으로 환경을 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)

            if step-stuck_interval > 0:
                future_agent_pos = env.agent.position
                length = np.linalg.norm(future_agent_pos-start_position)
                if length <1:
                    break
                else:
                    start_position= future_agent_pos
                    stuck_interval+=10

            
            # 환경 렌더링 및 정보 표시
            env.render(
                text={
                    "Agent Position": np.round(env.agent.position, 2),
                    "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                    "Reward": f"{reward:.2f}",
                    "Ego Goal Position": np.round(ego_goal_position, 2)
                }
            )

            # 일정이상 시간동안 끼어서 안움직이면 종료. 
            # stuck_distance = sqrt((x_{t+30} - x_t)² + (y_{t+30} - y_t)²) 
            
            # if stuck_distance < 1:
                # break 
            
            # 에피소드 종료 조건
            if terminated or truncated or step >= 800 or reward <0:
                episode_info = {
                    "seed": episode + 42,
                    "terminated": terminated,
                    "truncated": truncated,
                    "total_reward": episode_reward,
                    "episode_length": step,
                    "success": reward > 0.5  # 성공 기준
                }
            
            step+=1 
            
        # 에피소드 하나 종료 및 시작 
        collector.finish_episode(episode_info)

        # 에피소드 종료 조건 확인
        if terminated or truncated:
            print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}")
            break


        # 데이터셋 정보 저장
        collector.save_dataset_info()
        collector.create_train_test_split()
        
        print(f"\nDataset collection completed!")
        print(f"Total episodes: {collector.episode_counter}")
        print(f"Total samples: {collector.dataset_info['total_samples']}")
        print(f"Dataset saved to: {collector.dataset_root}")

finally:
    # 종료 시 리소스 정리
    env.close()



"""    
1.  **목표 지점 계산 로직 추가**: 메인 루프 안에서 `nav.checkpoints`를 가져와 마지막 웨이포인트를 목표 지점으로 설정하고, 계속 자기위치 기반으로 바로 앞에 가야할 위치를 업데이트 하면서 조종
2. PD controller를 통해  이동하도록 지시 


일단은 depth,segmantic, rgb데이터의 png, action & reward json or csv , 
waypoints, ego state에 대한 csv or json을 에피소드 별로 저장하려고해. 
그런데 문제는 끝나는 지점이 각각 달라서 항상 임의의 개수만큼 에피소드마다데이터가 쌓인다는거야. 
이걸 어떻게 pytorch dataset dataloader로 만들까?


# 데이터 구조 분리.
data/
├── episode_0001/
│   ├── rgb/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   ├── depth/
│   │   ├── 0000.png
│   ├── semantic/
│   │   ├── 0000.png
│   ├── action_reward.json
│   ├── ego_state.json        # 위치, 헤딩, 속도 등
│   ├── waypoints.json
├── episode_0002/
...


action_reward.json
[
  {"step": 0, "action": [0.1, 0.4], "reward": 0.2, "done": false},
  {"step": 1, "action": [0.0, 0.5], "reward": 0.3, "done": false},
  {"step": 2, "action": [-0.2, 0.3], "reward": -1.0, "done": true}
]

ego_state.json
[
  {"position": [1.2, 0.4], "heading": 0.12},
  {"position": [1.5, 0.5], "heading": 0.13},
  ...
]


"""

class TransitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.Resize((160, 256)),
            T.ToTensor()
        ])
        self.transitions = []  # (episode_path, idx)

        self._build_index()

    def _build_index(self):
        # 각 에피소드에서 transition (t, t+1)을 수집
        for ep_name in sorted(os.listdir(self.root_dir)):
            ep_path = os.path.join(self.root_dir, ep_name)
            if not os.path.isdir(ep_path):
                continue
            with open(os.path.join(ep_path, "action_reward.json")) as f:
                actions = json.load(f)
            length = len(actions)
            for i in range(length - 1):
                self.transitions.append((ep_path, i))

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        ep_path, i = self.transitions[idx]

        def load_image(folder, idx):
            img = Image.open(os.path.join(ep_path, folder, f"{idx:04d}.png"))
            return self.transform(img)

        # Load obs_t
        rgb_t = load_image("rgb", i)
        depth_t = load_image("depth", i)
        semantic_t = load_image("semantic", i)

        # Load obs_{t+1}
        rgb_tp1 = load_image("rgb", i+1)
        depth_tp1 = load_image("depth", i+1)
        semantic_tp1 = load_image("semantic", i+1)

        with open(os.path.join(ep_path, "action_reward.json")) as f:
            ar_data = json.load(f)
        with open(os.path.join(ep_path, "ego_state.json")) as f:
            ego_data = json.load(f)

        action = torch.tensor(ar_data[i]["action"], dtype=torch.float32)
        reward = torch.tensor(ar_data[i]["reward"], dtype=torch.float32)
        done = torch.tensor(ar_data[i]["done"], dtype=torch.bool)

        # Ego state
        pos_t = torch.tensor(ego_data[i]["position"], dtype=torch.float32)
        pos_tp1 = torch.tensor(ego_data[i+1]["position"], dtype=torch.float32)
        heading_t = torch.tensor(ego_data[i]["heading"], dtype=torch.float32)
        heading_tp1 = torch.tensor(ego_data[i+1]["heading"], dtype=torch.float32)

        return {
            "rgb_t": rgb_t,
            "depth_t": depth_t,
            "semantic_t": semantic_t,
            "action": action,
            "reward": reward,
            "done": done,
            "rgb_tp1": rgb_tp1,
            "depth_tp1": depth_tp1,
            "semantic_tp1": semantic_tp1,
            "pos_t": pos_t,
            "pos_tp1": pos_tp1,
            "heading_t": heading_t,
            "heading_tp1": heading_tp1
        }


# 가변 에피소드 처리
def collate_episodes(batch):
    """
    batch: list of episodes. Each episode is a list of dicts (one per timestep)
    Returns: padded tensors or batched sequences
    """
    batch_data = []
    for episode in batch:
        frames = episode["episode"]
        batch_data.append(frames)
    return batch_data


# 사용 예시
from torch.utils.data import DataLoader

dataset = EpisodeDataset("data")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_episodes)

for batch in loader:
    # batch[i][t]["rgb"] 형태로 접근
    # i: 에피소드 index, t: timestep index
    rgb_seq_0 = [step["rgb"] for step in batch[0]]
    action_seq_0 = [step["action"] for step in batch[0]]
