# inference_metaurban_nav.py
import os
import math
import time
import random
import numpy as np
from typing import Deque, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# =========================
# Configs (edit as needed)
# =========================
CHECKPOINT_PATH = "checkpoints/nav_model_epoch_90.pth"  # <-- put your checkpoint here
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10                    # must match training sequence_length (your main config uses 10)
STEER_SAT = 1.0                 # env steering range [-1, 1]
THROTTLE_BIAS = 0.4             # optional bias for throttle when using PD or low-confidence
WAYPOINT_AHEAD_INDEX = 5        # use the k-th checkpoint ahead
WAYPOINT_REACH_DIST = 5.0       # meters to switch to next waypoint
MAX_EPISODES = 10
RENDER_TEXT = True
USE_PD_UNTIL_WARM = True        # PD controller while buffer not yet filled

# ================
# Env base config
# ================
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True,
    map="X",
    manual_control=False,
    crswalk_density=1,
    object_density=0.01,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=1000,

    vehicle_config=dict(enable_reverse=True),

    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,

    random_spawn_lane_index=True,
    num_scenarios=100000,
    accident_prob=0,
    max_lateral_dist=5.0,
    agent_type="coco",

    relax_out_of_road_done=False,
    agent_observation=ThreeSourceMixObservation,

    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)

# ====================
# Geometry utilities
# ====================
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """
    World -> ego (agent-centric): rotate by -heading and translate by -agent_pos.
    Returns ego_vector: [x_left_right, y_forward_backward]
    """
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ vec_in_world

# ===============
# PD Controller
# ===============
class PD_Controller:
    def __init__(self, kp=0.2, kd=0.0, min_dt=0.1, throttle=THROTTLE_BIAS):
        self.kp = kp
        self.kd = kd
        self.min_dt = min_dt
        self.last_error = 0.0
        self.last_time = time.time()
        self.throttle = throttle

    def update(self, lateral_error):
        now = time.time()
        dt = max(self.min_dt, now - self.last_time)
        de = (lateral_error - self.last_error) / (dt + 1e-9)
        steer = self.kp * lateral_error + self.kd * de
        steer = float(np.clip(steer, -STEER_SAT, STEER_SAT))
        self.last_error = lateral_error
        self.last_time = now
        return [steer, self.throttle]

pd_controller = PD_Controller(kp=0.2, kd=0.0)

# ==========================
# Model: NavigationModel
# (matches your training)
# ==========================

# ====================================
# Navigation Model (Depth + Goal → Action)
# ====================================
class NavigationModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, action_dim=2):
        super(NavigationModel, self).__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten(),
            nn.Linear(256 * 7 * 7, hidden_size)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, hidden_size)
        )
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        
    def forward(self, depth_seq, goal_seq, hidden=None):
        batch_size, seq_len = depth_seq.shape[:2]
        depth_features, goal_features = [], []
        for t in range(seq_len):
            depth_features.append(self.depth_encoder(depth_seq[:, t]))
            goal_features.append(self.goal_encoder(goal_seq[:, t]))
        
        depth_features = torch.stack(depth_features, dim=1)
        goal_features = torch.stack(goal_features, dim=1)
        
        combined_features = torch.cat([depth_features, goal_features], dim=-1)
        gru_output, hidden = self.gru(combined_features, hidden)
        return self.action_head(gru_output), hidden



# ==========================
# Transforms (match training)
# ==========================
depth_transform = transforms.Compose([
    transforms.ToPILImage(mode="F") if hasattr(transforms, "ToPILImage") else (lambda x: Image.fromarray(x)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                      # (1,224,224) since grayscale
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_depth_from_obs(depth_obs_hw13_last: np.ndarray) -> torch.Tensor:
    """
    depth_obs_hw13_last: (H, W, 1) numpy float array (already selected last frame)
    Returns a (1, 1, 224, 224) torch tensor.
    """
    # ensure float32
    d = depth_obs_hw13_last.astype(np.float32)
    # torchvision expects HxW or HxWxC; here we have HxWx1
    depth_tensor = depth_transform(d).unsqueeze(0)  # (1,1,224,224)
    return depth_tensor

# ===================================
# Rolling buffer for (depth, goal)
# ===================================
class HistoryBuffer:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.depth_q: Deque[torch.Tensor] = deque(maxlen=seq_len)  # each (1,1,224,224)
        self.goal_q:  Deque[torch.Tensor] = deque(maxlen=seq_len)  # each (1,2)

    def reset(self):
        self.depth_q.clear()
        self.goal_q.clear()

    def push(self, depth_1_1_224_224: torch.Tensor, goal_1_2: torch.Tensor):
        self.depth_q.append(depth_1_1_224_224)
        self.goal_q.append(goal_1_2)

    def ready(self) -> bool:
        return (len(self.depth_q) == self.seq_len) and (len(self.goal_q) == self.seq_len)

    def as_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns depth_seq: (1,S,1,224,224) and goal_seq: (1,S,2)
        """
        depth_seq = torch.cat(list(self.depth_q), dim=0).unsqueeze(0)  # (1,S,1,224,224)
        goal_seq  = torch.cat(list(self.goal_q),  dim=0).unsqueeze(0)  # (1,S,2)
        return depth_seq.to(device), goal_seq.to(device)

    def warm_fill(self):
        """
        If buffer has n<S frames, left-pad by duplicating the first frame.
        Useful at episode start to avoid PD fallback.
        """
        if not self.depth_q:
            return
        first_d = self.depth_q[0].clone()
        first_g = self.goal_q[0].clone()
        while len(self.depth_q) < self.seq_len:
            self.depth_q.appendleft(first_d.clone())
            self.goal_q.appendleft(first_g.clone())

# ==========================
# Helper: extract obs frames
# ==========================
def get_latest_depth_frame(obs) -> np.ndarray:
    """
    MetaUrban depth shape is (H, W, 1, 3). We take the most recent (index -1) along the last axis.
    Returns (H, W, 1) float32 numpy.
    """
    d = obs["depth"][..., -1]  # (H,W,1)
    if d.dtype != np.float32:
        d = d.astype(np.float32)
    return d

def compute_ego_goal(env, waypoint_idx) -> Tuple[np.ndarray, int, int]:
    nav = env.agent.navigation
    wps = nav.checkpoints
    if len(wps) == 0:
        return np.array([0.0, 0.0], dtype=np.float32), 0, 0
    num_wps = len(wps)
    k = min(waypoint_idx, num_wps - 1)
    global_target = wps[k]
    agent_pos = env.agent.position
    agent_heading = env.agent.heading_theta
    ego_goal = convert_to_egocentric(global_target, agent_pos, agent_heading)
    return ego_goal.astype(np.float32), k, num_wps

def model_action_from_output(a_tanh_2: torch.Tensor) -> np.ndarray:
    """
    a_tanh_2: (1,S,2) in [-1,1]; we take the last step.
    Map to env action: [steer, throttle], steer in [-1,1], throttle in [0,1].
    """
    a_last = a_tanh_2[0, -1]                  # (2,)
    steer = float(a_last[0].clamp(-1, 1).item())
    throttle_unit = float(a_last[1].clamp(-1, 1).item())
    throttle = 0.5 * (throttle_unit + 1.0)    # [-1,1] -> [0,1]
    return np.array([steer, throttle], dtype=np.float32)

# ==========================
# Load model checkpoint
# ==========================
def load_navigation_model(checkpoint_path: str, device: torch.device) -> NavigationModel:
    model = NavigationModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # support both {"nav_model_state_dict": ...} and pure state_dict
    state_dict = ckpt["nav_model_state_dict"] if "nav_model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# ============
# Main loop
# ============
def main():
    # 1) Load model
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    model = load_navigation_model(CHECKPOINT_PATH, DEVICE)

    # 2) Create env
    env = SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)

    try:
        for ep in range(MAX_EPISODES):
            obs, info = env.reset(seed=ep + 2)

            # ensure enough waypoints
            waypoints = env.agent.navigation.checkpoints
            tries = 0
            while len(waypoints) < 30 and tries < 10:
                obs, info = env.reset(seed=random.randint(1, 50000))
                waypoints = env.agent.navigation.checkpoints
                tries += 1

            buf = HistoryBuffer(SEQ_LEN)
            k = WAYPOINT_AHEAD_INDEX

            # Prime one step to allow warm_fill
            ego_goal, k, num_wps = compute_ego_goal(env, k)
            d_hw1 = get_latest_depth_frame(obs)  # (H,W,1)
            d_t = preprocess_depth_from_obs(d_hw1)  # (1,1,224,224)
            g_t = torch.from_numpy(ego_goal).view(1, 2).to(torch.float32)  # (1,1,2)
            buf.push(d_t, g_t)
            buf.warm_fill()  # duplicate first frame until SEQ_LEN

            terminated = truncated = False
            hidden = None

            while True:
                # Update ego goal & buffer
                ego_goal, k, num_wps = compute_ego_goal(env, k)
                d_hw1 = get_latest_depth_frame(obs)
                d_t = preprocess_depth_from_obs(d_hw1)                # (1,1,224,224)
                g_t = torch.from_numpy(ego_goal).view(1, 2).to(torch.float32)  # (1,1,2)
                buf.push(d_t, g_t)

                # Choose action: PD until buffer ready (optional)
                if USE_PD_UNTIL_WARM and not buf.ready():
                    action = pd_controller.update(ego_goal[0])  # lateral error ~= ego x (left/right)
                    action = [0,-1]
                    print('pd !! this is pd controller!')
                else:
                    depth_seq, goal_seq = buf.as_batch(DEVICE)  # (1,S,1,224,224), (1,S,2)
                    with torch.no_grad():
                        a_seq, hidden = model(depth_seq, goal_seq, hidden=hidden)
                        print(a_seq)
                        print(hidden.shape)
                    print('nav model infer!!!')
                    action = model_action_from_output(a_seq)

                # Waypoint switching
                dist_to_goal = float(np.linalg.norm(ego_goal))
                if dist_to_goal < WAYPOINT_REACH_DIST:
                    k = min(k + 1, num_wps - 1)

                # Step env
                obs, reward, terminated, truncated, info = env.step(action)

                if RENDER_TEXT:
                    env.render(text={
                        "Agent Position": np.round(env.agent.position, 2),
                        "Agent Heading": f"{math.degrees(env.agent.heading_theta):.1f} deg",
                        "Reward": f"{reward:.2f}",
                        "Ego Goal (x,y)": np.round(ego_goal, 2),
                        "Action [steer,throttle]": np.round(action, 3),
                        "WP idx": k,
                        "Dist→WP": f"{dist_to_goal:.2f} m"
                    })
                else:
                    env.render()

                # Safety / termination
                if reward < 0:
                    print("[WARN] negative reward (possible collision). Ending episode.")
                    break
                if terminated or truncated:
                    print(f"Episode finished. Terminated={terminated}, Truncated={truncated}")
                    break
    finally:
        env.close()

if __name__ == "__main__":
    main()

