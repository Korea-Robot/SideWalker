import torch
import numpy as np
import cv2
import random
from pathlib import Path
from dataclasses import dataclass

# Import necessary classes from your training script
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
import torch.nn as nn

# --- Core Classes from Training Script (Keep these for model definition) ---

SENSOR_SIZE = (256, 160)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=6, feature_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, SENSOR_SIZE[1], SENSOR_SIZE[0])
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PPOPolicy(nn.Module):
    def __init__(self, feature_dim=512, goal_vec_dim=2, action_dim=2):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        combined_dim = feature_dim + goal_vec_dim
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, images, goal_vec):
        img_features = self.feature_extractor(images)
        combined = torch.cat([img_features, goal_vec], dim=1)
        mean = self.policy_head(combined)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

class ObservationProcessor:
    @staticmethod
    def preprocess_observation(obs):
        depth = obs["depth"][..., -1]
        depth = np.concatenate([depth, depth, depth], axis=-1)
        semantic = obs["semantic"][..., -1]
        
        combined_img = np.concatenate([depth, semantic], axis=-1)
        combined_img = combined_img.astype(np.float32) / 255.0
        combined_img = np.transpose(combined_img, (2, 0, 1))
        
        goal_vec = obs["goal_vec"].astype(np.float32)
        
        return {
            'images': torch.tensor(combined_img).unsqueeze(0),
            'goal_vec': torch.tensor(goal_vec).unsqueeze(0)
        }

# --- Inference Script ---

def run_inference(model_path: str):
    """
    Loads a trained PPO model and runs inference in the MetaUrban environment.
    """
    # 1. Setup device and environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Environment config for visualization
    env_config = {
        "use_render": True,  # <<< Enable rendering to see the agent
        "image_observation": True,
        "sensors": {
            "rgb_camera": (RGBCamera, *SENSOR_SIZE),
            "depth_camera": (DepthCamera, *SENSOR_SIZE),
            "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
        },
        "agent_observation": ThreeSourceMixObservation,
        "map": "X",
        "crswalk_density": 1,
        "out_of_route_done": True,
        "log_level": 50,
    }
    env = SidewalkStaticMetaUrbanEnv(env_config)

    # 2. Load the trained policy model
    policy = PPOPolicy().to(device)
    
    # Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval() # Set the model to evaluation mode
    
    print(f"Model loaded successfully from {model_path}")

    # 3. Run the simulation loop
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Preprocess the observation
        nav = env.vehicle.navigation.get_navi_info()
        obs["goal_vec"] = np.array(nav[:2], dtype=np.float32)
        state = ObservationProcessor.preprocess_observation(obs)

        # Move state to the correct device
        images = state['images'].to(device)
        goal_vec = state['goal_vec'].to(device)

        # Get action from the policy (no gradients needed)
        with torch.no_grad():
            dist = policy(images, goal_vec)
            # Use the mean of the distribution for deterministic action
            action = dist.mean
            action = torch.tanh(action) # Apply tanh squashing like in training

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action.cpu().squeeze().numpy())
        done = terminated or truncated

        # Render the environment view
        env.render(mode="top_down", film_size=(800, 800))
        
        if done:
            print("Episode finished.")
            
    env.close()

if __name__ == "__main__":
    # Ensure you have a trained model at this path
    MODEL_FILE_PATH = "../checkpoints/final_model.pth" 
    
    if not Path(MODEL_FILE_PATH).exists():
        print(f"Error: Model file not found at '{MODEL_FILE_PATH}'")
        print("Please make sure you have trained the model and saved it correctly.")
    else:
        run_inference(model_path=MODEL_FILE_PATH)