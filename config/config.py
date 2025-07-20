"""
PPO Environment Configuration for MetaUrban
"""
# from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

import torch
from dataclasses import dataclass

# Environment Configuration
SENSOR_SIZE = (256, 160)

BASE_ENV_CFG = dict(

    num_scenarios=1000,
    start_seed=1000,
    
    use_render=False,
    map='X', 
    manual_control=False, 
    crswalk_density=1, 
    object_density=0.1, 
    walk_on_all_regions=False,
    drivable_area_extension=55, 
    height_scale=1, 
    horizon=300,  # Long horizon
    
    vehicle_config=dict(enable_reverse=True), # 후진 가능 매우 중요
    
    show_sidewalk=True, 
    show_crosswalk=True,
    random_lane_width=True, 
    random_agent_model=True, 
    random_lane_num=True,
    
    # scenario setting
    random_spawn_lane_index=False,
    accident_prob=0,
    # relax_out_of_road_done=True,
    max_lateral_dist=5.0,    
    
    agent_type = 'coco', #['whellcahir']
    
    relax_out_of_road_done=False,  # More strict termination
    max_lateral_dist=10.0,  # Larger tolerance
    
    agent_observation=ThreeSourceMixObservation,
    
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),                
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50,
)


"""
circular C
InRamp	r	
OutRamp	R
Roundabout	O	
Intersection	X
Merge	y	
Split	Y
Tollgate	$	
Parking lot	P
TInterection	T	
Fork	WIP


"""
TRAIN_ENV_CFG = dict(
    use_render=False,
    map='X', 
    manual_control=False,
    
    num_scenarios=1000,
    start_seed=1000,
    training=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    
    crswalk_density=1,
    object_density=0.2,
    walk_on_all_regions=False,
    
    drivable_area_extension=55,
    height_scale=1,
    show_mid_block_map=False,
    show_ego_navigation=False,
    debug=False,
    horizon=300,
    on_continuous_line_done=False,
    out_of_route_done=True,
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=True,
    ),
    show_sidewalk=True,
    show_crosswalk=True,
    random_spawn_lane_index=False,
    accident_prob=0,
    relax_out_of_road_done=True,
    max_lateral_dist=5.0,
)



# def create_env():
#     """Create and return a configured MetaUrban environment"""
#     return SidewalkStaticMetaUrbanEnv(BASE_ENV_CFG)

# PPO Configuration
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 8  # More epochs for replay
    batch_size: int = 128
    buffer_size: int = 1024
    update_frequency: int = 256
    
    # Mixture policy weights with schedule
    policy_weight_start: float = 0.3
    policy_weight_end: float = 0.8
    prior_weight_start: float = 0.5
    prior_weight_end: float = 0.15
    exploration_weight: float = 0.05
    
    max_episodes: int = 500
    max_steps_per_episode: int = 300  # Long horizon
    save_frequency: int = 100
    eval_frequency: int = 50
    log_frequency: int = 10

@dataclass
class Experience:
    state: dict
    action: torch.Tensor
    reward: float
    next_state: dict
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
