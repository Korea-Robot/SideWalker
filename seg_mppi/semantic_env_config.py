# env_config.py
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

@dataclass
class EnvConfig:
    # sensor_size: Tuple[int, int] = (160, 120)
    sensor_size: Tuple[int, int] = (640, 360)
    base_env_cfg: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.base_env_cfg = dict(
            crswalk_density=1,
            object_density=0.01,
            use_render=True,
            walk_on_all_regions=False,
            map='X',
            manual_control=True,
            drivable_area_extension=55,
            height_scale=1,
            spawn_deliveryrobot_num=2,
            show_mid_block_map=False,
            show_ego_navigation=False,
            debug=False,
            horizon=300,
            on_continuous_line_done=False,
            out_of_route_done=True,
            vehicle_config=dict(
                show_lidar=False,
                show_navi_mark=True,
                show_line_to_navi_mark=False,
                show_dest_mark=False,
                enable_reverse=True,
                policy_reverse=False,
            ),
            show_sidewalk=True,
            show_crosswalk=True,
            random_spawn_lane_index=False,
            num_scenarios=200000,
            accident_prob=0,
            window_size=(1200, 900),
            relax_out_of_road_done=True,
            max_lateral_dist=1e10,
            
            camera_dist = -2,
            camera_height = 2.2,
            camera_pitch = None,
            camera_fov = 90,
            norm_pixel=False,
            image_observation=True,
            sensors=dict(
                # Perspective-view cameras
                rgb_camera=(RGBCamera,640, 360),
                depth_camera=(DepthCamera, 640, 360),
                semantic_camera=(SemanticCamera, 640,360),
                
                ####### ADDED #######
                # Top-down semantic camera
                # 해상도는 필요에 따라 조절 (e.g., 512, 512)
                top_down_semantic=(SemanticCamera, 512, 512) 
            ),
            agent_observation=ThreeSourceMixObservation,
            interface_panel=[]
        )
