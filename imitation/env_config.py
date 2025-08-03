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
            use_render=True,
            map='X',
            manual_control=False,
            crswalk_density=1,
            object_density=0.01,
            walk_on_all_regions=False,
            drivable_area_extension=55,
            height_scale=1,
            horizon=1000, #에피소드 최대 길이 

            vehicle_config=dict(enable_reverse=True), # 후진 기능 활성화 

            show_sidewalk=True,
            show_crosswalk=True,
            random_lane_width=True,
            random_agent_model=True,
            random_lane_num=True,

            # random_spawn_lane_index=False,
            random_spawn_lane_index=True,
            num_scenarios=1000000,
            accident_prob=0,
            # max_lateral_dist=5.0,

            agent_type='coco',

            relax_out_of_road_done=False,
            agent_observation=ThreeSourceMixObservation,

            image_observation=True,
            sensors={
                "rgb_camera": (RGBCamera, *self.sensor_size),
                "depth_camera": (DepthCamera, *self.sensor_size),
                "semantic_camera": (SemanticCamera, *self.sensor_size),
            },
            log_level=50,
        )


"""
# 환경 설정
SENSOR_SIZE = (256, 160)
BASE_ENV_CFG = dict(
    use_render=True,
    map='X',
    manual_control=False,
    crswalk_density=1, #1,
    object_density=0.01, #0.1,
    walk_on_all_regions=False,
    drivable_area_extension=55,
    height_scale=1,
    horizon=1000,  # 에피소드 최대 길이
    
    vehicle_config=dict(enable_reverse=True), # 후진 기능 활성화
    
    show_sidewalk=True,
    show_crosswalk=True,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    
    # 시나리오 설정
    # random_spawn_lane_index=False,
    random_spawn_lane_index=True,
    num_scenarios=100000,
    accident_prob=0,
    max_lateral_dist=5.0,
    
    agent_type='coco', # 에이전트 타입
    
    relax_out_of_road_done=False, # 경로 이탈 시 종료 조건 강화
    
    agent_observation=ThreeSourceMixObservation,
    
    image_observation=True,
    sensors={
        "rgb_camera": (RGBCamera, *SENSOR_SIZE),
        "depth_camera": (DepthCamera, *SENSOR_SIZE),
        "semantic_camera": (SemanticCamera, *SENSOR_SIZE),
    },
    log_level=50, # 로그 레벨 (50은 에러만 표시)
)

"""