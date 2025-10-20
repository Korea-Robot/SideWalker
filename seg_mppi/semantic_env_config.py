"""
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
            object_density=0.9,
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
            camera_fov = 90, # 66
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
"""


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
            # ===== 교통 및 정적 장애물 설정 =====
            num_agents=40,
            object_density=1.0,

            # ===== 동적 보행자 및 로봇 설정 =====
            spawn_adult_num=10,         # [추가] 성인 보행자 수
            spawn_child_num=5,          # [추가] 어린이 보행자 수
            spawn_wheelchairman_num=2,
            spawn_deliveryrobot_num=15,
            spawn_edog_num=2,
            spawn_erobot_num=2,
            spawn_drobot_num=2,         # [추가] 배달 로봇(drobot) 수
            
            crswalk_density=1,
            # ===== 맵 복잡도 증가 =====
            map='CCCCr',                        # 교차로(C) 4개와 원형 교차로(r) 1개로 구성된 복잡한 맵을 생성합니다.
            num_scenarios=200000,               # 다양한 맵 생성을 위해 시나리오 수는 높게 유지합니다.

            # ===== 모델 학습을 위한 설정 변경 =====
            manual_control=False,               # 모델 학습을 위해 수동 조작을 비활성화합니다.
            horizon=1500,                       # 에피소드 최대 길이를 늘려 더 긴 주행을 학습하게 합니다.
            out_of_route_done=True,             # 경로 이탈 시 에피소드를 종료합니다.

            # ===== 기존 설정 유지/일부 조정 =====
            use_render=True,                    # 시각적 확인을 위해 렌더링은 유지합니다.
            drivable_area_extension=55,
            height_scale=1,
            show_mid_block_map=False,
            show_ego_navigation=False,
            debug=False,
            on_continuous_line_done=False,
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
            accident_prob=0,
            window_size=(1200, 900),
            relax_out_of_road_done=True,
            max_lateral_dist=1e10,

            # ===== 카메라 및 관측 설정 (기존 유지) =====
            camera_dist = -2,
            camera_height = 2.2,
            camera_pitch = None,
            camera_fov = 90,
            norm_pixel=False,
            image_observation=True,
            sensors=dict(
                # Perspective-view cameras
                rgb_camera=(RGBCamera, 640, 360),
                depth_camera=(DepthCamera, 640, 360),
                semantic_camera=(SemanticCamera, 640, 360),

                ####### ADDED #######
                # Top-down semantic camera
                # 해상도는 필요에 따라 조절 (e.g., 512, 512)
                top_down_semantic=(SemanticCamera, 512, 512)
            ),
            agent_observation=ThreeSourceMixObservation,
            interface_panel=[]
        )