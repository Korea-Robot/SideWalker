from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

sensor_size = (128, 128)
cfg=dict(object_density=0.1,
         agent_observation=ThreeSourceMixObservation,
         image_observation=True,
         sensors={"rgb_camera": (RGBCamera, *sensor_size),
                  "depth_camera": (DepthCamera, *sensor_size),
                  "semantic_camera": (SemanticCamera, *sensor_size)},
         log_level=50) # turn off log

from metaurban.utils import generate_gif
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from IPython.display import Image
import numpy as np
import cv2

frames = []
env=SidewalkStaticMetaUrbanEnv(cfg)
try:
    env.reset()
    print("Observation shape: \n", env.observation_space)
    for step in range(150):
        o, r, d, _, _ = env.step([0,0.5]) # simulation
        
        # visualize image observation
        o_1 = o["depth"][..., -1]
        o_1 = np.concatenate([o_1, o_1, o_1], axis=-1) # align channel
        o_2 = o["image"][..., -1]
        o_3 = o["semantic"][..., -1]
        ret = cv2.hconcat([o_1, o_2, o_3])*255
        ret=ret.astype(np.uint8)
        frames.append(ret[..., ::-1])
        if d:
            break
    generate_gif(frames) # only show 250 frames
finally:
    env.close()
