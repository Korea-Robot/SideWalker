from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
import cv2
from IPython.display import Image
from metaurban.utils import generate_gif
import numpy as np
import os
sensor_size = (640, 360) if os.getenv('TEST_DOC') else (200, 100)

cfg=dict(object_density=0.1,
         image_observation=True, 
         vehicle_config=dict(image_source="rgb_camera"),
         sensors={"rgb_camera": (RGBCamera, *sensor_size)},
         stack_size=3,
        )

env=SidewalkStaticMetaUrbanEnv(cfg)
frames = []
try:
    env.reset()
    for _ in range(100):
        # simulation
        o, r, d, _, _ = env.step([0,1])
        # rendering, the last one is the current frame
        ret=o["image"][..., -1]*255 # [0., 1.] to [0, 255]
        
        breakpoint()
        
        ret=ret.astype(np.uint8)
        frames.append(ret[..., ::-1])
        if d:
            break
    generate_gif(frames)
finally:
    env.close()