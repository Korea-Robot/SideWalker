from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.semantic_camera import SemanticCamera
import matplotlib.pyplot as plt
import os

size = (256, 128) if not os.getenv('TEST_DOC') else (16, 16) # for github CI

env = SidewalkStaticMetaUrbanEnv(dict(
    object_density=0.1,
    log_level=50, # suppress log
    image_observation=True,
    show_terrain=not os.getenv('TEST_DOC'),
    sensors={"sementic_camera": [SemanticCamera, *size]},
    vehicle_config={"image_source": "sementic_camera"},
    stack_size=3,
))
obs, info = env.reset()
for _ in range(5):
    obs, r, d, t, i = env.step((0, 1))

env.close()

print({k: v.shape for k, v in obs.items()})  # Image is in shape (H, W, C, num_stacks)

plt.subplot(131)
plt.imshow(obs["image"][:, :, :, 0])
plt.axis('off')
plt.subplot(132)
plt.imshow(obs["image"][:, :, :, 1])
plt.axis('off')
plt.subplot(133)
plt.imshow(obs["image"][:, :, :, 2])
plt.axis('off')
