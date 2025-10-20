from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.component.delivery_robot.deliveryrobot_type import EgoVehicle
from metaurban.utils import generate_gif

env=SidewalkStaticMetaUrbanEnv(dict(map="S", traffic_density=0, object_density=0.1, walk_on_all_regions=False))
frames = []
try:
    env.reset()
    cfg=env.config["vehicle_config"]
    cfg["navigation"]=None # it doesn't need navigation system
    v = env.engine.spawn_object(EgoVehicle, 
                                vehicle_config=cfg, 
                                position=[30,0], 
                                heading=0)
    for _ in range(100):
        v.before_step([0, 0.5])
        env.step([0,0])
        env.agents['default_agent'].set_position([25, 0])
        frame=env.render(mode="topdown", 
                         window=False,
                         screen_size=(800, 200),
                         camera_position=(60, 7))
        frames.append(frame)
    generate_gif(frames, gif_name="demo.gif")
finally:
    env.close()
