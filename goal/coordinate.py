from metaurban.envs import SidewalkStaticMetaUrbanEnv
import os
test = os.getenv('TEST_DOC')


# create environment
env = SidewalkStaticMetaUrbanEnv(dict(use_render=False, 
                        # debug = True,
                        show_coordinates=True, 
                        num_scenarios=1,
                        map="XSOS",
                        object_density=0.1,
                        drivable_area_extension=55))


from metaurban import SidewalkStaticMetaUrbanEnv
import tqdm

env = SidewalkStaticMetaUrbanEnv(dict(
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
))


# reset environment
env.reset()
try:
    for i in range(1000):
        o,r,d,t,_ = env.step([0,1])
        nav = env.agent.navigation 
        # goal_position = env.agent.navigation.destination
        pos = env.agent.position
        # 수동으로 설정
        # training_env.agent.navigation.set_destination((x, y))

        breakpoint()
finally:
    env.close()
    
    
    
