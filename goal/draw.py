from metaurban.envs import SidewalkStaticMetaUrbanEnv
import numpy as np
import os
render = not os.getenv('TEST_DOC')

# Define a tool function. 
def make_line(x_offset, y_offset, height, y_dir=1, color=(1,105/255,180/255)):
    points = [(x_offset+x,x*y_dir+y_offset,height*x/10+height) for x in range(10)]
    colors = [np.clip(np.array([*color,1])*(i+1)/11, 0., 1.0) for i in range(10)]
    if y_dir<0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors

# create environment
env = SidewalkStaticMetaUrbanEnv(dict(use_render=render, object_density=0.1)) 


### draw line
# env.reset() # launch the simulation
# line_1, color_1 = make_line(env.agent.position[0], env.agent.position[1], 0.5, 1) # define line 1 for test
# line_2, color_2 = make_line(env.agent.position[0], env.agent.position[1], 0.5, -1) # define line 2 for test
# lines = [line_1, line_2]
# colors = [color_1, color_2]

# try:
#     drawer = env.engine.make_line_drawer(thickness=5) # create a line drawer
#     drawer.draw_lines(lines, colors) # draw lines
    
#     for i in range(100):
#         env.step([0,0])
# finally:    
#     env.close()


## draw points!!4

# env.reset() # launch the simulation
# try:
#     drawer = env.engine.make_point_drawer(scale=1) # create a point drawer
#     for i in range(100):
        
#         # draw different lines every step
#         line_1, color_1 = make_line(env.agent.position[0], env.agent.position[1], 0.5, 0.01*i) # define line 1 for test
#         line_2, color_2 = make_line(env.agent.position[0], env.agent.position[1], 0.5, -0.01*i) # define line 2 for test
#         points = line_1 + line_2 # create point list
#         colors = color_1+ color_2
#         drawer.reset()
#         drawer.draw_points(points, colors) # draw points
        
#         env.step([0,0])
# finally:    
#     env.close()
    
    
env.reset() # launch the simulation

nav = env.agent.navigation
polyline = nav.reference_trajectory.get_polyline() 
try:
    point_drawer = env.engine.make_point_drawer(scale=1) # create a point drawer
    line_drawer = env.engine.make_line_drawer(thickness=5) # create a line drawer
    for i in range(100):
            
        if i%5==0:
            # draw different lines every step
            line_1, color_1 = make_line(env.agent.position[0], env.agent.position[1], 0.5, 0.01*i) # define line 1 for test
            line_2, color_2 = make_line(env.agent.position[0], env.agent.position[1], 0.5, -0.01*i) # define line 2 for test
            points = line_1
            point_colors = color_1
            lines = [line_2]
            line_colors = [color_2]
            # drawer.reset()
            point_drawer.draw_points(points, point_colors) # draw lines
            line_drawer.draw_lines(lines, line_colors)
        
        env.step([0,0])
finally:    
    env.close()