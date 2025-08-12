"""
Real-time multi-camera streaming with keyboard control for MetaUrban
Controls:
- W/S: Forward/Backward (throttle)
- A/D: Left/Right (steering)
- Q: Quit
- SPACE: Reset environment
"""
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
import cv2
import os
import numpy as np
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from stable_baselines3.common.monitor import Monitor
import argparse
import time

class KeyboardController:
    def __init__(self):
        self.steering = 0.0  # -1 to 1
        self.throttle = 0.0  # -1 to 1
        self.reset_flag = False
        self.quit_flag = False
        
    def process_key(self, key):
        """Process keyboard input and update action directly."""
        # Reset actions to zero at the beginning of each step
        # If no key is pressed, the action will be [0, 0] (stop)
        self.steering = 0.0
        self.throttle = 0.0

        if key == ord('w') or key == ord('W'):
            # 전진: [steering, throttle] = [0, 1]
            self.steering = 0.0
            self.throttle = 1.0
        elif key == ord('s') or key == ord('S'):
            # 후진: [steering, throttle] = [0, -1]
            self.steering = 0.0
            self.throttle = -1.0
        elif key == ord('a') or key == ord('A'):
            # 좌회전: [steering, throttle] = [1, 1]
            # MetaUrban 기본값과 반대이지만 요청에 따라 설정
            self.steering = 1.0
            self.throttle = 1.0
        elif key == ord('d') or key == ord('D'):
            # 우회전: [steering, throttle] = [-1, 1]
            # MetaUrban 기본값과 반대이지만 요청에 따라 설정
            self.steering = -1.0
            self.throttle = 1.0
            
        # --- Other controls (no changes needed here) ---
        elif key == ord(' '):  # Space bar
            self.reset_flag = True
        elif key == ord('q') or key == ord('Q'):
            self.quit_flag = True
        elif key == 27:  # ESC key
            self.quit_flag = True
            
    def get_action(self):
        return [self.steering, self.throttle]

def setup_windows():
    """Setup OpenCV windows for display"""
    window_names = ['RGB Camera', 'Depth Camera', 'Semantic Camera', 'Top-down Semantic']
    
    for i, name in enumerate(window_names):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        # Position windows in a 2x2 grid
        x = (i % 2) * 650
        y = (i // 2) * 400
        cv2.moveWindow(name, x, y)
    
    return window_names

def process_rgb_image(rgb_data, config):
    """Process RGB camera data"""
    max_rgb_value = rgb_data.max()
    rgb = rgb_data[..., ::-1]  # BGR to RGB
    if max_rgb_value > 1:
        rgb = rgb.astype(np.uint8)
    else:
        rgb = (rgb * 255).astype(np.uint8)
    return rgb

def process_depth_image(depth_data):
    """Process depth camera data"""
    depth_reshaped = depth_data.reshape(360, 640, -1)[..., -1]
    depth_normalized = cv2.normalize(depth_reshaped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colored

def process_semantic_image(semantic_data):
    """Process semantic camera data"""
    semantic = (semantic_data[..., ::-1] * 255).astype(np.uint8)
    return semantic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_images", action="store_true", help="Save images to disk")
    parser.add_argument("--out_dir", type=str, default="saved_imgs", help="Output directory for saved images")
    args = parser.parse_args()
    
    if args.save_images:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Environment configuration
    map_type = 'X'
    config = dict(
        crswalk_density=1,
        object_density=0.4,
        use_render=True,
        walk_on_all_regions=False,
        map=map_type,
        manual_control=False,  # We'll control via keyboard
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
        num_scenarios=100,
        accident_prob=0,
        window_size=(1200, 900),
        relax_out_of_road_done=True,
        max_lateral_dist=1e10,
        camera_dist=0.8,
        camera_height=1.5,
        camera_pitch=None,
        camera_fov=66,
        norm_pixel=False,
    )
    
    # Add camera sensors
    config.update(
        dict(
            image_observation=True,
            sensors=dict(
                rgb_camera=(RGBCamera, 640, 360),
                depth_camera=(DepthCamera, 640, 360),
                semantic_camera=(SemanticCamera, 640, 360),
                top_down_semantic=(SemanticCamera, 512, 512)
            ),
            agent_observation=ThreeSourceMixObservation,
            interface_panel=[]
        )
    )
    
    # Initialize environment and controller
    env = SidewalkStaticMetaUrbanEnv(config)
    controller = KeyboardController()
    
    # Setup display windows
    window_names = setup_windows()
    
    print("=== CONTROLS ===")
    print("W/S: Forward/Backward (throttle)")
    print("A/D: Left/Right (steering)")
    print("SPACE: Reset environment")
    print("Q/ESC: Quit")
    print("================")
    
    o, _ = env.reset(seed=20)
    scenario_t = 0
    
    try:
        while True:
            # Get keyboard input
            key = cv2.waitKey(1) & 0xFF
            controller.process_key(key)
            
            if controller.quit_flag:
                break
            
            if controller.reset_flag:
                env.reset(env.current_seed + 1)
                controller.reset_flag = False
                scenario_t = 0
                print(f"Environment reset to seed: {env.current_seed}")
                continue
            
            # Get action from controller
            action = controller.get_action()
            
            action = [0,1]
            # Step environment
            o, r, tm, tc, info = env.step(action)
            
            # Print position info every 30 frames (~1 second at 30fps)
            if scenario_t % 30 == 0:
                agent_pos = env.agent.position
                agent_heading = env.agent.heading_theta
                waypoints = env.agent.navigation.checkpoints
                print(f"Step {scenario_t:06d} | Pos: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}) | "
                      f"Heading: {agent_heading:.2f} | Action: [{action[0]:.2f}, {action[1]:.2f}]")
            
            # ===== Get camera data =====
            front_offset = -1.5
            camera_height = 1.5
            
            # RGB Camera
            rgb_camera = env.engine.get_sensor("rgb_camera")
            rgb_data = rgb_camera.perceive(
                to_float=config['norm_pixel'],
                new_parent_node=env.agent.origin,
                position=[0, front_offset, camera_height],
                hpr=[0, 0, 0]
            )
            rgb_img = process_rgb_image(rgb_data, config)
            
            # Depth Camera
            depth_camera = env.engine.get_sensor("depth_camera")
            depth_data = depth_camera.perceive(
                to_float=config['norm_pixel'],
                new_parent_node=env.agent.origin,
                position=[0, front_offset, camera_height],
                hpr=[0, 0, 0]
            )
            depth_img = process_depth_image(depth_data)
            
            # Semantic Camera (perspective)
            semantic_camera = env.engine.get_sensor("semantic_camera")
            semantic_data = semantic_camera.perceive(
                to_float=config['norm_pixel'],
                new_parent_node=env.agent.origin,
                position=[0, front_offset, camera_height],
                hpr=[0, 0, 0]
            )
            semantic_img = process_semantic_image(semantic_data)
            
            # Top-down Semantic Camera
            top_down_camera = env.engine.get_sensor("top_down_semantic")
            top_down_data = top_down_camera.perceive(
                new_parent_node=env.agent.origin,
                position=[0, 0, 15],  # Height for top-down view
                hpr=[0, -90, 0]       # Look straight down
            )
            top_down_img = process_semantic_image(top_down_data)
            
            # ===== Display images =====
            cv2.imshow(window_names[0], rgb_img)
            cv2.imshow(window_names[1], depth_img)
            cv2.imshow(window_names[2], semantic_img)
            cv2.imshow(window_names[3], top_down_img)
            
            # ===== Save images if requested =====
            if args.save_images and scenario_t % 10 == 0:  # Save every 10 frames
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_rgb.png"), 
                           rgb_img[..., ::-1])  # Convert back to BGR for saving
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic.png"), 
                           semantic_img[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_depth.png"), 
                           depth_img)
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_topdown.png"), 
                           top_down_img[..., ::-1])
            
            scenario_t += 1
            
            # Reset if episode is done
            if tm or tc:
                env.reset(env.current_seed + 1)
                scenario_t = 0
                print(f"Episode finished. Reset to seed: {env.current_seed}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()