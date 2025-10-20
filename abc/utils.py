import numpy as np 
import math

class PDController:
    def __init__(self, p_gain, d_gain):
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.prev_error = 0

    def get_control(self, target_angle, current_angle):
        error = target_angle - current_angle
        # 각도 차이가 -pi ~ pi 범위에 있도록 정규화
        error = (error + math.pi) % (2 * math.pi) - math.pi
        
        control = self.p_gain * error + self.d_gain * (error - self.prev_error)
        self.prev_error = error
        return control

# --- 유틸리티 함수 (unchanged) ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """월드 좌표계의 목표 지점을 에이전트 중심의 자기 좌표계로 변환"""
    vec_in_world = global_target_pos - agent_pos
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])
    
    ego_vector = rotation_matrix @ vec_in_world
    return ego_vector

def extract_sensor_data(obs):
    """관찰에서 센서 데이터 추출"""
    if 'image' in obs:
        rgb_data = obs['image'][..., -1]
        rgb_data = (rgb_data * 255).astype(np.uint8)
    else:
        rgb_data = None
    
    # depth 1 => 3채널로 확장 (PerceptNet expects 3 channels)
    depth_data = obs["depth"][..., -1]
    depth_data = np.concatenate([depth_data,depth_data,depth_data], axis=-1)
    
    semantic_data = obs["semantic"][..., -1]
    
    return rgb_data, depth_data, semantic_data

import matplotlib.pyplot as plt 

def create_and_save_plots(returns, actor_losses, critic_losses):
    """학습 결과 그래프 생성 및 저장"""
    
    # Returns 그래프
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    return_means = [np.mean(r) for r in returns]
    return_stds = [np.std(r) for r in returns]
    
    plt.plot(return_means, label="Mean Return", color='blue')
    plt.fill_between(range(len(return_means)),  
                     np.array(return_means) - np.array(return_stds),  
                     np.array(return_means) + np.array(return_stds),  
                     alpha=0.3, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.title("Training Returns")
    plt.legend()
    plt.grid(True)
    
    # Actor Loss 그래프
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses, label="Actor Loss", color='green', alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.legend()
    plt.grid(True)
    
    # Critic Loss 그래프
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses, label="Critic Loss", color='orange', alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metaurban_training_results_efficientnet.png', dpi=300)
    
    # wandb.log({"training_plots": wandb.Image(plt)})
    plt.close()