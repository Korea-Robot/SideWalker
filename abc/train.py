import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import typing
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import wandb
import torchvision.models as models
import transformers
from transformers import SegformerForSemanticSegmentation

 
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

# Import configurations and utilities
from env_config import EnvConfig
from config import Config
from utils import convert_to_egocentric, extract_sensor_data, create_and_save_plots

# Import PPO model
from perceptnet import PerceptNet
from ppo_model import Actor,Critic


class NNPolicy:
    """Neural Network Policy"""
    def __init__(self, actor: Actor):
        self.actor = actor
        
    def __call__(self, obs_data: dict) -> tuple[float, float]:
        device = deviceof(self.actor)
        
        # Convert observation to tensors
        rgb = torch.tensor(obs_data['rgb'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth = torch.tensor(obs_data['depth'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        semantic = torch.tensor(obs_data['semantic'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        goal = torch.tensor(obs_data['goal'], dtype=torch.float32).unsqueeze(0)
        
        rgb = rgb.to(device)
        depth = depth.to(device)
        semantic = semantic.to(device)
        goal = goal.to(device)
        
        with torch.no_grad():
            self.actor.eval()
            action_dist = self.actor(rgb, semantic, depth, goal)
            action = action_dist.sample()
            self.actor.train()
        
        return tuple(action.cpu().numpy()[0])


# Load configurations
env_config = EnvConfig()


def collect_trajectory(env, policy: typing.Callable, max_steps: int = 512) -> tuple[list[dict], list[tuple[float, float]], list[float]]:
    """환경에서 trajectory 수집"""
    observations = []
    actions = []
    rewards = []
    
    obs, info = env.reset()
    waypoints = env.agent.navigation.checkpoints
    
    # 일정 거리 이상의 waypoints가 없을 경우, 환경을 재설정
    while len(waypoints) < 31:
        obs, info = env.reset()
        waypoints = env.agent.navigation.checkpoints
        
    step_count = 0
    
    while step_count < max_steps:
        # 목표 지점 계산
        ego_goal_position = np.array([0.0, 0.0])
        nav = env.agent.navigation
        waypoints = nav.checkpoints
        
        k = 15
        if len(waypoints) > k:
            global_target = waypoints[k]
            agent_pos = env.agent.position
            agent_heading = env.agent.heading_theta
            ego_goal_position = convert_to_egocentric(global_target, agent_pos, agent_heading)
        
        # 관찰 데이터 준비
        rgb_data, depth_data, semantic_data = extract_sensor_data(obs)
        
        obs_data = {
            'rgb': rgb_data,
            'depth': depth_data,
            'semantic': semantic_data,
            'goal': ego_goal_position
        }
        observations.append(obs_data)
        
        # 행동 선택
        action = policy(obs_data)
        actions.append(action)
        
        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    return observations, actions, rewards


def deviceof(m: nn.Module) -> torch.device:
    """모듈의 device 반환"""
    return next(m.parameters()).device


def obs_batch_to_tensor(obs_batch: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """관찰 배치를 텐서로 변환"""
    rgb_batch = []
    depth_batch = []
    semantic_batch = []
    goal_batch = []
    
    for obs in obs_batch:
        # RGB: (H, W, 3) -> (3, H, W)
        rgb = torch.tensor(obs['rgb'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        rgb_batch.append(rgb)
        
        # Depth: (H, W, 3) -> (3, H, W)
        depth = torch.tensor(obs['depth'], dtype=torch.float32).permute(2, 0, 1)
        depth_batch.append(depth)
        
        # semantic: (H, W, 3) -> (3, H, W)
        semantic = torch.tensor(obs['semantic'], dtype=torch.float32).permute(2, 0, 1) / 255.0
        semantic_batch.append(semantic)
        
        # Goal: (2,)
        goal = torch.tensor(obs['goal'], dtype=torch.float32)
        goal_batch.append(goal)
    
    rgb_tensor = torch.stack(rgb_batch).to(device)
    depth_tensor = torch.stack(depth_batch).to(device)
    semantic_tensor = torch.stack(semantic_batch).to(device)
    goal_tensor = torch.stack(goal_batch).to(device)
    
    return rgb_tensor, depth_tensor, semantic_tensor, goal_tensor


def rewards_to_go(trajectory_rewards: list[float], gamma: float) -> list[float]:
    """감마 할인된 reward-to-go 계산"""
    trajectory_len = len(trajectory_rewards)
    v_batch = np.zeros(trajectory_len)
    v_batch[-1] = trajectory_rewards[-1]
    
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + gamma * v_batch[t + 1]
    
    return list(v_batch)


def compute_advantage(
    critic: Critic,
    trajectory_observations: list[dict],
    trajectory_rewards: list[float],
    gamma: float
) -> list[float]:
    """GAE를 사용한 advantage 계산"""
    trajectory_len = len(trajectory_rewards)
    
    # Value 계산
    critic.eval()
    with torch.no_grad():
        rgb_tensor, depth_tensor, semantic_tensor, goal_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        obs_values = critic.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor).detach().cpu().numpy()
    critic.train()
    
    # Advantage = Reward-to-go - Value
    trajectory_advantages = np.array(rewards_to_go(trajectory_rewards, gamma)) - obs_values
    
    return list(trajectory_advantages)


@dataclass
class PPOConfig:
    ppo_eps: float
    ppo_grad_descent_steps: int


def compute_ppo_loss(
    pi_thetak_given_st: torch.distributions.MultivariateNormal,
    pi_theta_given_st: torch.distributions.MultivariateNormal,
    a_t: torch.Tensor,
    A_pi_thetak_given_st_at: torch.Tensor,
    config: PPOConfig
) -> torch.Tensor:
    """PPO 클립 손실 계산"""
    # Detach the old policy probabilities from the computation graph
    log_prob_thetak = pi_thetak_given_st.log_prob(a_t).detach()
    likelihood_ratio = torch.exp(pi_theta_given_st.log_prob(a_t) - log_prob_thetak)
    
    ppo_loss_per_example = -torch.minimum(
        likelihood_ratio * A_pi_thetak_given_st_at,
        torch.clip(likelihood_ratio, 1 - config.ppo_eps, 1 + config.ppo_eps) * A_pi_thetak_given_st_at,
    )
    
    return ppo_loss_per_example.mean()


def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[dict],
    action_batch: list[tuple[float, float]],
    advantage_batch: list[float],
    reward_to_go_batch: list[float],
    config: PPOConfig
) -> tuple[list[float], list[float]]:
    """PPO 학습 함수"""
    device = deviceof(critic)
    
    # 데이터를 텐서로 변환
    rgb_tensor, depth_tensor, semantic_tensor, goal_tensor = obs_batch_to_tensor(observation_batch, device)
    true_value_batch_tensor = torch.tensor(reward_to_go_batch, dtype=torch.float32, device=device)
    chosen_action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=device)
    advantage_batch_tensor = torch.tensor(advantage_batch, dtype=torch.float32, device=device)
    
    # Normalize advantages
    advantage_batch_tensor = (advantage_batch_tensor - advantage_batch_tensor.mean()) / (advantage_batch_tensor.std() + 1e-8)
    
    # 이전 정책의 행동 확률
    with torch.no_grad():
        old_policy_action_dist = actor.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
    
    # Actor and Critic 학습
    actor_losses = []
    critic_losses = []
    for _ in range(config.ppo_grad_descent_steps):
        # Critic Update
        critic_optimizer.zero_grad()
        pred_value_batch_tensor = critic.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
        critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)

        
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))

        # Actor Update
        actor_optimizer.zero_grad()
        current_policy_action_dist = actor.forward(rgb_tensor, semantic_tensor, depth_tensor, goal_tensor)
        actor_loss = compute_ppo_loss(
            old_policy_action_dist,
            current_policy_action_dist,
            chosen_action_tensor,
            advantage_batch_tensor,
            config
        )
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(float(actor_loss.item()))
    
        # breakpoint()
    return actor_losses, critic_losses


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """학습률 설정"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    """메인 학습 함수"""
    
    config = Config()

    # episodes_per_batch 값을 16으로 변경하고 싶다면 여기서 수정할 수 있습니다.
    # config.episodes_per_batch = 16 # 현재는 8

    # 2. 생성된 config 객체를 wandb.init에 직접 전달
    wandb.init(
        project="metaurban-rl-multimodal",
        config=config  # 객체를 그대로 전달
    )
    
    # 디바이스 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 네트워크 초기화
    actor = Actor(hidden_dim=config.hidden_dim).to(device)
    critic = Critic(hidden_dim=config.hidden_dim).to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in list(actor.parameters()) + list(critic.parameters()))
    trainable_params = sum(p.numel() for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 옵티마이저
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    
    # 정책
    policy = NNPolicy(actor)
    
    # 환경 초기화
    env = SidewalkStaticMetaUrbanEnv(env_config.base_env_cfg)
    
    # PPO 설정
    ppo_config = PPOConfig(
        ppo_eps=config.ppo_eps,
        ppo_grad_descent_steps=config.ppo_grad_descent_steps,
    )
    
    # 학습 통계
    returns = []
    all_actor_losses = []
    all_critic_losses = []
    
    # 학습 루프
    for epoch in range(config.train_epochs):
        obs_batch = [] # len : 500
        
        # (Pdb) obs_batch[0].keys()
        # dict_keys(['rgb', 'depth', 'semantic', 'goal'])
        # (Pdb) obs_batch[0]['rgb'].shape
        # (120, 160, 3)
        # (Pdb) obs_batch[0]['depth'].shape
        # (120, 160, 3)
        # (Pdb) obs_batch[0]['semantic'].shape
        # (120, 160, 3)
        # (Pdb) obs_batch[0]['goal'].shape
        # (2,)

        act_batch = []
        # (Pdb) act_batch[0]
        # (1.5369167, 0.24249047)
        
        rtg_batch = []
        # 아래 reward를 보면 문제가 많음. 왜냐하면 앞으로 가지않기에  sparse reward가 계  다..된속ㄷ  
        # (Pdb) rtg_batch
        # [0.14372963352920956, 0.1447059935959939, 0.14327540759853047, 0.13617005414370117, 0.12054448197991, 0.11892798330029737, 0.12012927606090644, 0.12134270309182468, 0.12256838696143907, 0.12380645147620108, 0.1250570216931324, 0.12632022393245698, 0.12759618579036058, 0.12888503615187938, 0.13018690520391857, 0.13150192444840259, 0.13283022671555816, 0.13417194617733147, 0.1355272183609409, 0.13689618016256655, 0.1271156747524657, 0.12839967146713707, 0.12969663784559302, 0.1310067048945384, 0.13233000494397817, 0.13366667166058402, 0.135016840061196, 0.1363806465264606, 0.13775822881460667, 0.13914972607536027, 0.14055527886400027, 0.14197502915555582, 0.13858735815039058, 0.13551101696774503, 0.13687981511893438, 0.13826243951407513, 0.1396590298121971, 0.14106972708302737, 0.1399424714139696, 0.1289687089188863, 0.10570255738009039, 0.09240655640239975, 0.077578826293454, 0.06069855772239231, 0.04998068855738037, 0.03617619178071821, 0.02520125635836621, 0.019740861534414594, 0.007949041613736507, 0.004421549615441164, 0.004466211732768852, 0.0045113249825948, 0.00455689392181293, 0.004602923153346394, 0.0046494173266125195, 0.004696381137992444, 0.004743819331305499, 0.004791736698288383, 0.004840138079079175, 0.0048890283627062376, 0.004938412487582059, 0.004988295442002079, 0.005038682264648565, 0.005089578045099561, 0.0051409879243429915, 0.005192917095295951, 0.0052453708033292435, 0.005298354346797216, 0.0053518730775729455, 0.005405932401588834, 0.0054605377793826605, 0.005515694726649152, 0.005571408814797124, 0.005627685671512246, 0.005684530981325501, 0.005741950486187375, 0.005799949986047854, 0.005858535339442277, 0.005917712464083108, 0.005977487337457685, 0.006037865997432005, 0.006098854542860611, 0.006160459134202638, 0.006222685994144079, 0.006285541408226343, 0.006349031725481154, 0.006413163359071873, 0.006477942786941286, 0.0006688728060637106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        adv_batch = []
        # 마찬가지로  advantage도 일정..
        # Pdb) adv_batch
        # [0.32044524594070856, 0.3192408806714181, 0.32060786847881, 0.3148253060043198, 0.2965223379007016, 0.2987292387648512, 0.2978841252563868, 0.2967349731800205, 0.29867007153175157, 0.29951335958300496, 0.3013153472699146, 0.3016757179548264, 0.3059234034884578, 0.30450580064582167, 0.30781908804045116, 0.30879528468170947, 0.31049364238595306, 0.3117229325865188, 0.3139165440988941, 0.3144945671655115, 0.3059738953297057, 0.30369404092628927, 0.3076663617854887, 0.30560435566091104, 0.3113256887332314, 0.3087224735290227, 0.31229956086541044, 0.31308675199711794, 0.31582636002302644, 0.3147651857559175, 0.3193061067725162, 0.3188220102041398, 0.3191736663912811, 0.31284252417370817, 0.31550624813380407, 0.31539492681113324, 0.3167286640113396, 0.3187170196970105, 0.3201998514226671, 0.30630601267655383, 0.28459955077875676, 0.26783759931202195, 0.25734911714504705, 0.23701040827018283, 0.2287452702376782, 0.2137383987642845, 0.2040689093706419, 0.19616914975577385, 0.18492540944496666, 0.18158997851329883, 0.18114887767686827, 0.1861637815045827, 0.1860457207927312, 0.17954145288318954, 0.1788980828978253, 0.1795283377629772, 0.17925826201357173, 0.1817709208489842, 0.1866151417507253, 0.1859339198393374, 0.18860517817743191, 0.18469916046275403, 0.18755784992631544, 0.18329916597673257, 0.18467770307422246, 0.1833554024563952, 0.18336002343699623, 0.18009523363833163, 0.18378716889780916, 0.18151895675556984, 0.18356911073928256, 0.18259421001786313, 0.18370456869066687, 0.18270737815446056, 0.18280867362811506, 0.1811695065240902, 0.18328556500107776, 0.18148236947259047, 0.18318888486837234, 0.18181789494739725, 0.18353266012775732, 0.18337266445268116, 0.1840336533518845, 0.1809570698156315, 0.18379663740889773, 0.18251571516092122, 0.18501851123085228, 0.18425614210201238, 0.17933373591565233, 0.17666691541671753, 0.17742833495140076, 0.17486581206321716, 0.17728087306022644, 0.1755128800868988, 0.17924603819847107, 0.17536717653274536, 0.1770990937948227, 0.17690598964691162, 0.17699584364891052, 0.1759587824344635, 0.17748036980628967, 0.176483154296875, 0.17739035189151764, 0.17690113186836243, 0.17860254645347595, 0.17812125384807587, 0.1785590946674347, 0.1787380576133728, 0.17766335606575012, 0.18123289942741394, 0.18155379593372345, 0.18149495124816895, 0.1831580400466919, 0.18141932785511017, 0.18194866180419922, 0.18357813358306885, 0.183519184589386, 0.1821843385696411, 0.18373987078666687, 0.17930513620376587, 0.18101029098033905, 0.17852413654327393, 0.1810629963874817, 0.17875811457633972, 0.17708809673786163, 0.17425793409347534, 0.1747777760028839, 0.170608788728714, 0.17114128172397614, 0.16988137364387512, 0.17452913522720337, 0.1680467575788498, 0.1709112524986267, 0.17047756910324097, 0.17301803827285767, 0.17161783576011658, 0.17412742972373962, 0.17555256187915802, 0.18435101211071014, 0.18062745034694672, 0.19046975672245026, 0.18505337834358215, 0.1899777054786682, 0.18502545356750488, 0.18874040246009827, 0.1854998767375946, 0.18675357103347778, 0.1852245330810547, 0.18748091161251068, 0.18435215950012207, 0.187961146235466, 0.1886352300643921, 0.1875624805688858, 0.18436139822006226, 0.18816888332366943, 0.18730026483535767, 0.1897231638431549, 0.18567678332328796, 0.189679816365242, 0.18212401866912842, 0.18354551494121552, 0.1820414960384369, 0.18715205788612366, 0.18738573789596558, 0.18965303897857666, 0.18659739196300507, 0.18941497802734375, 0.18699663877487183, 0.18876513838768005, 0.18801037967205048, 0.18913596868515015, 0.1864205151796341, 0.191133052110672, 0.19116488099098206, 0.18877196311950684, 0.18831253051757812, 0.18881568312644958, 0.18828585743904114, 0.1901599019765854, 0.19094102084636688, 0.1903713047504425, 0.19077365100383759, 0.1898825764656067, 0.1900455355644226, 0.18921314179897308, 0.18865618109703064, 0.18920737504959106, 0.1858777552843094, 0.18943151831626892, 0.18694600462913513, 0.18910722434520721, 0.1863904893398285, 0.18853479623794556, 0.1883002519607544, 0.18995997309684753, 0.18777844309806824, 0.1896866261959076, 0.18578363955020905, 0.18978267908096313, 0.18802379071712494, 0.1911953091621399, 0.19156962633132935, 0.19020496308803558, 0.19129322469234467, 0.18930134177207947, 0.1882254183292389, 0.18973401188850403, 0.18758806586265564, 0.18896278738975525, 0.18882343173027039, 0.18874366581439972, 0.18924814462661743, 0.19061064720153809, 0.19091038405895233, 0.19097647070884705, 0.18907052278518677, 0.18947651982307434, 0.18920135498046875, 0.1905149221420288, 0.19004036486148834, 0.191048264503479, 0.1894773691892624, 0.18887613713741302, 0.1889166533946991, 0.1832621693611145, 0.18809758126735687, 0.18001022934913635, 0.1835247129201889, 0.1812121421098709, 0.1752982884645462, 0.17723286151885986, 0.17138724029064178, 0.1744922399520874, 0.17479273676872253, 0.17494790256023407, 0.1735357791185379, 0.17487835884094238, 0.17090731859207153, 0.17357635498046875, 0.1713283360004425, 0.1740838587284088, 0.17370489239692688, 0.17433862388134003, 0.17333891987800598, 0.1722404956817627, 0.17147010564804077, 0.17381799221038818, 0.17038115859031677, 0.170731782913208, 0.1702667474746704, 0.16951727867126465, 0.171343132853508, 0.17369811236858368, 0.17012900114059448, 0.17664311826229095, 0.17466595768928528, 0.17405912280082703, 0.17159724235534668, 0.17455889284610748, 0.17342519760131836, 0.1766146719455719, 0.17397581040859222, 0.1741095781326294, 0.17137737572193146, 0.17280155420303345, 0.171362966299057, 0.17432577908039093, 0.1743433177471161, 0.17361286282539368, 0.1703837811946869, 0.17277950048446655, 0.17174318432807922, 0.17197686433792114, 0.1739753782749176, 0.17543897032737732, 0.17821697890758514, 0.17791831493377686, 0.17681600153446198, 0.17411281168460846, 0.17064997553825378, 0.1735866367816925, 0.17579203844070435, 0.17406553030014038, 0.17145958542823792, 0.1697569489479065, 0.16870291531085968, 0.16952499747276306, 0.1716107726097107, 0.1729261726140976, 0.1758040338754654, 0.17647096514701843, 0.16942265629768372, 0.17159916460514069, 0.17445212602615356, 0.17413294315338135, 0.1699945330619812, 0.17340250313282013, 0.17050446569919586, 0.17215247452259064, 0.16946904361248016, 0.1728280484676361, 0.17565906047821045, 0.1760549545288086, 0.17439469695091248, 0.17375659942626953, 0.1754852831363678, 0.17708849906921387, 0.1758149117231369, 0.17648382484912872, 0.1755966991186142, 0.1738608479499817, 0.17496919631958008, 0.17129212617874146, 0.16979700326919556, 0.17154696583747864, 0.17166808247566223, 0.17517831921577454, 0.17578232288360596, 0.1718161702156067, 0.17023113369941711, 0.17267313599586487, 0.16944503784179688, 0.1710667908191681, 0.1708945780992508, 0.1739710122346878, 0.17352230846881866, 0.1725989133119583, 0.17522816359996796, 0.17285829782485962, 0.17158597707748413, 0.17593489587306976, 0.17211788892745972, 0.17117704451084137, 0.16826368868350983, 0.17005395889282227, 0.16910193860530853, 0.17002414166927338, 0.16784478724002838, 0.1703214943408966, 0.17060238122940063, 0.17296719551086426, 0.16907232999801636, 0.17378348112106323, 0.17038893699645996, 0.17250442504882812, 0.17287541925907135, 0.1725378781557083, 0.17224028706550598, 0.17172089219093323, 0.17160728573799133, 0.17081359028816223, 0.17048075795173645, 0.17147350311279297, 0.17080608010292053, 0.17093199491500854, 0.1700582504272461, 0.1706024408340454, 0.1710614264011383, 0.16920238733291626, 0.1684827208518982, 0.17088162899017334, 0.1722640097141266, 0.17107979953289032, 0.16877757012844086, 0.1689317524433136, 0.1699897050857544, 0.17080536484718323, 0.17102544009685516, 0.17297491431236267, 0.1711609661579132, 0.16978269815444946, 0.17086689174175262, 0.17332857847213745, 0.17065200209617615, 0.1726103276014328, 0.17074444890022278, 0.17257677018642426, 0.16954410076141357, 0.17127487063407898, 0.16854915022850037, 0.17442817986011505, 0.17032217979431152, 0.16971486806869507, 0.17385554313659668, 0.17258286476135254, 0.17270216345787048, 0.17305296659469604, 0.17561504244804382, 0.18096670508384705, 0.18113946914672852, 0.17958667874336243, 0.18127217888832092, 0.17794257402420044, 0.17939677834510803, 0.17715874314308167, 0.1780633181333542, 0.17770160734653473, 0.17930495738983154, 0.1779543161392212, 0.1794564127922058, 0.17808908224105835, 0.17736902832984924, 0.17718522250652313, 0.17554008960723877, 0.17554591596126556, 0.17730450630187988, 0.17779749631881714, 0.17683975398540497, 0.17661474645137787, 0.1767215132713318, 0.17637813091278076, 0.17711535096168518, 0.17671465873718262, 0.1771451085805893, 0.17506197094917297, 0.1741904318332672, 0.17587806284427643, 0.17712301015853882, 0.17729675769805908, 0.1747320294380188, 0.17635321617126465, 0.178166925907135, 0.17747575044631958, 0.17754849791526794, 0.17929625511169434, 0.17832893133163452, 0.17807888984680176, 0.1753074824810028, 0.17729438841342926, 0.17508214712142944, 0.17934846878051758, 0.17582625150680542, 0.1773822009563446, 0.1756592094898224, 0.17784376442432404, 0.17693984508514404, 0.17836377024650574, 0.17673206329345703, 0.17549780011177063, 0.17509645223617554, 0.17628192901611328, 0.17442095279693604, 0.17796021699905396, 0.17887739837169647, 0.17757397890090942, 0.17592889070510864, 0.17764095962047577, 0.1749846488237381, 0.178951233625412, 0.17679299414157867, 0.1751364767551422, 0.17561545968055725, 0.17461761832237244, 0.1745341271162033, 0.17445224523544312, 0.1752588152885437, 0.1765938252210617, 0.17663058638572693, 0.17902575433254242, 0.1791306436061859, 0.1761968731880188, 0.1752125769853592, 0.17355364561080933, 0.17539069056510925, 0.17344969511032104, 0.17546376585960388, 0.17602971196174622, 0.17812030017375946, 0.1781407594680786, 0.17872166633605957, 0.18090739846229553, 0.17832297086715698, 0.17480862140655518, 0.174492746591568, 0.17267876863479614, 0.17236191034317017, 0.17371948063373566, 0.17519629001617432, 0.17363828420639038, 0.17763787508010864, 0.17473992705345154, 0.1797391176223755, 0.17517636716365814, 0.17662113904953003, 0.17461593449115753, 0.1778431534767151, 0.17610377073287964, 0.1760459840297699, 0.17635062336921692, 0.17494091391563416, 0.17525428533554077, 0.1751244217157364, 0.17401614785194397, 0.17305558919906616, 0.176130473613739, 0.1758631318807602, 0.1744435429573059, 0.17449817061424255, 0.17541366815567017, 0.17543958127498627]

        trajectory_returns = []
        
        # 배치 수집
        for episode in range(config.episodes_per_batch):
            print(f"Epoch {epoch}/{config.train_epochs}, Collecting Episode {episode+1}/{config.episodes_per_batch}...")
            obs_traj, act_traj, rew_traj = collect_trajectory(env, policy,max_steps=config.max_steps)
            rtg_traj = rewards_to_go(rew_traj, config.gamma)
            adv_traj = compute_advantage(critic, obs_traj, rew_traj, config.gamma)
            
            obs_batch.extend(obs_traj)
            act_batch.extend(act_traj)
            rtg_batch.extend(rtg_traj)
            adv_batch.extend(adv_traj)
            trajectory_returns.append(sum(rew_traj))

        print('action batch')
        print(act_batch)
        print('reward batch')
        print(rtg_batch)
        # PPO 업데이트
        batch_actor_losses, batch_critic_losses = train_ppo(
            actor, critic, actor_optimizer, critic_optimizer,
            obs_batch, act_batch, adv_batch, rtg_batch, ppo_config
        )
        
        # 통계 수집
        returns.append(trajectory_returns)
        all_actor_losses.extend(batch_actor_losses)
        all_critic_losses.extend(batch_critic_losses)
        
        # 로깅
        avg_return = np.mean(trajectory_returns)
        std_return = np.std(trajectory_returns)
        median_return = np.median(trajectory_returns)
        
        print(f"Epoch {epoch}, Avg Returns: {avg_return:.3f} +/- {std_return:.3f}, "
              f"Median: {median_return:.3f}, Actor Loss: {np.mean(batch_actor_losses):.3f}, "
              f"Critic Loss: {np.mean(batch_critic_losses):.3f}")
        
        # WandB 로깅 (선택사항)
        wandb.log({
            "epoch": epoch,
            "avg_return": avg_return,
            "std_return": std_return,
            "median_return": median_return,
            "actor_loss": np.mean(batch_actor_losses),
            "critic_loss": np.mean(batch_critic_losses),
        })
        
        # 주기적으로 모델 저장
        if epoch % 10 == 0: # and epoch > 0:
            torch.save(actor.state_dict(), f'checkpoints/metaurban_actor_epoch_{epoch}.pt')
            torch.save(critic.state_dict(), f'checkpoints/metaurban_critic_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
    
    # 최종 모델 저장
    torch.save(actor.state_dict(), 'metaurban_actor_multimodal_final.pt')
    torch.save(critic.state_dict(), 'metaurban_critic_multimodal_final.pt')
    print("Training completed! Final models saved.")
    
    # 그래프 생성 및 저장
    create_and_save_plots(returns, all_actor_losses, all_critic_losses)
    
    env.close()
    wandb.finish()  # WandB 사용시 주석 해제


if __name__ == "__main__":
    main()