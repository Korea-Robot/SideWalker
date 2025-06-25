import torch
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform

class DummyPolicy:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def forward(self, images, goal_vec):
        # dummy mean/std: just zeros and ones
        batch = images.size(0)
        mean = torch.zeros(batch, self.action_dim)
        std  = torch.ones(batch, self.action_dim) * 0.5
        return mean, std

    def get_mixture_action_log_prob_mean_std(
        self,
        images: torch.Tensor,      # (batch, C, H, W)
        goal_vec: torch.Tensor,    # (batch, goal_dim)
        num_samples: int = 1,
        action: torch.Tensor = None
    ):
        # 1) policy mean/std
        policy_mean, policy_std = self.forward(images, goal_vec)  # (batch, action_dim)
        batch, action_dim = policy_mean.shape

        # 2) define tanh-squashed Gaussian
        tanh = [TanhTransform(cache_size=1)]
        policy_dist = TransformedDistribution(
            Normal(policy_mean, policy_std),
            tanh
        )

        # 3) sample & log_prob
        if num_samples == 1:
            action_tanh = policy_dist.rsample()                  # (batch, action_dim)
            log_prob    = policy_dist.log_prob(action_tanh).sum(-1)  # (batch,)
        else:
            action_tanh = policy_dist.rsample((num_samples,))     # (num_s, batch, action_dim)
            log_prob    = policy_dist.log_prob(action_tanh).sum(-1) # (num_s, batch)

        # 4) approximate mean & std after tanh
        mixture_mean = policy_mean.tanh()
        mixture_std  = policy_std * (1 - mixture_mean.pow(2))

        return action_tanh, log_prob, mixture_mean, mixture_std

# --------------------------
# 테스트 코드: dummy tensors 로 실행해 보기
# --------------------------
if __name__ == "__main__":
    batch      = 4
    C, H, W    = 3, 64, 64
    goal_dim   = 2
    num_samps  = 3

    # dummy inputs
    images   = torch.randn(batch, C, H, W)
    goal_vec = torch.randn(batch, goal_dim)

    policy = DummyPolicy(action_dim=2)
    actions, log_probs, means, stds = policy.get_mixture_action_log_prob_mean_std(
        images, goal_vec, num_samples=num_samps
    )

    print("actions.shape:", actions.shape)    # → (num_samps, batch, action_dim)
    print(actions)
    print("log_probs.shape:", log_probs.shape)  # → (num_samps, batch)
    print(log_probs)
    print("mixture_mean.shape:", means.shape)   # → (batch, action_dim)
    print(means)
    print("mixture_std.shape:", stds.shape)     # → (batch, action_dim)
    print(stds)
