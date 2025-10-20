# config.py
from dataclasses import dataclass

@dataclass
class Config:
    train_epochs: int           = 200
    episodes_per_batch: int     = 8 # 16
    gamma: float                = 0.99
    ppo_eps: float              = 0.2
    ppo_grad_descent_steps: int = 10
    actor_lr: float             = 3e-4
    critic_lr: float            = 1e-3
    hidden_dim: int             = 512
    max_steps: int              = 256

# 미리 인스턴스 하나 만들어 둘 수도 있고
# config = Config()
