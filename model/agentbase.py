
# model/abstract.py
from abc import ABC, abstractmethod

class AgentBase(ABC):
    @abstractmethod
    def __init__(self,act_dim, config):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def action(self, obs, deterministic=False):
        pass

    @abstractmethod
    def learn(self, batch):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
