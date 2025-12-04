from __future__ import annotations

from tqdm import tqdm
import gymnasium as gym
import abc
from typing import Any, Dict, Optional, Union
import numpy as np

class Agent(abc.ABC):
    def __init__(self, env: Union[gym.Env, str], learning_rate: float):
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        self.learning_rate = learning_rate
        self.metrics: Dict[str, list] = {}

    @abc.abstractmethod
    def act(self, state: Any) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError