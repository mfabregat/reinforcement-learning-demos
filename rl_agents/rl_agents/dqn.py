from typing import Union
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .agent import Agent



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class DQNAgent(Agent):
     def __init__(
        self,
        env: Union[gym.Env, str],
        batch_size: int,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.9,
        epsilon_decay: float = 0.99, # 2500
        final_epsilon: float = 0.01,
        gamma: float = 0.99,
        tau: float = 0.005,
        ):
        super().__init__(env, learning_rate)
        self.eps = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma

        self.dqn = DQN(self.env.observation_space.shape[0], self.env.action_space.n)