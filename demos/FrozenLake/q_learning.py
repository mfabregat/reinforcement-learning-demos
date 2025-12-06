import gymnasium as gym
from tqdm import tqdm
import numpy as np
import pickle
import os

import sys
print(sys.path)


from rl_agents.tabular_q_learning import TabularQLearningAgent

n_episodes = 50_000

env = gym.make("FrozenLake-v1", is_slippery=True)

learning_rate = 0.1
initial_epsilon = 1.0
final_epsilon = 0.00001
# Decay epsilon over 80% of training episodes
epsilon_decay = (final_epsilon / initial_epsilon) ** (1 / (n_episodes * 0.8))

agent = TabularQLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    final_epsilon=final_epsilon,
    epsilon_decay=epsilon_decay,
    gamma=0.99,
)

agent.train(n_episodes=n_episodes)
# agent.save("taxi_q_learning_agent.npz")

# agent = TabularQLearningAgent.load("taxi_q_learning_agent.npz", env=env)

agent.plot_metrics()