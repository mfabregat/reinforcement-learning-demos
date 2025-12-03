from collections import defaultdict
import gymnasium as gym
import numpy as np


class QLearningFrozenLakeAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 0.1,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.99,
            final_epsilon: float = 0.1,
            gamma: float = 0.95,
            ):
        """Initialize the Q-learning agent with the given parameters.
        Args:
            env (gym.Env): The environment in which the agent operates.
            learning_rate (float): The rate at which the agent learns.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The decay rate of epsilon after each episode.
            final_epsilon (float): The minimum exploration rate.
            gamma (float): The discount factor for future rewards.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma

        # Action-value function maps the expected reward for a given action performed in a given state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.error = []

    def choose_action(self, observation: np.ndarray) -> int:
        """Choose an action based on the epsilon-greedy policy.
        Args:
            observation (np.ndarray): The current state observation.
        Returns:
            int: The action to be taken.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore: return a random action
        else:
            return int(np.argmax(self.q_table[observation]))  # Exploit: return the action with max value (greedy action)
