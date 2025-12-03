from collections import defaultdict
import gymnasium as gym
import numpy as np


# Q-Learning Agent - only tested in FrozenLake environment
class QLearningAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 0.1,
            initial_eps: float = 1.0,
            eps_decay: float = 0.99,
            final_eps: float = 0.1,
            gamma: float = 0.95,
            ):
        """Initialize the Q-learning agent with the given parameters.
        Args:
            env (gym.Env): The environment in which the agent operates.
            learning_rate (float): The rate at which the agent learns.
            initial_eps (float): The initial exploration rate epsilon.
            eps_decay (float): The decay rate of epsilon after each episode.
            final_eps (float): The minimum exploration rate epsilon.
            gamma (float): The discount factor for future rewards.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.eps = initial_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps
        self.gamma = gamma

        # Action-value function maps the expected reward for a given action performed in a given state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.training_error = []  # To track training error over time

    def choose_action(self, state: int) -> int:
        """Choose an action based on the eps-greedy policy.
        Args:
            state (np.ndarray): The current state state.
        Returns:
            int: The action to be taken.
        """
        if np.random.rand() < self.eps:
            return self.env.action_space.sample()  # Explore: return a random action
        else:
            return int(np.argmax(self.q_table[state]))  # Exploit: return the action with max value (greedy action)

    def update_q_value(
            self,
            state: int,
            action: int,
            reward: float,
            terminated: bool,
            next_state: int,
            ) -> None:
        """Update the Q-value for the given state and action.
        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            next_state (int): The next state after taking the action.
        """

        # Bellman Equation - estimate of optimal future value
        next_q_value = (not terminated) * np.max(self.q_table[next_state]) # If terminated, next_q_value is 0
        target = reward + self.gamma * next_q_value

        # Temporal Difference - the difference between our estimate and the current Q-value
        temporal_difference = target - self.q_table[state][action]
        self.training_error.append(abs(temporal_difference))

        # Update Q-value using the learning rate and temporal difference (error of our estimate)
        if state == 15:  # Goal state in FrozenLake
            print("-"*50)
            print(self.q_table[state][action])
        self.q_table[state][action] = (
            self.q_table[state][action] + self.learning_rate * temporal_difference
        )

        if state == 15:  # Goal state in FrozenLake
            print(self.q_table[state][action])

    def decay_eps(self) -> None:
        """Decay the exploration rate eps after each episode."""
        self.eps = max(self.final_eps, self.eps * self.eps_decay)
