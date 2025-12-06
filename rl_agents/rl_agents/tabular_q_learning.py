from collections import defaultdict

from tqdm import tqdm
import gymnasium as gym
import numpy as np
from typing import Union

from .agent import Agent

class TabularQLearningAgent(Agent):
    def __init__(
            self,
            env: Union[gym.Env, str],
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
            initial_epsilon (float): The initial exploration rate epsilon.
            epsilon_decay (float): The decay rate of epsilon after each episode.
            final_epsilon (float): The minimum exploration rate epsilon.
            gamma (float): The discount factor for future rewards.
        """
        super().__init__(env, learning_rate)
        self.eps = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma

        # Action-value function maps the expected reward for a given action performed in a given state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))


    def act(self, state: int) -> int:
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

    def update(
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
        self.metrics["training_error"].append(abs(temporal_difference))

        # Update Q-value using the learning rate and temporal difference (error of our estimate)
        self.q_table[state][action] = (
            self.q_table[state][action] + self.learning_rate * temporal_difference
        )

    def train(self, n_episodes: int, max_steps: Union[int, None] = 100) -> None:
        if max_steps is None:
            max_steps = np.inf

        for episode in tqdm(range(n_episodes)):
            # Start a new episode
            obs, info = self.env.reset()
            done = False

            # Play one complete episode
            episode_reward = 0
            episode_length = 0
            while not done and episode_length < max_steps:
                # Agent chooses action (initially random, gradually more intelligent)
                action = self.act(obs)

                # Take action and observe result
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                # Learn from this experience
                self.update(obs, action, reward, terminated, next_obs)

                # Move to next state
                done = terminated or truncated
                obs = next_obs

            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(episode_length)

            self.decay_eps()

    def decay_eps(self) -> None:
        """Decay the exploration rate eps after each episode."""
        self.eps = max(self.final_epsilon, self.eps * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save the agent to a file.
        Args:
            filepath (str): The path to the file where the agent will be saved.
        """
        agent_data = {
            "q_table": dict(self.q_table),
            "learning_rate": self.learning_rate,
            "eps": self.eps,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "gamma": self.gamma,
            "metrics": self.metrics,
        }
        np.savez_compressed(filepath, **agent_data)

    @classmethod
    def load(cls, filepath: str, env: Union[gym.Env, str]):
        """Load an agent from a file.
        Args:
            filepath (str): The path to the file from which the agent will be loaded.
            env (gym.Env): The environment in which the agent will operate.
        Returns:
            QLearningAgent: The loaded agent.
        """
        data = np.load(filepath, allow_pickle=True)
        agent = cls(
            env=env,
            learning_rate=float(data["learning_rate"]),
            initial_epsilon=float(data["eps"]),
            epsilon_decay=float(data["epsilon_decay"]),
            final_epsilon=float(data["final_epsilon"]),
            gamma=float(data["gamma"]),
        )
        agent.q_table = defaultdict(
            lambda: np.zeros(agent.env.action_space.n),
            {int(k): v for k, v in data["q_table"].item().items()}
        )
        agent.metrics = {k: v for k, v in data["metrics"].item().items()}
        return agent
