from collections import defaultdict

from tqdm import tqdm
import gymnasium as gym
import numpy as np
from typing import Union

from .agent import Agent

# Q-Learning Agent - only tested in FrozenLake environment
class QLearningAgent(Agent):
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

        self.metrics = {
            "training_error": [],
            "episode_rewards": [],
            "episode_lengths": [],
        }

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

    def plot_metrics(self) -> None:
        """Plot training metrics with improved smoothing, alignment and headless fallback."""
        import matplotlib.pyplot as plt

        def moving_average(arr, window):
            arr = np.array(arr, dtype=float).flatten()
            if arr.size == 0:
                return np.array([], dtype=int), np.array([], dtype=float)
            window = max(1, min(window, arr.size))
            if window == 1:
                return np.arange(arr.size), arr
            ma = np.convolve(arr, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, arr.size)
            return x, ma

        rolling_length = 500
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        axs = axs.ravel()

        # Episode rewards
        rewards = self.metrics.get("episode_rewards", [])
        x_r, r_ma = moving_average(rewards, rolling_length)
        axs[0].plot(rewards, color="0.85", label="raw")
        if r_ma.size:
            axs[0].plot(x_r, r_ma, color="C0", lw=1.5, label=f"MA(window={min(rolling_length, max(1, len(rewards)))})")
        axs[0].set_title("Episode rewards")
        axs[0].set_ylabel("Reward")
        axs[0].set_xlabel("Episode")
        axs[0].grid(True)
        axs[0].legend()

        # Episode lengths
        lengths = self.metrics.get("episode_lengths", [])
        x_l, l_ma = moving_average(lengths, rolling_length)
        axs[1].plot(lengths, color="0.85", label="raw")
        if l_ma.size:
            axs[1].plot(x_l, l_ma, color="C1", lw=1.5, label="smoothed")
        axs[1].set_title("Episode lengths")
        axs[1].set_ylabel("Steps")
        axs[1].set_xlabel("Episode")
        axs[1].grid(True)
        axs[1].legend()

        # Training error (TD)
        errors = self.metrics.get("training_error", [])
        # TD error is higher-frequency — smooth with a smaller window
        err_window = min(100, max(1, len(errors)))
        x_e, e_ma = moving_average(errors, err_window)
        axs[2].plot(errors, color="0.85", label="raw")
        if e_ma.size:
            axs[2].plot(x_e, e_ma, color="C2", lw=1.5, label=f"MA(window={err_window})")
        axs[2].set_title("Training error (|TD|)")
        axs[2].set_ylabel("Abs TD error")
        axs[2].set_xlabel("Step")
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()