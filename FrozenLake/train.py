import gymnasium as gym
from tqdm import tqdm
import numpy as np
import pickle
import os

from agents.q_learning import QLearningAgent

n_episodes = 50_000

env = gym.make("FrozenLake-v1", is_slippery=True)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

learning_rate = 0.1
initial_eps = 1.0
final_eps = 0.00001
# Decay epsilon over 80% of training episodes
eps_decay = (final_eps / initial_eps) ** (1 / (n_episodes * 0.8))


agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_eps=initial_eps,
    final_eps=final_eps,
    eps_decay=eps_decay,
    gamma=0.99,
)


for episode in tqdm(range(n_episodes)):
    # Start a new episode
    obs, info = env.reset()
    done = False

    # Play one complete episode
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.choose_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update_q_value(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_eps()


from matplotlib import pyplot as plt


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()
