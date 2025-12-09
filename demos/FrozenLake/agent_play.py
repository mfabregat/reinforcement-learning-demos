import gymnasium as gym
import numpy as np
from rl_agents import QLearningAgent
from rl_agents.agent_play import play

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))

    env = gym.make("FrozenLake-v1", render_mode="rgb_array")

    video_dir = os.path.join(current_dir, "videos")
    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)

    # Load the pre-trained agent
    agent_dir = "models/q_learning_frozenlake.npz"
    agent = QLearningAgent.load(agent_dir, env)
    agent.plot_metrics()

    # Play the environment
    play(env, agent)