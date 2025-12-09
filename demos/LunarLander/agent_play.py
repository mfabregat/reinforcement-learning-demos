import gymnasium as gym
import numpy as np
from rl_agents import DQNAgent
from rl_agents import play

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    video_dir = os.path.join(current_dir, "videos")
    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)

    # Load the pre-trained agent
    model_dir = "models/dqn_lunarlander.pt"
    agent = DQNAgent.load(model_dir, env)

    agent.plot_metrics(rolling_length=50)

    # Play the environment
    play(env, agent)