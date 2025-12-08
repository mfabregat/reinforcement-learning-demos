import gymnasium as gym
from rl_agents import QLearningAgent
from rl_agents import play

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the Taxi environment
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    video_dir = os.path.join(current_dir, "videos")
    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)

    # Load the pre-trained agent
    model_dir = "models/q_learning_taxi.npz"
    agent = QLearningAgent.load(model_dir, env)

    # Play the Taxi environment
    play(env, agent)