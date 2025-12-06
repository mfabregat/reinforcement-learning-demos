import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from rl_agents.dqn import DQNAgent

def train():
    env = gym.make("LunarLander-v3")
    
    # Hyperparameters from the original demo
    agent = DQNAgent(
        env=env,
        batch_size=128,
        learning_rate=3e-4,
        initial_epsilon=0.9,
        epsilon_decay=10000, # Slower decay for LunarLander? Original was complex. 
        final_epsilon=0.01,
        gamma=0.99,
        tau=0.005,
        buffer_size=10000,
        device="auto"
    )

    print(f"Training on {agent.device}")
    
    # Train for a few episodes to verify it runs
    agent.train(n_episodes=20, max_steps=1000)
    
    agent.save("lunar_dqn.pt")
    agent.plot_metrics()

    # Verify loading
    loaded_agent = DQNAgent.load("lunar_dqn.pt", env=env)
    print("Agent loaded successfully.")

if __name__ == "__main__":
    train()
