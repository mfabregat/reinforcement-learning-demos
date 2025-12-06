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
        initial_epsilon=1.0,  # Start with full exploration
        epsilon_decay=15000,  # Faster decay for quicker exploitation
        final_epsilon=0.05,  # Higher final epsilon for more exploration
        gamma=0.98,  # Slightly reduced for better short-term rewards
        tau=0.01,  # Increased for smoother target updates
        buffer_size=50000,  # Increased to store more experiences
    )

    print(f"Training on {agent.device}")
    
    # Train for a few episodes to verify it runs
    agent.train(n_episodes=500, max_steps=1000)
    
    agent.save("lunar_dqn.pt")
    agent.plot_metrics(rolling_length=50)

    # # Verify loading
    # loaded_agent = DQNAgent.load("lunar_dqn.pt", env=env)
    # print("Agent loaded successfully.")

if __name__ == "__main__":
    train()
