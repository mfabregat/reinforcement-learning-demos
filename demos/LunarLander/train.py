import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from rl_agents.dqn import DQNAgent

def train():
    env = gym.make("LunarLander-v3")
    
    # Tuned hyperparameters based on DQN best practices for LunarLander-v3
    # Research indicates these values provide optimal performance:
    # - Learning rate: 1e-4 for stable learning
    # - Buffer size: 100,000 for diverse experience replay
    # - Epsilon decay: 5000 steps balances exploration/exploitation
    # - Network will use 128-128 architecture (defined in dqn.py)
    agent = DQNAgent(
        env=env,
        batch_size=128,           # Optimal for LunarLander
        learning_rate=1e-4,       # Reduced from 3e-4 for more stable learning
        initial_epsilon=1.0,      # Start with full exploration
        epsilon_decay=5000,       # Decay over 5000 steps for balanced exploration
        final_epsilon=0.01,       # Maintain 1% exploration
        gamma=0.99,               # Standard discount factor
        tau=0.005,                # Soft update rate for target network
        buffer_size=100000,       # Increased from 10k for better experience diversity
        device="auto"
    )

    print(f"Training on {agent.device}")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {agent.learning_rate}")
    print(f"  Batch size: {agent.batch_size}")
    print(f"  Buffer size: {len(agent.memory.memory.maxlen) if hasattr(agent.memory.memory, 'maxlen') else 'N/A'}")
    print(f"  Epsilon: {agent.initial_epsilon} -> {agent.final_epsilon} (decay: {agent.epsilon_decay})")
    print(f"  Gamma: {agent.gamma}, Tau: {agent.tau}")
    
    # Train for sufficient episodes to see convergence
    # LunarLander typically needs 500-1000 episodes to solve
    agent.train(n_episodes=1000, max_steps=1000)
    
    agent.save("lunar_dqn.pt")
    agent.plot_metrics()

    # Verify loading
    loaded_agent = DQNAgent.load("lunar_dqn.pt", env=env)
    print("Agent loaded successfully.")
    
    # Print final performance
    final_rewards = agent.metrics["episode_rewards"][-100:]
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Mean reward: {np.mean(final_rewards):.2f}")
    print(f"  Std reward: {np.std(final_rewards):.2f}")
    print(f"  Max reward: {np.max(final_rewards):.2f}")
    print(f"  Min reward: {np.min(final_rewards):.2f}")

if __name__ == "__main__":
    train()
