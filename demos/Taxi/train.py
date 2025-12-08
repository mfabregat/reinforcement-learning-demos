import gymnasium as gym

from rl_agents import QLearningAgent

n_episodes = 15_000

env = gym.make("Taxi-v3")

learning_rate = 0.1
initial_epsilon = 1.0
final_epsilon = 0.01
# Decay epsilon over 80% of training episodes
epsilon_decay = (final_epsilon / initial_epsilon) ** (1 / (n_episodes * 0.8))

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    final_epsilon=final_epsilon,
    epsilon_decay=epsilon_decay,
    gamma=0.99,
)

agent.train(n_episodes=n_episodes, max_steps=200)

model_dir = "models/q_learning_taxi.npz"
agent.save(model_dir)

agent.plot_metrics()