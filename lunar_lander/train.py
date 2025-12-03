import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=21)

print("Initial Observation:", observation)


episode_over = False
total_reward = 0.0

while not episode_over:
    action = env.action_space.sample()  # Sample a random action
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()