def play(env, agent, episodes=5):
    """
    Play the an environment using a pre-trained agent.

    Args:
        env: The environment.
        agent: The pre-trained agent.
        episodes: Number of episodes to play.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"Episode {episode + 1}")

        while not done:
            # Render the environment
            env.render()

            # Choose the best action using the agent
            action = agent.act(state)

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Update the state
            state = next_state
            done = terminated or truncated

    env.close()
