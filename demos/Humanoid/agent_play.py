import gymnasium as gym
from stable_baselines3 import SAC
import glob
import os

def play():
    env_name = "Humanoid-v5"
    model_path = "models/sac_humanoid.zip"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train.py first.")
        return

    env = gym.make(env_name, render_mode="human")

    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)

    obs, _ = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

if __name__ == "__main__":
    play()
