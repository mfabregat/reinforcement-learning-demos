import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder
import os
from itertools import count

def main():
    env_id = "HalfCheetah-v4"
    models_dir = "models"
    model_path = os.path.join(models_dir, "ppo_halfcheetah")
    vec_norm_path = os.path.join(models_dir, "ppo_halfcheetah_vecnormalize.pkl")

    # Verify paths exist (checking with .zip extension for the model)
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return
    if not os.path.exists(vec_norm_path):
        print(f"Normalization stats not found at {vec_norm_path}")
        return

    # Create the environment.
    print(f"Loading environment {env_id}...")
    env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": "rgb_array"})

    # Load the normalization stats
    # This wraps the environment in the VecNormalize wrapper with the loaded stats
    print(f"Loading normalization stats from {vec_norm_path}...")
    env = VecNormalize.load(vec_norm_path, env)

    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(current_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda x: True)

    
    # Disable training mode for normalization (don't update stats)
    env.training = False
    # Disable reward normalization so we see the raw environment reward
    env.norm_reward = False

    # Load the trained agent
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Starting playback. Press Ctrl+C to stop.")
    
    obs = env.reset()
    total_reward = 0
    
    try:
        for episode in range(200):
            # env.render()
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = env.step(action)
            
            # VecEnv automatically resets when done
            total_reward += rewards[0]
            
            # Since we are using a VecEnv, 'dones' implies the episode ended and was reset automatically.
            if dones[0]:
                print(f"Episode finished. Total reward: {total_reward:.2f}")
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
