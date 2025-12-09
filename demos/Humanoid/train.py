import gymnasium as gym
import torch
from stable_baselines3 import SAC
import os

def train():
    env_name = "Humanoid-v5"
    print(f"Training SAC on {env_name}...")
    
    env = gym.make(env_name, render_mode=None)

    # Hyperparameters inspired by RL Baselines3 Zoo
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=7.3e-4, 
        buffer_size=300000, # Reduced from 1M to save RAM, still sufficient
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        use_sde=True, # State Dependent Exploration usually helps with continuous control
        use_sde_at_warmup=True,
        verbose=1,
        device="auto"
    )

    # Convert to roughly 5 million steps for full training, 
    # but we'll set a smaller number for the demo script or allow user to ctrl+c
    total_timesteps = 1_000_000
    
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sac_humanoid")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    # Save replay buffer if needed (large file)
    # model.save_replay_buffer(os.path.join(model_dir, "sac_humanoid_replay_buffer"))

if __name__ == "__main__":
    train()
