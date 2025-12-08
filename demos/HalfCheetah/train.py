import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os

def main():
    env_id = "HalfCheetah-v4"
    # Dinamically set n_envs based CPU cores minus one for system stability
    n_envs = 10
    # n_timesteps from config: 1e6
    n_timesteps = int(1e6)

    # Create the environment
    # Using make_vec_env ensures it is wrapped in a VecEnv (DummyVecEnv by default for n_envs=1)
    env = make_vec_env(env_id, n_envs=n_envs)
    
    # Normalize observations and rewards as per hyperparams
    # "normalize: true" implies both obs and reward normalization usually in rl-zoo context
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.98)

    # Hyperparameters
    # policy_kwargs: "dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))"
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=64,
        n_steps=512,
        gamma=0.98,
        learning_rate=2.0633e-05,
        ent_coef=0.000401762,
        clip_range=0.1,
        n_epochs=20,
        gae_lambda=0.92,
        max_grad_norm=0.8,
        vf_coef=0.58096,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )

    print(f"Training PPO on {env_id} for {n_timesteps} timesteps...")
    
    try:
        model.learn(total_timesteps=n_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted explicitly.")
    
    # Save the model and the normalization stats
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "ppo_halfcheetah")
    vec_norm_path = os.path.join(models_dir, "ppo_halfcheetah_vecnormalize.pkl")
    
    model.save(model_path)
    env.save(vec_norm_path)
    
    print(f"Model saved to {model_path}")
    print(f"VecNormalize stats saved to {vec_norm_path}")

if __name__ == "__main__":
    main()
