import gymnasium as gym
from gymnasium.utils.play import play
import pygame

mapping = {
    pygame.K_LEFT: 3,   # Press LEFT to fire right engine (turn left)
    pygame.K_RIGHT: 1,  # Press RIGHT to fire left engine (turn right)
    pygame.K_UP: 2,     # Press UP to fire main engine (thrust down)
}

env = gym.make("LunarLander-v3", render_mode="rgb_array")
play(env, keys_to_action=mapping, zoom=3)
env.close()