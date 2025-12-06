import gymnasium as gym
from gymnasium.utils.play import play
import pygame

mapping = {
    pygame.K_LEFT: 0,   # Press LEFT to move left
    pygame.K_DOWN: 1,   # Press DOWN to move down
    pygame.K_RIGHT: 2,  # Press RIGHT to move right
    pygame.K_UP: 3,     # Press UP to move up
}

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
play(env, keys_to_action=mapping, zoom=3, wait_on_player=True)
env.close()