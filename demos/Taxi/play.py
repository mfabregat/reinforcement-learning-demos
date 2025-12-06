import gymnasium as gym
from gymnasium.utils.play import play
import pygame

mapping = {
    pygame.K_DOWN: 0,   # Press DOWN to move down
    pygame.K_UP: 1,     # Press UP to move up
    pygame.K_RIGHT: 2,  # Press RIGHT to move right
    pygame.K_LEFT: 3,   # Press LEFT to move left
    pygame.K_a: 4,      # Press A to pick up passenger
    pygame.K_d: 5,      # Press D to drop off passenger
}

env = gym.make("Taxi-v3", render_mode="rgb_array")
play(env, keys_to_action=mapping, zoom=3, wait_on_player=True)
env.close()