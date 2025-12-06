from typing import Union, Optional
import gymnasium as gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .agent import Agent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class DQNAgent(Agent):
    def __init__(
        self,
        env: Union[gym.Env, str],
        batch_size: int = 128,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.9,
        epsilon_decay: float = 5000,  # Now always interpreted as decay steps for exponential schedule
        final_epsilon: float = 0.01,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 10000,
        device: str = "auto"
    ):
        super().__init__(env, learning_rate)
        self.batch_size = batch_size
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay  # Now always number of steps for exp decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.tau = tau

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        n_actions = self.env.action_space.n
        n_observations = self.env.observation_space.shape[0]

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(buffer_size)

        self.steps_done = 0

    def act(self, state: np.ndarray) -> int:
        sample = random.random()
        # Always use exponential decay schedule per step
        eps_threshold = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
            math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state).max(1).indices.view(1, 1).item()
        else:
            return self.env.action_space.sample()

    def update(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.metrics["training_error"].append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, n_episodes: int, max_steps: Optional[int] = None) -> None:
        if max_steps is None:
            max_steps = 1000 # Default for LunarLander

        for i_episode in range(n_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0.0
            
            for t in count():
                action = self.act(state)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                # We need to make sure action is a tensor for the batch processing
                action_tensor = torch.tensor([[action]], device=self.device)
                self.memory.push(state, action_tensor, next_state, reward)

                state = next_state

                self.update()
                


                if done or (max_steps and t >= max_steps):
                    self.metrics["episode_rewards"].append(episode_reward)
                    self.metrics["episode_lengths"].append(t + 1)
                    break
            
            # Print progress every 10 episodes
            if (i_episode + 1) % 10 == 0:
                avg_reward = np.mean(self.metrics["episode_rewards"][-10:])
                # Compute current epsilon for reporting
                current_epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                    math.exp(-1.0 * self.steps_done / self.epsilon_decay)
                print(f"Episode {i_episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {current_epsilon:.4f}")

    def save(self, filepath: str) -> None:
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'steps_done': self.steps_done,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, env: Union[gym.Env, str], **kwargs) -> 'DQNAgent':
        # We need to instantiate the agent first. 
        # Ideally we would load the config from the file too, but for now we assume the user passes the correct env.
        # We can load the file first to check if there are any params we can infer, but usually we just load weights.
        
        checkpoint = torch.load(filepath, weights_only=False)
        agent = cls(env=env, **kwargs) # User must provide other init params or we rely on defaults
        
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.metrics = checkpoint['metrics']
        agent.steps_done = checkpoint.get('steps_done', 0)
        
        return agent