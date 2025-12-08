import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Union, Optional, Any, Tuple
import gymnasium as gym

from .agent import Agent

class ReinforcePolicy(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, hidden_size: int = 128):
        super(ReinforcePolicy, self).__init__()
        self.action_space = action_space
        n_observations = observation_space.shape[0]

        self.affine1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(p=0.6)

        if isinstance(action_space, gym.spaces.Discrete):
            self.affine2 = nn.Linear(hidden_size, action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            n_actions = action_space.shape[0]
            self.affine2_mu = nn.Linear(hidden_size, n_actions)
            self.affine2_sigma = nn.Linear(hidden_size, n_actions)
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class ReinforceAgent(Agent):
    def __init__(
        self,
        env: Union[gym.Env, str],
        learning_rate: float = 1e-2,
        gamma: float = 0.99,
        hidden_size: int = 128,
        device: str = "auto"
    ):
        super().__init__(env, learning_rate)
        self.gamma = gamma
        
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        self.policy = ReinforcePolicy(self.env.observation_space, self.env.action_space, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def act(self, state: Any) -> Any:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self) -> None:
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        if policy_loss:
            policy_loss = torch.cat(policy_loss).sum()
            
            loss = policy_loss
            
            self.metrics["training_error"].append(loss.item())
            loss.backward()
            self.optimizer.step()
        
        self.saved_log_probs = []
        self.rewards = []

    def train(self, n_episodes: int, max_steps: Optional[int] = None) -> None:
        for i_episode in range(1, n_episodes + 1):
            state, _ = self.env.reset()
            ep_reward = 0
            
            self.saved_log_probs = []
            self.rewards = []
            
            done = False
            t = 0
            while not done:
                action = self.act(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                self.rewards.append(reward)
                ep_reward += reward
                t += 1
                
                if terminated or truncated:
                    done = True
                
                if max_steps and t >= max_steps:
                    done = True

            self.update()
            
            self.metrics["episode_rewards"].append(ep_reward)
            self.metrics["episode_lengths"].append(t)
            
            if i_episode % 10 == 0:
                 avg_last_10 = np.mean(self.metrics["episode_rewards"][-10:])
                 print(f"Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward (last 10): {avg_last_10:.2f}")

    def save(self, filepath: str) -> None:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, filepath)

    @classmethod
    def load(cls, filepath: str, env: Union[gym.Env, str], **kwargs) -> 'ReinforceAgent':
        checkpoint = torch.load(filepath, map_location=kwargs.get('device', 'cpu'), weights_only=False)
        agent = cls(env, **kwargs)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.metrics = checkpoint.get('metrics', agent.metrics)
        return agent
