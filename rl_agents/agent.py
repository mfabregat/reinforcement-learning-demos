from tqdm import tqdm
import gymnasium as gym
import abc
from typing import Any, Dict, Optional, Union
import numpy as np

class Agent(abc.ABC):
    def __init__(self, env: Union[gym.Env, str], learning_rate: float):
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        self.learning_rate = learning_rate
        self.metrics: Dict[str, list] = {
            "training_error": [],
            "episode_rewards": [],
            "episode_lengths": [],
        }

    @abc.abstractmethod
    def act(self, state: Any) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def plot_metrics(self, metrics: Optional[list] = None, rolling_length: int = 500) -> None:
        import matplotlib.pyplot as plt
        from .utils import moving_average, almost_factors

        if metrics is None:
            metrics = self.metrics.keys()
        
        n_metrics = len(metrics)
        ncols, nrows = almost_factors(n_metrics)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows))
        axs = np.ravel(axs)

        for i, metric in enumerate(metrics):
            data = self.metrics.get(metric, [])
            x, ma = moving_average(data, rolling_length)
            axs[i].plot(data, color="0.85", label="raw")
            if ma.size:
                axs[i].plot(x, ma, color=f"C{i}", lw=1.5, label=f"MA(window={min(rolling_length, max(1, len(data)))})")
            axs[i].set_title(metric.replace("_", " ").title())
            axs[i].set_ylabel(metric.replace("_", " ").title())
            axs[i].set_xlabel("Episode" if "episode" in metric else "Step")
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()
        plt.show()

    @abc.abstractmethod
    def save(self, filepath: str) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: str, env: Union[gym.Env, str]) -> 'Agent':
        raise NotImplementedError