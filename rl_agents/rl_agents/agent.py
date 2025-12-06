from __future__ import annotations

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
    
    def plot_metrics(self, metrics: Optional[list] = None) -> None:
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = self.metrics.keys()
        
        rolling_length = 500
        n_metrics = len(metrics)
        ncols, nrows = self.almost_factors(n_metrics)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows))
        axs = np.ravel(axs)

        for i, metric in enumerate(metrics):
            data = self.metrics.get(metric, [])
            x, ma = self.moving_average(data, rolling_length)
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

    @staticmethod
    def moving_average(arr, window):
        arr = np.array(arr, dtype=float).flatten()
        if arr.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        window = max(1, min(window, arr.size))
        if window == 1:
            return np.arange(arr.size), arr
        ma = np.convolve(arr, np.ones(window) / window, mode="valid")
        x = np.arange(window - 1, arr.size)
        return x, ma
    
    @staticmethod
    def almost_factors(number):
        '''
        https://stackoverflow.com/a/77243426
        find a pair of factors that are close enough for a number that is close enough
        '''
        def close_factors(number):
            ''' 
            find the closest pair of factors for a given number
            '''
            factor1 = 0
            factor2 = number
            while factor1 +1 <= factor2:
                factor1 += 1
                if number % factor1 == 0:
                    factor2 = number // factor1
                
            return factor1, factor2
        while True:
            factor1, factor2 = close_factors(number)
            if 1/2 * factor1 <= factor2: # the fraction in this line can be adjusted to change the threshold aspect ratio
                break
            number += 1
        return factor1, factor2
    