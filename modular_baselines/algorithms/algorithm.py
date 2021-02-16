import numpy as np
import torch
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import BaseCollector
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback


class BaseAlgorithm(ABC):
    """ Base abstract class for Algorithms """

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def train(self):
        pass


class OnPolicyAlgorithm(BaseAlgorithm):
    """ Base on policy learning algorithm

    Args:
        policy (torch.nn.Module): Poliicy module with both heads
        buffer (BaseBuffer): Experience Buffer
        collector (BaseCollector): Experience collector
        env (VecEnv): Vectorized environment
        rollout_len (int): n-step length
        callbacks (List[BaseAlgorithmCallback], optional): Algorithm callbacks. Defaults to [].
        device (str, optional): Torch device. Defaults to "cpu".
    """

    def __init__(self,
                 policy: torch.nn.Module,
                 buffer: BaseBuffer,
                 collector: BaseCollector,
                 env: VecEnv,
                 rollout_len: int,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.collector = collector
        self.env = env
        self.device = device
        self.callbacks = callbacks
        self.rollout_len = rollout_len

    def learn(self, total_timesteps: int) -> None:
        """ Main loop for running the on policy algorithm

        Args:
            total_timesteps (int): Total environment timesteps to run
        """

        num_timesteps = 0
        iteration = 0

        for callback in self.callbacks:
            callback.on_training_start(locals())

        while num_timesteps < total_timesteps:

            num_timesteps = self.collector.collect(self.rollout_len)
            self.train()
            for callback in self.callbacks:
                callback.on_step(locals())
            iteration += 1

        for callback in self.callbacks:
            callback.on_training_end(locals())

        return None
