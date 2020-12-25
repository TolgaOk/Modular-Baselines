import torch
import numpy as np
from gym.spaces import Discrete
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from abc import ABC, abstractmethod
from collections import deque

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import BaseBuffer

from modular_baselines.collectors.callbacks import BaseCollectorCallback


class BaseCollector(ABC):
    """ Base abstract class for collectors """

    @abstractmethod
    def collect(self) -> None:
        pass


class BaseOnPolicyCollector(BaseCollector):
    """ Base On policy collector class. Collects a rollout with every collect
    call and add the experience into the buffer.

    Args:
        env (VecEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to load rollout experiences
        policy (torch.nn.Module): Policy module for action sampling
        callbacks (List[BaseCollectorCallback], optional): Collector callbacks.
            Defaults to [].
        device (str, optional): torch device that is used to convert
            experiences before feeding them into the policy module. Defaults
            to "cpu".
    """

    def __init__(self,
                 env: VecEnv,
                 buffer: BaseBuffer,
                 policy: torch.nn.Module,
                 callbacks: List[BaseCollectorCallback] = [],
                 device: str = "cpu"):
        super().__init__()

        self.env = env
        self.device = device
        self.buffer = buffer
        self.policy = policy

        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

        self.num_timesteps = 0

        self.callbacks = callbacks

    def collect(self, n_rollout_steps: int) -> int:
        """ Collect a rollout of experience using the policy and load them to
        buffer

        Args:
            n_rollout_steps (int): Length of the rollout

        Returns:
            int: Total number of timesteps passed.
        """

        n_steps = 0
        self.buffer.reset()

        for callback in self.callbacks:
            callback.on_rollout_start(locals())

        while n_steps < n_rollout_steps:

            actions, *values_to_save = self.process_step()
            new_obs, rewards, dones, infos = self.env.step(actions)

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if isinstance(self.env.action_space, Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self.buffer.add(self._last_obs,
                            actions,
                            rewards,
                            self._last_dones,
                            *values_to_save)

            self._last_obs = new_obs
            self._last_dones = dones

            for callback in self.callbacks:
                callback.on_rollout_step(locals())

        for callback in self.callbacks:
            callback.on_rollout_end(locals())

        return self.num_timesteps

    @abstractmethod
    def process_step(self):
        pass


class OnPolicyCollector(BaseOnPolicyCollector):
    """ On Policy collector that is mainly used with value based PG algorithms
    """

    def process_step(self) -> Tuple[torch.Tensor]:
        """ Sample actions, values and log probability of the actions using 
        the policy

        Returns:
            Tuple[torch.Tensor]: actions, values, and log probabilities tensors
        """

        with torch.no_grad():
            # Convert to pytorch tensor
            obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()
        return actions, values, log_probs
