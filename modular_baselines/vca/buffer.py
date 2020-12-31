import torch
import numpy as np
import gym
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class Buffer(ReplayBuffer):

    def __init__(self,
                 buffer_size: int,
                 observation_space: Union[gym.spaces.Box, gym.spaces.Discrete],
                 action_space: gym.spaces.Discrete,
                 device: str = "cpu",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False):
        super().__init__(buffer_size,
                         observation_space,
                         action_space,
                         device=device,
                         n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage)

    def sample_last_episode(self) -> ReplayBufferSamples:
        end_index = self.pos - 1

        while self.dones[end_index, 0] == False:
            end_index -= 1
            if end_index < 0 and self.full is False:
                return
            if end_index < (-self.size() + 1):
                return

        start_index = end_index - 1
        while self.dones[start_index, 0] == False:
            start_index -= 1
            if start_index < 0 and self.full is False:
                return
            if start_index < -self.size():
                return

        if (start_index % self.buffer_size) == (end_index % self.buffer_size):
            return

        batch_inds = np.arange(start_index+1, end_index+1)
        return self._get_samples(batch_inds, env=None)
