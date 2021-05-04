import torch
import numpy as np
from typing import Dict, Generator, Optional, Union
import gym
import warnings

from modular_baselines.buffers.buffer import GeneralBuffer
from modular_baselines.contrib.jacobian_trace.type_aliases import SequentialRolloutSamples
from stable_baselines3.common.vec_env import VecNormalize


class JTACBuffer(GeneralBuffer):
    
    def get_sequential_rollout(self,
                               rollout_size: int,
                               sample_last: int = True,
                               reverse: bool = False,
                               ) -> Generator[SequentialRolloutSamples, None, None]:
        assert (self.size() > rollout_size), ""
        assert (self.buffer_size > rollout_size), "Buffer size must be larger than the rollout_size"

        if sample_last:
            # We subsctract 1 from the upper index so that we can calculate advantage
            pos_indices = np.zeros(self.n_envs, dtype="int64") + self.pos - rollout_size - 1
        else:
            lower_index = self.pos - self.buffer_size if self.full else 0
            pos_indices = np.random.randint(lower_index, self.pos - rollout_size - 1, size=self.n_envs)
        
        rollout_indexes = list(reversed(range(rollout_size)) if reverse else range(rollout_size))
        for index in rollout_indexes:
            yield self._get_rollout_samples(pos_indices=pos_indices + index,
                                            env_indices=np.arange(self.n_envs))

    def compute_returns_and_advantage(self,
                                      rollout_size: int,
                                      initial_pos_indices: np.ndarray,
                                      gamma: float,
                                      gae_lambda: Optional[float] = 1.0
                                      ) -> None:
        assert self.size() > rollout_size, (
            ("Buffer size {} must be at least 1 larger"
             " than the rollout size {}").format(self.size(), rollout_size))
        pos_indices = initial_pos_indices - 1

        advantage = 0
        env_indices = np.arange(self.n_envs)
        for index in reversed(range(rollout_size)):
            pos_indices = initial_pos_indices + index

            termination = (1 - self.dones[pos_indices, env_indices])
            td_error = (self.values[pos_indices + 1, env_indices] * gamma * termination
                        + self.rewards[pos_indices, env_indices] 
                        - self.values[pos_indices, env_indices])
            
            advantage = td_error + advantage * gamma * gae_lambda * termination
            self.advantages[pos_indices, env_indices] = advantage
            self.returns[pos_indices, env_indices] = advantage + self.values[pos_indices, env_indices]

    def _get_rollout_samples(self,
                             pos_indices: np.ndarray,
                             env_indices: np.ndarray,
                             env: Optional[VecNormalize] = None
                             ) -> SequentialRolloutSamples:
        data = (
            self.observations[pos_indices, env_indices],
            self.next_observations[pos_indices, env_indices],
            self.actions[pos_indices, env_indices],
            self.advantages[pos_indices, env_indices].flatten(),
            self.returns[pos_indices, env_indices].flatten(),
            self.dones[pos_indices, env_indices].flatten(),
            self.rewards[pos_indices, env_indices].flatten(),
        )
        return SequentialRolloutSamples(*list(map(self.to_torch, data)))