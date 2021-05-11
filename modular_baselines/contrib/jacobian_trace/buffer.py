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
                               batch_size: int,
                               maximum_horizon: Optional[int] = None,
                               ) -> Generator[SequentialRolloutSamples, None, None]:
        assert (self.size() >= rollout_size), ""
        assert (self.buffer_size >= rollout_size), "Buffer size must be larger than the rollout_size"

        if maximum_horizon is None:
            maximum_horizon = self.size()
        maximum_horizon = min(maximum_horizon, self.size())

        assert maximum_horizon >= rollout_size, "Maximum horizon must be larger than rollout size"

        lower_index = self.pos - maximum_horizon
        pos_indices = np.random.randint(low=lower_index, high=self.pos-rollout_size+1, size=batch_size)
        pos_indices = pos_indices.reshape(-1, 1) + np.arange(rollout_size).reshape(1, -1)

        env_indices = np.random.randint(0, self.n_envs, size=(
            batch_size, 1)).repeat(rollout_size, axis=1)

        return self._get_rollout_samples(pos_indices=pos_indices,
                                         env_indices=env_indices)

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
            self.returns[pos_indices, env_indices] = advantage + \
                self.values[pos_indices, env_indices]

    def _get_rollout_samples(self,
                             pos_indices: np.ndarray,
                             env_indices: np.ndarray,
                             env: Optional[VecNormalize] = None
                             ) -> SequentialRolloutSamples:
        data = (
            self.observations[pos_indices, env_indices],
            self.next_observations[pos_indices, env_indices],
            self.actions[pos_indices, env_indices],
            self.advantages[pos_indices, env_indices],
            self.returns[pos_indices, env_indices],
            self.dones[pos_indices, env_indices],
            self.rewards[pos_indices, env_indices],
        )
        return SequentialRolloutSamples(*list(map(self.to_torch, data)))
