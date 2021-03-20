import torch
import numpy as np
import gym
from typing import Dict, Generator, Optional, Union, NamedTuple
from collections import deque

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import (RolloutBuffer,
                                              RolloutBufferSamples,
                                              BaseBuffer)


class RolloutBufferItem(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray


class NstepRolloutBuffer(RolloutBuffer):

    def __init__(
            self,
            buffer_size: int,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            device: Union[torch.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            batch_size: int = None):

        self.batch_size = batch_size
        super().__init__(
            buffer_size * 2,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,)
        self.second_half = slice(self.buffer_size // 2, self.buffer_size, 1)
        self.first_half = slice(0, self.buffer_size // 2, 1)

    def reset(self):
        if self.pos == 0:
            # Only the first time the pos attribute will be 0
            return RolloutBuffer.reset(self)
        self._shift_rollout()
        self.pos = self.buffer_size // 2
        self.generator_ready = False
        self.full = False

    def _shift_rollout(self):
        for name in RolloutBufferItem._fields:
            getattr(self, name)[self.first_half] = getattr(
                self, name)[self.second_half]

    def compute_returns_and_advantage(self, **kwargs) -> None:
        """ Computes n-step Advantage.
        """
        nstep = self.buffer_size // 2
        td_target = self.rewards[:-1] + self.gamma * \
            (1 - self.dones[:-1]) * self.values[1:]
        td_error = td_target - self.values[:-1]

        indexes = np.arange(nstep).reshape(
            1, -1) + np.arange(nstep).reshape(-1, 1)

        # Indexed TD error 3Dtensor (nstep, nstep, nenv)
        td_tensor = td_error[indexes]
        # GAE filter (nstep, nstep)
        _filter = np.tile(self.gae_lambda ** np.arange(nstep).reshape(1, -1, 1) , (nstep, 1, self.n_envs))
        # Termination mask (2 * nstep - 1, nenv)
        mask = (np.cumsum(self.dones[indexes], axis=1) == 0).astype(np.float32)
        mask = np.concatenate([np.ones((nstep, 1, self.n_envs)), mask[:, :-1, :]], axis=1)

        _filter *= mask
        advantage = (td_tensor * _filter).sum(1)

        self.advantages[self.first_half] = advantage
        self.returns[self.first_half] = advantage + \
            self.values[self.first_half]

    def get(self, **kwargs) -> Generator[RolloutBufferSamples, None, None]:
        """ Taken from SB3 and slightly modified
        """
        assert self.full, ""
        indices = np.random.permutation((self.buffer_size // 2) * self.n_envs)
        # Prepare the data
        batch_arrays = {}
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                batch_arrays[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor][self.first_half])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if self.batch_size is None:
            self.batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + self.batch_size],
                                    batch_arrays)
            start_idx += self.batch_size

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     batch_arrays: Dict[str, np.ndarray],
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        """ Taken from SB3 and slightly modified
        """
        data = (
            batch_arrays["observations"][batch_inds],
            batch_arrays["actions"][batch_inds],
            batch_arrays["values"][batch_inds].flatten(),
            batch_arrays["log_probs"][batch_inds].flatten(),
            batch_arrays["advantages"][batch_inds].flatten(),
            batch_arrays["returns"][batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
