import torch
import numpy as np
from typing import Dict, Generator, Optional, Union
import gym

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class GeneralBuffer(ReplayBuffer):
    """ Buffer that combines ReplayBuffer and Rollout buffer.

    Method:
        get_rollout: Sample the last experiences from the buffer.
        sample: Uniform sampling from the buffer

    """

    def get_rollout(self, rollout_size: int) -> ReplayBufferSamples:
        """Sample the latest experiences to form a Rollout.

        Args:
            rollout_size (int): Horizon of the rollout

        Returns:
            ReplayBufferSamples: Replay Buffer structure
        """
        assert (self.size() >= rollout_size), ""
        # Prepare the data
        indices = np.arange(-rollout_size, 0) + self.pos - 1
        rollout = {}
        for tensor_name in ["observations", "actions", "rewards",
                            "dones", "next_observations"]:
            rollout[tensor_name] = getattr(self, tensor_name)[indices]

        if isinstance(self.action_space, gym.spaces.Discrete):
            rollout["actions"] = self.onehot(rollout["actions"], axis=-1)

        return ReplayBufferSamples(**rollout)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """ IID sampling method

        Args:
            batch_size (int): Size of the sample

        Returns:
            ReplayBufferSamples: Replay Buffer structure
        """
        env_batch_size = int(np.ceil(batch_size / self.n_envs).item())
        return super().sample(env_batch_size)

    def _get_samples(self, batch_inds: np.ndarray,
                     env=None) -> ReplayBufferSamples:
        if env is not None:
            raise ValueError("_get_sample does not support env normalization")
        if self.optimize_memory_usage:
            raise RuntimeError("Memory optimized usage is not available")

        actions = self.actions[batch_inds].reshape(-1, self.action_dim)
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = self.onehot(np.expand_dims(actions, axis=-1), axis=-1)
            actions = actions.reshape(-1, actions.shape[-1])

        data = (
            self.observations[batch_inds].reshape(-1, *self.obs_shape),
            actions,
            self.next_observations[batch_inds].reshape(-1, *self.obs_shape),
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def onehot(self, tens: torch.Tensor, axis: int):
        """ Make the given axis one hot vector with respect to action space
        size.

        Args:
            tens (torch.Tensor): Tensor whose "axis" dimension is singular and
                contains integer values ranging from 0 to size of the action
                space.
            axis (int): The dimenstion to convert into one-hot representation

        Raise:
            assertion: If the dimension stated by the "axis" in the given
                tensor "tens" has a length different than 1 
        """
        assert tens.shape[axis] == 1, "Dim {} is not one".format(axis)
        max_size = self.action_space.n
        shape = [1] * len(tens.shape)
        shape[axis] = -1
        arange = np.arange(max_size).reshape(*shape)
        return (arange == tens).astype(tens.dtype)
