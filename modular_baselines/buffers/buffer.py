import torch
import numpy as np
from typing import Dict, Generator, Optional, Union
import gym
import warnings

from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class GeneralBuffer(ReplayBuffer):
    """ Buffer that combines ReplayBuffer and Rollout buffer.

    Method:
        get_rollout: Sample the last experiences from the buffer.
        sample: Uniform sampling from the buffer

    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 device: Union[torch.device, str] = "cpu",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 ):
        """ This function is only overwritten to remove assertion for n_envs > 1.
        """
        BaseBuffer.__init__(self,
                            buffer_size,
                            observation_space,
                            action_space,
                            device,
                            n_envs=n_envs)
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype)
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Rollout based memory
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + \
                self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB")

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: torch.Tensor,
            log_prob: torch.Tensor):
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        super().add(obs, next_obs, action, reward, done)

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, self.n_envs, len(batch_inds))
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.dones[batch_inds, env_indices],
            self._normalize_reward(self.rewards[batch_inds, env_indices], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def get_rollout(self,
                    rollout_size: int,
                    batch_size: Optional[int] = None
                    ) -> Generator[RolloutBufferSamples, None, None]:
        """Sample the latest experiences to form a Rollout.

        Args:
            rollout_size (int): Horizon of the rollout

        Returns:
            ReplayBufferSamples: Replay Buffer structure
        """
        assert (self.size() >= rollout_size), ""
        assert (self.buffer_size > rollout_size), "Buffer size must be larger than the rollout_size"
        pos_indices = np.arange(-rollout_size, 0) + self.pos - 1
        rollout = {}
        for tensor_name in ["observations", "actions", "values", "log_probs",
                            "advantages", "returns"]:
            rollout[tensor_name] = getattr(self, tensor_name)[pos_indices]

        if batch_size is None:
            batch_size = rollout_size * self.n_envs

        start_idx = 0
        batch_indices = np.random.permutation(rollout_size * self.n_envs)
        while start_idx < rollout_size * self.n_envs:
            indices = batch_indices[start_idx: start_idx + batch_size]
            yield self._get_rollout_samples(
                pos_indices[indices // self.n_envs], indices % self.n_envs)
            start_idx += batch_size

    def _get_rollout_samples(self,
                             pos_indices: np.ndarray,
                             env_indices: np.ndarray,
                             env: Optional[VecNormalize] = None
                             ) -> RolloutBufferSamples:
        data = (
            self.observations[pos_indices, env_indices],
            self.actions[pos_indices, env_indices],
            self.values[pos_indices, env_indices].flatten(),
            self.log_probs[pos_indices, env_indices].flatten(),
            self.advantages[pos_indices, env_indices].flatten(),
            self.returns[pos_indices, env_indices].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def compute_returns_and_advantage(self,
                                      rollout_size: int,
                                      gamma: float,
                                      gae_lambda: Optional[float] = 1.0
                                      ) -> None:
        assert self.size() > rollout_size, (
            ("Buffer size {} must be at least 1 larger"
             " than the rollout size {}").format(self.size(), rollout_size))
        steps = np.arange(-rollout_size, 0) + self.pos - 1

        td_array = (self.values[steps + 1] * gamma * (1 - self.dones[steps])
                    + self.rewards[steps] - self.values[steps])
        advantage = 0
        for index in reversed(range(rollout_size)):
            advantage = td_array[index] + advantage * gamma * \
                gae_lambda * (1 - self.dones[steps[index]])
            self.advantages[steps[index]] = advantage
            self.returns[steps[index]] = advantage + self.values[steps[index]]
