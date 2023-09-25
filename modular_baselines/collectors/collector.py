from typing import List, Optional, Union, Dict, Any, Tuple, Protocol
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Discrete, Box
from gymnasium.vector import VectorEnv

from flax.core.frozen_dict import FrozenDict
from typing import Any, Callable

from modular_baselines.buffers.buffer import BaseBuffer
from modular_baselines.loggers.datalog import ListDataLog, HistogramDataLog
from modular_baselines.networks.network_jax import TrainState

class BaseCollector(ABC):
    """ Base abstract class for collectors """

    @abstractmethod
    def collect(self, n_rollout_steps: int) -> int:
        pass


class Agent(Protocol):
    def sample_action(self, observation: np.ndarray
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]: ...


class RolloutLogger(Protocol):
    env_reward: ListDataLog
    env_length: ListDataLog
    actions: HistogramDataLog


class RolloutCollector(BaseCollector):
    """ Collects a rollout with every collect call and add the experience into the buffer.

    Args:
        env (VectorEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to push rollout experiences
        agent (Agent): Action sampling agent
        logger (RolloutLogger): Data logger to log environment reward and lengths at termination
    """

    _required_buffer_fields = ("observation", "next_observation", "reward",
                               "termination", "truncation", "action")

    def __init__(self,
                 env: VectorEnv,
                 buffer: BaseBuffer,
                 agent: Agent,
                 logger: RolloutLogger,
                 rng_seed: Optional[int] = None,
                 store_normalizer_stats: bool = False):
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.store_normalizer_stats = store_normalizer_stats
        self.logger = logger
        self.rng = np.random.default_rng(rng_seed)

        for field in self._required_buffer_fields:
            assert field in buffer.struct.names, (
                "Buffer does not contain the field name {}".format(field))

        self.num_timesteps = 0
        self._last_obs, _ = self.env.reset(seed=[item.item() for item in self.rng.integers(0, 2**30, env.num_envs)])

    def collect(self, n_rollout_steps: int) -> int:
        """ Collect a rollout of experience using the agent and load them to
        buffer

        Args:
            n_rollout_steps (int): Length of the rollout

        Returns:
            int: Total number of time steps passed.
        """

        n_steps = 0
        while n_steps < n_rollout_steps:

            actions, policy_content = self.get_actions()
            normalizer_stats = {}
            if self.store_normalizer_stats:
                normalizer_stats["obs_rms_mean"] = np.expand_dims(
                    self.env.obs_rms.mean, axis=0).repeat(self.env.num_envs, axis=0)
                normalizer_stats["obs_rms_var"] = np.expand_dims(
                    self.env.obs_rms.var, axis=0).repeat(self.env.num_envs, axis=0)

            new_obs, rewards, termination, truncation, infos = self.environment_step(actions)
            dones = np.logical_or(termination, truncation)
            next_obs = new_obs.copy() if dones.any() else new_obs
            for index in np.argwhere(dones.flatten()).flatten():
                next_obs[index] = infos["final_observation"][index]

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if self.store_normalizer_stats:
                normalizer_stats["return_rms_var"] = self.env.return_rms.var.reshape(
                    1, -1).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_mean"] = np.expand_dims(
                    self.env.obs_rms.mean, axis=0).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_var"] = np.expand_dims(
                    self.env.obs_rms.var, axis=0).repeat(self.env.num_envs, axis=0)

            self.buffer.push({
                "observation": self._last_obs,
                "next_observation": next_obs,
                "reward": rewards,
                "termination": termination,
                "truncation": truncation,
                "action": actions,
                **policy_content,
                **normalizer_stats,
            })

            # Log environment info
            if dones.any():
                for index in np.argwhere(dones.flatten()).flatten():
                    self.logger.env_reward.push(infos["episode"]["r"][index].item())                 
                    self.logger.env_length.push(infos["episode"]["l"][index].item())                 
            # Logging actions for histogram
            self.logger.actions.push(actions)
            self._last_obs = new_obs

        return self.num_timesteps

    def get_actions(self):
        return self.agent.sample_action(self._last_obs)

    def environment_step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if isinstance(self.env.action_space, Discrete):
            actions = actions.reshape(-1)
        if isinstance(self.env.action_space, Box):
            actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        observation, rewards, termination, truncation, infos = self.env.step(actions)
        return (observation,
                *(array.reshape(-1, 1) for array in (rewards, termination, truncation)),
                infos)

class RolloutCollectorJax(RolloutCollector):
    def __init__(self,
                 env: VectorEnv,
                 buffer: BaseBuffer,
                 agent: Agent,
                 logger: RolloutLogger,
                 rng_seed: Optional[int] = None,
                 store_normalizer_stats: bool = False):
        super().__init__(env, buffer, agent, logger, rng_seed, store_normalizer_stats)

    def collect(self, n_rollout_steps: int, state: TrainState) -> int:
        """ Collect a rollout of experience using the agent and load them to
        buffer

        Args:
            n_rollout_steps (int): Length of the rollout

        Returns:
            int: Total number of time steps passed.
        """

        n_steps = 0
        while n_steps < n_rollout_steps:

            actions, policy_content = self.get_actions(state.apply_fn, state.params)
            normalizer_stats = {}
            if self.store_normalizer_stats:
                normalizer_stats["obs_rms_mean"] = np.expand_dims(
                    self.env.obs_rms.mean, axis=0).repeat(self.env.num_envs, axis=0)
                normalizer_stats["obs_rms_var"] = np.expand_dims(
                    self.env.obs_rms.var, axis=0).repeat(self.env.num_envs, axis=0)

            new_obs, rewards, termination, truncation, infos = self.environment_step(actions)
            dones = np.logical_or(termination, truncation)
            next_obs = new_obs.copy() if dones.any() else new_obs
            for index in np.argwhere(dones.flatten()).flatten():
                next_obs[index] = infos["final_observation"][index]

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if self.store_normalizer_stats:
                normalizer_stats["return_rms_var"] = self.env.return_rms.var.reshape(
                    1, -1).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_mean"] = np.expand_dims(
                    self.env.obs_rms.mean, axis=0).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_var"] = np.expand_dims(
                    self.env.obs_rms.var, axis=0).repeat(self.env.num_envs, axis=0)

            self.buffer.push({
                "observation": self._last_obs,
                "next_observation": next_obs,
                "reward": rewards,
                "termination": termination,
                "truncation": truncation,
                "action": actions,
                **policy_content,
                **normalizer_stats,
            })

            # Log environment info
            if dones.any():
                for index in np.argwhere(dones.flatten()).flatten():
                    self.logger.env_reward.push(infos["episode"]["r"][index].item())                 
                    self.logger.env_length.push(infos["episode"]["l"][index].item())                 
            # Logging actions for histogram
            self.logger.actions.push(actions)
            self._last_obs = new_obs

        return self.num_timesteps
    
    def get_actions(self, 
                    apply_fn: Callable[..., Any],
                    params: FrozenDict
                    ):
        return self.agent.sample_action(self._last_obs,apply_fn, params)
