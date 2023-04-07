from typing import List, Optional, Union, Dict, Any, Tuple, Protocol
import numpy as np
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from modular_baselines.buffers.buffer import BaseBuffer
from modular_baselines.loggers.datalog import ListDataLog, SingularDataLog, HistogramDataLog


class BaseCollector():
    """ Base abstract class for collectors """

    @abstractmethod
    def collect(self, n_rollout_steps: int) -> int:
        pass


class Agent(Protocol):
    def sample_action(self, observation: np.ndarray) -> np.ndarray: ...


class RolloutLogger(Protocol):
    env_reward: ListDataLog
    env_length: ListDataLog
    actions: HistogramDataLog


class RolloutCollector(BaseCollector):
    """ Collects a rollout with every collect call and add the experience into the buffer.

    Args:
        env (VecEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to push rollout experiences
        agent (Agent): Action sampling agent
        logger (RolloutLogger): Data logger to log environment reward and lengths at termination
    """

    _required_buffer_fields = ("observation", "next_observation", "reward",
                               "termination", "action")

    def __init__(self,
                 env: VecEnv,
                 buffer: BaseBuffer,
                 agent: Agent,
                 logger: RolloutLogger,
                 store_normalizer_stats: bool = False):
        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.store_normalizer_stats = store_normalizer_stats
        self.logger = logger

        for field in self._required_buffer_fields:
            assert field in buffer.struct.names, (
                "Buffer does not contain the field name {}".format(field))

        self.num_timesteps = 0
        self._last_obs = self.env.reset()

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

            new_obs, rewards, dones, infos = self.environment_step(actions)
            next_obs = new_obs
            # Terminated new_obs is different than the given next_obs
            terminated_indexes = np.argwhere(dones.flatten() == 1).flatten()
            if len(terminated_indexes) > 0:
                next_obs = new_obs.copy()
                for index in terminated_indexes:
                    next_obs[index] = infos[index]["terminal_observation"]

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if self.store_normalizer_stats:
                normalizer_stats["reward_rms_var"] = self.env.ret_rms.var.reshape(
                    1, -1).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_mean"] = np.expand_dims(
                    self.env.obs_rms.mean, axis=0).repeat(self.env.num_envs, axis=0)
                normalizer_stats["next_obs_rms_var"] = np.expand_dims(
                    self.env.obs_rms.var, axis=0).repeat(self.env.num_envs, axis=0)

            self.buffer.push({
                "observation": self._last_obs,
                "next_observation": next_obs,
                "reward": rewards,
                "termination": dones,
                "action": actions,
                **policy_content,
                **normalizer_stats,
            })

            # Log environment info
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    self.logger.env_reward.push(maybe_ep_info["r"])
                    self.logger.env_length.push(maybe_ep_info["l"])
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
        observation, rewards, dones, infos = self.env.step(actions)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        return observation, rewards, dones, infos
