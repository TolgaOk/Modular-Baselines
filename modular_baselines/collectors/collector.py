
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env import VecEnv

from modular_baselines.component import Component
from modular_baselines.buffers.buffer import BaseBuffer
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, ListDataLog, HistListDataLog


class BaseCollectorCallback(ABC):
    """ Base class for buffer callbacks that only supports:
    on_rollout_start, on_rollout_step, and on_rollout_end calls.
    """

    @abstractmethod
    def on_rollout_start(self, *args) -> None:
        pass

    @abstractmethod
    def on_rollout_step(self, *args) -> None:
        pass

    @abstractmethod
    def on_rollout_end(self, *args) -> None:
        pass


class BaseCollector(Component):
    """ Base abstract class for collectors """

    @abstractmethod
    def collect(self, n_rollout_steps: int) -> int:
        pass


class RolloutCollector(BaseCollector):
    """ Collects a rollout with every collect call and add the experience into the buffer.

    Args:
        env (VecEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to push rollout experiences
        agent (BaseAgent): Action sampling agent
        logger (DataLogger): Data logger to log environment reward and lengths at termination
        callbacks (Optional[Union[List[BaseCollectorCallback], BaseCollectorCallback]], optional): Collector Callback. Defaults to [] (no callbacks).
    """

    _required_buffer_fields = ("observation", "next_observation", "reward",
                               "termination", "action")

    def __init__(self,
                 env: VecEnv,
                 buffer: BaseBuffer,
                 agent: BaseAgent,
                 logger: DataLogger,
                 callbacks: Optional[Union[List[BaseCollectorCallback],
                                           BaseCollectorCallback]] = None):
        self.env = env
        self.buffer = buffer
        self.agent = agent
        super().__init__(logger)

        for field in self._required_buffer_fields:
            assert field in buffer.struct.names, (
                "Buffer does not contain the field name {}".format(field))

        self.num_timesteps = 0
        self._last_obs = self.env.reset()

        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks] if callbacks is not None else []
        self.callbacks = callbacks

    def _init_default_loggers(self) -> None:
        loggers = {
            "scalar/collector/env_reward": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/collector/env_length": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "histogram/actions": HistListDataLog(n_bins=10, reduce_fn=lambda values: {"action": np.stack(values)}),
        }
        self.logger.add_if_not_exists(loggers)

    def collect(self, n_rollout_steps: int) -> int:
        """ Collect a rollout of experience using the agent and load them to
        buffer

        Args:
            n_rollout_steps (int): Length of the rollout

        Returns:
            int: Total number of time steps passed.
        """

        n_steps = 0
        for callback in self.callbacks:
            callback.on_rollout_start(locals())

        while n_steps < n_rollout_steps:

            actions, policy_content = self.get_actions()

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

            self.buffer.push({
                "observation": self._last_obs,
                "next_observation": next_obs,
                "reward": rewards,
                "termination": dones,
                "action": actions,
                **policy_content
            })

            # Log environment info
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    getattr(self.logger, "scalar/collector/env_reward").push(maybe_ep_info["r"])
                    getattr(self.logger, "scalar/collector/env_length").push(maybe_ep_info["l"])
            # Logging actions for histogram
            getattr(self.logger, "histogram/actions").push(actions)

            self._last_obs = new_obs

            for callback in self.callbacks:
                callback.on_rollout_step(locals())

        for callback in self.callbacks:
            callback.on_rollout_end(locals())

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
