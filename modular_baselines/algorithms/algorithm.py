from abc import ABC, abstractmethod
from typing import List, Optional, Union
from time import time
import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.component import Component
from modular_baselines.collectors.collector import BaseCollector
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, LastDataLog, ListDataLog


class BaseAlgorithmCallback(ABC):
    """ Base class for buffer callbacks that only supports:
    on_training_start, _on_training_start, and on_training_end calls.
    """

    @abstractmethod
    def on_training_start(self, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_step(self, *args) -> bool:
        pass

    @abstractmethod
    def on_training_end(self, *args) -> None:
        pass


class BaseAlgorithm(Component):
    """ Base abstract class for Algorithms """

    @abstractmethod
    def learn(self, total_timesteps: int) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass


class OnPolicyAlgorithm(BaseAlgorithm):
    """ Base on policy learning algorithm

    Args:
        agent (BaseAgent): Agent to be learned
        buffer (BaseBuffer): Experience Buffer
        collector (BaseCollector): Experience collector
        env (VecEnv): Vectorized environment
        rollout_len (int): n-step length
        callbacks (List[BaseAlgorithmCallback], optional): Algorithm callbacks. Defaults to [].
        device (str, optional): Torch device. Defaults to "cpu".
    """

    def __init__(self,
                 agent: BaseAgent,
                 collector: BaseCollector,
                 rollout_len: int,
                 logger: DataLogger,
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        self.agent = agent
        self.collector = collector
        self.rollout_len = rollout_len
        super().__init__(logger)

        self.buffer = self.collector.buffer
        self.num_envs = self.collector.env.num_envs

        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks] if callbacks is not None else []
        self.callbacks = callbacks

    def learn(self, total_timesteps: int) -> None:
        """ Main loop for running the on-policy algorithm

        Args:
            total_timesteps (int): Total environment time steps to run
        """

        train_start_time = time()
        num_timesteps = 0
        iteration = 0

        for callback in self.callbacks:
            callback.on_training_start(locals())

        while num_timesteps < total_timesteps:
            iteration_start_time = time()

            num_timesteps = self.collector.collect(self.rollout_len)
            self.train()
            iteration += 1

            getattr(self.logger, "scalar/algorithm/iteration").push(iteration)
            getattr(self.logger, "scalar/algorithm/timesteps").push(num_timesteps)
            getattr(self.logger, "scalar/algorithm/time_elapsed").push(time() - train_start_time)
            getattr(self.logger, "scalar/algorithm/fps").push((time() -
                                                               iteration_start_time) / (self.num_envs * self.rollout_len))

            for callback in self.callbacks:
                callback.on_step(locals())

        for callback in self.callbacks:
            callback.on_training_end(locals())

        return None

    def _init_default_loggers(self) -> None:
        loggers = {
            "scalar/algorithm/iteration": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/timesteps": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/time_elapsed": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/fps": ListDataLog(reduce_fn=lambda values: int(1 / np.mean(values))),
        }
        self.logger.add_if_not_exists(loggers)

    @staticmethod
    def _setup(env: VecEnv):
        # TODO: Add different observation spaces
        observation_space = env.observation_space
        # TODO: Add different action spaces
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise NotImplementedError("Only Box observations are available")
        if not isinstance(action_space, (spaces.Box, spaces.Discrete)):
            raise NotImplementedError("Only Discrete and Box actions are available")

        action_dim = action_space.shape[-1] if isinstance(action_space, spaces.Box) else 1
        return observation_space, action_space, action_dim
