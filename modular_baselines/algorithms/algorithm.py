from abc import ABC, abstractmethod
from typing import List, Optional, Union
from time import time
import numpy as np

from modular_baselines.component import Component
from modular_baselines.collectors.collector import BaseCollector
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, DataLog, ListLog


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

            self.logger.iteration.push(iteration)
            self.logger.timesteps.push(num_timesteps)
            self.logger.time_elapsed.push(time() - train_start_time)
            self.logger.fps.push((time() - iteration_start_time) / (self.num_envs * self.rollout_len))

            for callback in self.callbacks:
                callback.on_step(locals())

        for callback in self.callbacks:
            callback.on_training_end(locals())

        return None
        
    def _init_default_loggers(self) -> None:
        loggers = dict(
            iteration = DataLog(formatting=lambda value: value),
            timesteps = DataLog(formatting=lambda value: value),
            time_elapsed = DataLog(formatting=lambda value: value),
            fps = ListLog(formatting=lambda values: int(1 / np.mean(values))),
        )
        self.logger.add_if_not_exists(loggers)
