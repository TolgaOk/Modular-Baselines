from abc import ABC, abstractmethod
from typing import List, Optional, Union


from modular_baselines.collectors.collector import BaseCollector
from modular_baselines.policies.policy import BasePolicy


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


class BaseAlgorithm(ABC):
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
        policy (torch.nn.Module): Poliicy module with both heads
        buffer (BaseBuffer): Experience Buffer
        collector (BaseCollector): Experience collector
        env (VecEnv): Vectorized environment
        rollout_len (int): n-step length
        callbacks (List[BaseAlgorithmCallback], optional): Algorithm callbacks. Defaults to [].
        device (str, optional): Torch device. Defaults to "cpu".
    """

    def __init__(self,
                 policy: BasePolicy,
                 collector: BaseCollector,
                 rollout_len: int,
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        self.policy = policy
        self.collector = collector
        self.rollout_len = rollout_len

        self.buffer = self.collector.buffer
        self.num_envs = self.collector.env.num_envs

        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks] if callbacks is not None else []
        self.callbacks = callbacks

    def learn(self, total_timesteps: int) -> None:
        """ Main loop for running the on policy algorithm

        Args:
            total_timesteps (int): Total environment timesteps to run
        """

        num_timesteps = 0
        iteration = 0

        for callback in self.callbacks:
            callback.on_training_start(locals())

        while num_timesteps < total_timesteps:

            num_timesteps = self.collector.collect(self.rollout_len)
            loss_dict = self.train()
            iteration += 1
            for callback in self.callbacks:
                callback.on_step(locals())

        for callback in self.callbacks:
            callback.on_training_end(locals())

        return None
