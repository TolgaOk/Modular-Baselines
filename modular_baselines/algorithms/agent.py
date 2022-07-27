from typing import Optional, Any, Dict, Tuple, Union, List, Callable
from abc import ABC, abstractmethod
import numpy as np
from torch.types import Device
import torch
from gym.spaces import Space

from modular_baselines.component import Component
from modular_baselines.loggers.data_logger import DataLogger


class BaseAgent(Component):

    def __init__(self, observation_space: Space, action_space: Space, logger: DataLogger) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__(logger)

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[np.ndarray, Union[np.ndarray, None], Dict[str, Any]]:
        pass

    @abstractmethod
    def update_parameters(self, sample: np.ndarray) -> Dict[str, float]:
        """ Update policy parameters using the given sample and a2c update mechanism.

            Args:
                sample (np.ndarray): Sample that contains at least observation, next_observation,
                    reward, termination, action, and policy_state if the policy is a recurrent policy.

            Returns:
                Dict[str, float]: Dictionary of losses to log
        """
        pass

    @abstractmethod
    def train_mode(self):
        pass

    @abstractmethod
    def eval_mode(self):
        pass


class BaseRecurrentAgent(Component):

    @abstractmethod
    def init_hidden_state(self, batch_size: Optional[int] = None) -> Any:
        pass


def nested(function: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]):
    def nested_apply(self, collection: Union[np.ndarray, Dict[str, Any], List[Any], Tuple[Any]], **kwargs):
        if isinstance(collection, dict):
            return {name: nested_apply(self, value, **kwargs) for name, value in collection.items()}
        if isinstance(collection, (list, tuple)):
            cls = type(collection)
            return cls([nested_apply(self, value, **kwargs) for value in collection])
        if isinstance(collection, (torch.Tensor, np.ndarray)):
            return function(self, collection, **kwargs)
        raise ValueError(f"Type {type(collection)} is not supported!")
    return nested_apply


class TorchAgent(BaseAgent):

    def __init__(self,
                 policy: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 observation_space: Space,
                 action_space: Space,
                 logger: DataLogger) -> None:
        self.policy = policy
        self.optimizer = optimizer
        super().__init__(observation_space, action_space, logger)

    @property
    def device(self) -> Device:
        return next(iter(self.policy.parameters())).device

    def train_mode(self):
        self.policy.train(True)

    def eval_mode(self):
        self.policy.train(False)

    @nested
    def to_torch(self, ndarray: np.ndarray):
        return torch.from_numpy(ndarray).to(self.device)

    @nested
    def flatten_time(self, tensor: torch.Tensor) -> torch.Tensor:
        n_envs, n_rollout = tensor.shape[:2]
        return tensor.reshape(n_envs * n_rollout, *tensor.shape[2:])

    def param_dict_as_numpy(self) -> Dict[str, np.ndarray]:
        return {name: param.detach().cpu().numpy()
                for name, param in self.policy.named_parameters()}
            