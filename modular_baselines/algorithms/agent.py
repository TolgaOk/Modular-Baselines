from typing import Optional, Any, Dict, Tuple, Union
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
    def init_hidden_state(self, batch_size: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[np.ndarray, Union[np.ndarray, None], Dict[str, Any]]:
        pass

    @abstractmethod
    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
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


class TorchAgent(BaseAgent):

    def __init__(self, policy: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 observation_space: Space,
                 action_space: Space,
                 logger: DataLogger) -> None:
        super().__init__(observation_space, action_space, logger)
        self.policy = policy
        self.optimizer = optimizer

    @property
    def device(self) -> Device:
        return next(iter(self.policy.parameters())).device

    def init_hidden_state(self, batch_size=None) -> None:
        return None

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()