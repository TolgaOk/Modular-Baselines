from typing import Optional, Any, Dict, Tuple, Union, List, Callable
from abc import ABC, abstractmethod
import numpy as np
from torch.types import Device
import torch
from gym.spaces import Space, Discrete, Box

from modular_baselines.loggers.logger import MBLogger
from modular_baselines.utils.utils import to_torch


# TODO: Replace Base class with Protocol
class BaseAgent(ABC):

    def __init__(self, observation_space: Space, action_space: Space, logger: MBLogger) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logger

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[np.ndarray, Union[np.ndarray, None], Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class TorchAgent(BaseAgent):
    # TODO: Add discrete action space support

    def __init__(self,
                 network: torch.nn.Module,
                 observation_space: Space,
                 action_space: Space,
                 logger: MBLogger) -> None:
        self.network = network
        super().__init__(observation_space, action_space, logger)

    @property
    def device(self) -> Device:
        return next(iter(self.network.parameters())).device

    def param_dict_as_numpy(self) -> Dict[str, np.ndarray]:
        return {name: param.detach().cpu().numpy()
                for name, param in self.network.named_parameters()}

    def grad_dict_as_numpy(self) -> Dict[str, np.ndarray]:
        return {name: param.grad.cpu().numpy() if param.grad is not None else param.grad
                for name, param in self.network.named_parameters}

    def save(self, path: str) -> None:
        torch.save({
            "agent_state_dict": self.network.state_dict(),
        }, path)

    def forward(self,
                observation: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_params, value = self.network(observation)
        return policy_params, value

    def dist(self,
             parameters: torch.Tensor
             ) -> Union[torch.distributions.Normal, torch.distributions.Categorical]:
        if isinstance(self.action_space, Box):
            mean, std_logit = parameters.split(parameters.shape[-1] // 2, dim=1)
            std = torch.nn.functional.softplus(std_logit)
            dist = torch.distributions.Normal(loc=mean, scale=std + 0.05)
            dist = torch.distributions.independent.Independent(dist, 1)
            return dist

        if isinstance(self.action_space, Discrete):
            dist = torch.distributions.Categorical(logits=parameters)
            return dist

        raise ValueError(f"Action space: {self.action_space.__class__.__name__} is not supported!")

    def sample_action(self,
                      observation: np.ndarray,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            th_observation = to_torch(
                self.device, observation).float()

            policy_params, _ = self.forward(th_observation)
            policy_dist = self.dist(policy_params)
            th_action = policy_dist.sample()
            log_prob = policy_dist.log_prob(th_action).unsqueeze(-1)
            if isinstance(self.action_space, Discrete):
                # th_action = th_action.unsqueeze(-1)
                raise NotImplementedError(
                    f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
        return th_action.cpu().numpy(), {"old_log_prob": log_prob.cpu().numpy()}
