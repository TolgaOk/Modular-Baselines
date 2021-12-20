from typing import Tuple, Union, Dict, Generator
from abc import abstractmethod
import numpy as np
import torch
from torch.types import Device

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.a2c.a2c import A2CPolicy


class TorchA2CPolicy(A2CPolicy):
    """ Pytorch A2C Policy base class """

    @property
    @abstractmethod
    def device(self) -> Device:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def evaluate_rollout(self,
                         observation: torch.Tensor,
                         policy_state: Union[None, torch.Tensor],
                         action: torch.Tensor,
                         last_next_obseration: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass the given rollout. This is useful for both the advantage calculation
        and bacward pass. Note: BS -> batch size, R: rollout length

        Args:
            observation (torch.Tensor): Observation tensor with the shape (BS, R, *)
            policy_state (Union[None, torch.Tensor]): Policy state for reccurent models. None
                will be given if the buffer does not contain "policy_state" field.
            action (torch.Tensor): Action tensor with the shape (BS, R, *)
            last_next_obseration (torch.Tensor): [description]: Last observation tensor to
                calculate last value with the shape: (BS, *D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: values,
                log_probs, entropies, last_value
        """
        pass

    def maybe_to_torch(self, ndarray: np.ndarray):
        return torch.from_numpy(ndarray).to(self.device) if ndarray is not None else None

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
        """ Pytorch A2C parameter update method """
        batch_size, rollout_size = sample.shape
        policy_state = sample["policy_state"] if "policy_state" in sample.dtype.names else None
        values, log_probs, entropies, last_value = self.evaluate_rollout(
            *map(self.maybe_to_torch,
                 (sample["observation"],
                  policy_state,
                  sample["action"],
                  sample["next_observation"][:, -1]))
        )

        advantages, returns = map(
            self.maybe_to_torch,
            calculate_gae(
                rewards=sample["reward"],
                terminations=sample["termination"],
                values=values.detach().cpu().numpy(),
                last_value=last_value.detach().cpu().numpy(),
                gamma=gamma,
                gae_lambda=gae_lambda)
        )

        values, advantages, returns, log_probs, entropies = map(
            lambda tensor: tensor.reshape(batch_size * rollout_size, 1),
            (values, advantages, returns, log_probs, entropies))

        value_loss = torch.nn.functional.mse_loss(values, returns)
        policy_loss = (-log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        loss = value_loss * value_coef + policy_loss + entropy_loss * ent_coef

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

        return dict(value_loss=value_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy_loss=entropy_loss.item())
