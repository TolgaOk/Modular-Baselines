from typing import Tuple, Union, Dict, Generator
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Discrete
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

    def maybe_to_torch(self, ndarray: np.ndarray):
        return torch.from_numpy(ndarray).to(self.device) if ndarray is not None else None

    def maybe_flatten_batch(self, tensor: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
        if tensor is not None:
            n_envs, n_rollout = tensor.shape[:2]
            return tensor.reshape(n_envs * n_rollout, *tensor.shape[2:])
        return None

    def maybe_take_last_time(self, tensor: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
        return tensor[:, -1] if tensor is not None else None


    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            th_observation = self.maybe_to_torch(observation).float()
            th_policy_state = self.maybe_to_torch(policy_state)
            th_policy_state = th_policy_state.float() if th_policy_state is not None else None

            policy_dist, policy_state, _ = self(th_observation, th_policy_state)
            th_action = policy_dist.sample()
            if isinstance(self.action_space, Discrete):
                th_action = th_action.unsqueeze(-1)
        return th_action.cpu().numpy(), policy_state, {}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        policy_state = sample["policy_state"] if "policy_state" in sample.dtype.names else None
        next_policy_state = sample["next_policy_state"] if "next_policy_state" in sample.dtype.names else None
        th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state = [
            self.maybe_to_torch(array) for array in
            (sample["observation"],
             policy_state,
             sample["action"],
             self.maybe_take_last_time(sample["next_observation"]),
             self.maybe_take_last_time(next_policy_state))]
        return th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
        """ Pytorch A2C parameter update method """
        env_size, rollout_size = sample["observation"].shape[:2]

        th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state = self.rollout_to_torch(
            sample)

        policy_dist, _, th_flatten_values = self(*map(self.maybe_flatten_batch, (th_obs, th_policy_state)))
        tf_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, _, th_next_values = self(th_next_obs, th_next_policy_state)

        advantages, returns = map(
            self.maybe_to_torch,
            calculate_gae(
                rewards=sample["reward"],
                terminations=sample["termination"],
                values=tf_values.detach().cpu().numpy(),
                last_value=th_next_values.detach().cpu().numpy(),
                gamma=gamma,
                gae_lambda=gae_lambda)
        )

        log_probs = policy_dist.log_prob(self.maybe_flatten_batch(th_action))
        entropies = policy_dist.entropy()

        values, advantages, returns, log_probs, entropies = map(
            lambda tensor: tensor.reshape(env_size * rollout_size, 1),
            (tf_values, advantages, returns, log_probs, entropies))

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
