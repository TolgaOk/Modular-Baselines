from typing import Tuple, Union, Dict, Generator, Optional, Any, List
from abc import abstractmethod
import numpy as np
import torch
from torch.types import Device
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.ppo.ppo import PPOPolicy


class TorchPPOPolicy(PPOPolicy):
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
    def forward(self,
                observation: torch.Tensor,
                policy_state: Union[None, torch.Tensor],
                ) -> Tuple[torch.distributions.Distribution, Union[torch.Tensor, None], torch.Tensor]:
        pass

    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            th_observation = self.maybe_to_torch(observation).float()
            th_policy_state = self.maybe_to_torch(policy_state)
            th_policy_state = th_policy_state.float() if th_policy_state is not None else None

            policy_dist, policy_state, value = self(th_observation, th_policy_state)
            th_action = policy_dist.sample()
            log_prob = policy_dist.log_prob(th_action).unsqueeze(-1)
            if isinstance(self.action_space, Discrete):
                th_action = th_action.unsqueeze(-1)
        return th_action.cpu().numpy(), policy_state, {"old_log_prob": log_prob.cpu().numpy()}

    def maybe_to_torch(self, ndarray: Union[np.ndarray, None]) -> Union[torch.Tensor, None]:
        return torch.from_numpy(ndarray).to(self.device) if ndarray is not None else None

    def maybe_take_last_time(self, tensor: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
        return tensor[:, -1] if tensor is not None else None

    def maybe_flatten_batch(self, tensor: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
        if tensor is not None:
            n_envs, n_rollout = tensor.shape[:2]
            return tensor.reshape(n_envs * n_rollout, *tensor.shape[2:])
        return None

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        policy_state = sample["policy_state"] if "policy_state" in sample.dtype.names else None
        next_policy_state = sample["next_policy_state"] if "next_policy_state" in sample.dtype.names else None
        th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state, th_old_log_prob = [
            self.maybe_to_torch(array) for array in
            (sample["observation"],
             policy_state,
             sample["action"],
             self.maybe_take_last_time(sample["next_observation"]),
             self.maybe_take_last_time(next_policy_state),
             sample["old_log_prob"])]
        return th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state, th_old_log_prob

    def prepare_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_policy_state, th_action, th_next_obs, th_next_policy_state, th_old_log_prob = self.rollout_to_torch(
            sample)

        _, _, th_flatten_values = self(*map(self.maybe_flatten_batch, (th_obs, th_policy_state)))
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

        return list(map(self.maybe_flatten_batch,
                   (advantages, returns, th_obs, th_policy_state, th_action, th_old_log_prob)))

    def rollout_loader(self, batch_size: int, *tensors: Tuple[torch.Tensor]) -> Generator[Tuple[torch.Tensor], None, None]:
        if len(tensors) == 0:
            raise ValueError("Empty tensors")
        perm_indices = torch.randperm(tensors[0].shape[0])
        for index in range(0, len(perm_indices), batch_size):
            _slice = slice(index, index + batch_size)
            yield tuple([tensor[_slice] if tensor is not None else None for tensor in tensors])

    def init_state(self, batch_size=None):
        # Initialize Policy State. None for non-reccurent models
        return None

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          epochs: int,
                          clip_value: float,
                          batch_size: int,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
        """ Pytorch A2C parameter update method """

        rollout_data = self.prepare_rollout(sample, gamma, gae_lambda)

        value_losses = []
        policy_losses = []
        entropy_losses = []

        for epoch in range(epochs):
            for advantages, returns, obs, policy_state, action, old_log_prob in self.rollout_loader(batch_size, *rollout_data):

                policy_dist, _, values = self(obs, policy_state)
                if isinstance(self.action_space, Discrete):
                    action = action.squeeze(-1)
                log_probs = policy_dist.log_prob(action).unsqueeze(-1)
                entropies = policy_dist.entropy().unsqueeze(-1)

                value_loss = torch.nn.functional.mse_loss(values, returns)

                ratio = torch.exp(log_probs - old_log_prob.detach())
                surrugate_loss_1 = advantages * ratio
                surrugate_loss_2 = advantages * torch.clamp(ratio, 1 - clip_value, 1 + clip_value)
                policy_loss = -torch.minimum(surrugate_loss_1, surrugate_loss_2).mean()
                entropy_loss = -entropies.mean()
                loss = value_loss * value_coef + policy_loss + entropy_loss * ent_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()


                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())

        return dict(value_loss=np.mean(value_losses),
                    policy_loss=np.mean(policy_losses),
                    entropy_loss=np.mean(entropy_losses))
