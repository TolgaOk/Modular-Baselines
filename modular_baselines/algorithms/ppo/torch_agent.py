from typing import Tuple, Union, Dict, Generator, Optional, Any, List
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.agent import TorchAgent
from modular_baselines.loggers.data_logger import ListDataLog


class TorchPPOAgent(TorchAgent):
    """ Pytorch PPO Agent """

    def sample_action(self,
                      observation: np.ndarray,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            th_observation = self.to_torch(observation).float()

            policy_params, _ = self.policy(th_observation)
            policy_dist = self.policy.dist(policy_params)
            th_action = policy_dist.sample()
            log_prob = policy_dist.log_prob(th_action).unsqueeze(-1)
            if isinstance(self.action_space, Discrete):
                # th_action = th_action.unsqueeze(-1)
                raise NotImplementedError(
                    f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
        return th_action.cpu().numpy(), {"old_log_prob": log_prob.cpu().numpy()}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_next_obs, th_old_log_prob = self.to_torch([
            sample["observation"],
            sample["action"],
            sample["next_observation"][:, -1],
            sample["old_log_prob"]])
        return th_obs, th_action, th_next_obs, th_old_log_prob

    def prepare_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_next_obs, th_old_log_prob = self.rollout_to_torch(sample)

        _, th_flatten_values = self.policy(self.flatten_time(th_obs))
        th_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, th_next_value = self.policy(th_next_obs)

        advantages, returns = self.to_torch(calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=th_values.detach().cpu().numpy(),
            last_value=th_next_value.detach().cpu().numpy(),
            gamma=gamma,
            gae_lambda=gae_lambda)
        )

        return (advantages, returns, th_action, th_old_log_prob, th_obs)

    def rollout_loader(self, batch_size: int, *tensors: Tuple[torch.Tensor]
                       ) -> Generator[Tuple[torch.Tensor], None, None]:
        if len(tensors) == 0:
            raise ValueError("Empty tensors")
        flatten_tensors = [self.flatten_time(tensor) for tensor in tensors]
        perm_indices = torch.randperm(flatten_tensors[0].shape[0])

        for indices in perm_indices.split(batch_size):
            yield tuple([tensor[indices] for tensor in flatten_tensors])

    def replay_rollout(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_params, values = self.policy(obs)
        return policy_params, values

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          epochs: int,
                          lr: float,
                          clip_value: float,
                          batch_size: int,
                          max_grad_norm: float,
                          normalize_advantage: bool,
                          ) -> Dict[str, float]:
        """ Pytorch PPO parameter update method """

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        rollout_data = self.prepare_rollout(sample, gamma, gae_lambda)

        for epoch in range(epochs):
            for advantages, returns, action, old_log_prob, *replay_data in self.rollout_loader(batch_size, *rollout_data):

                if normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_params, values = self.replay_rollout(*replay_data)
                policy_dist = self.policy.dist(policy_params)
                if isinstance(self.action_space, Discrete):
                    # action = action.squeeze(-1)
                    raise NotImplementedError(
                        f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
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
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                getattr(self.logger, "scalar/agent/value_loss").push(value_loss.item())
                getattr(self.logger, "scalar/agent/policy_loss").push(policy_loss.item())
                getattr(self.logger, "scalar/agent/entropy_loss").push(entropy_loss.item())
                getattr(self.logger, "scalar/agent/approxkl").push(((ratio - 1) - ratio.log()).mean().item())
        getattr(self.logger, "scalar/agent/clip_range").push(clip_value)
        getattr(self.logger, "scalar/agent/learning_rate").push(lr)

        return dict()

    def _init_default_loggers(self) -> None:
        super()._init_default_loggers()
        loggers = {
            "scalar/agent/value_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/policy_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/entropy_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/learning_rate": ListDataLog(reduce_fn=lambda values: np.max(values)),
            "scalar/agent/clip_range": ListDataLog(reduce_fn=lambda values: np.max(values)),
            "scalar/agent/approxkl": ListDataLog(reduce_fn=lambda values: np.mean(values)),
        }
        self.logger.add_if_not_exists(loggers)
