from typing import Tuple, Union, Dict, Generator, Any, List
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.agent import TorchAgent
from modular_baselines.loggers.data_logger import ListLog


class TorchA2CAgent(TorchAgent):
    """ Pytorch A2C Agent """

    def init_hidden_state(self, batch_size=None) -> None:
        return None

    def sample_action(self,
                      observation: np.ndarray,
                      ) -> Tuple[np.ndarray, Union[np.ndarray, None], Dict[str, Any]]:

        with torch.no_grad():
            th_observation = self.to_torch(observation).float()

            policy_dist, _ = self.policy(th_observation)
            th_action = policy_dist.sample()
            if isinstance(self.action_space, Discrete):
                th_action = th_action.unsqueeze(-1)
        return th_action.cpu().numpy(), {}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_next_obs = [
            self.to_torch(array) for array in
            (sample["observation"],
             sample["action"],
             sample["next_observation"][:, -1])]
        return th_obs, th_action, th_next_obs

    def prepare_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_next_obs = self.rollout_to_torch(
            sample)

        policy_dist, th_flatten_values = self.policy(self.flatten_time(th_obs))

        tf_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, th_next_value = self.policy(th_next_obs)
        advantages, returns = self.to_torch(calculate_gae(
                rewards=sample["reward"],
                terminations=sample["termination"],
                values=tf_values.detach().cpu().numpy(),
                last_value=th_next_value.detach().cpu().numpy(),
                gamma=gamma,
                gae_lambda=gae_lambda)
        )

        return (*self.flatten_time([advantages, returns, th_action]), policy_dist, th_flatten_values)

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          lr: float,
                          max_grad_norm: float,
                          normalize_advantage: bool,
                          ) -> Dict[str, float]:
        """ Pytorch A2C parameter update method """
        env_size, rollout_size = sample["observation"].shape[:2]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        advantages, returns, th_action, policy_dist, th_values = self.prepare_rollout(
            sample, gamma=gamma, gae_lambda=gae_lambda)

        log_probs = policy_dist.log_prob((th_action))
        entropies = policy_dist.entropy()

        values, advantages, returns, log_probs, entropies = map(
            lambda tensor: tensor.reshape(env_size * rollout_size, 1),
            (th_values, advantages, returns, log_probs, entropies))

        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss = torch.nn.functional.mse_loss(values, returns)
        policy_loss = (-log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        loss = value_loss * value_coef + policy_loss + entropy_loss * ent_coef

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()

        self.logger.value_loss.push(value_loss.item())
        self.logger.policy_loss.push(policy_loss.item())
        self.logger.entropy_loss.push(entropy_loss.item())
        self.logger.learning_rate.push(lr)

        return dict()

    def _init_default_loggers(self) -> None:
        super()._init_default_loggers()
        loggers = dict(
            value_loss=ListLog(formatting=lambda value: np.mean(value)),
            policy_loss=ListLog(formatting=lambda value: np.mean(value)),
            entropy_loss=ListLog(formatting=lambda value: np.mean(value)),
            learning_rate=ListLog(formatting=lambda values: np.max(values)),
        )
        self.logger.add_if_not_exists(loggers)
