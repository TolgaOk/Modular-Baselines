from typing import Tuple, Union, Dict, Generator, Optional, Any, List
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Space

from modular_baselines.algorithms.ppo.torch_agent import TorchPPOAgent
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, ListDataLog


class TorchVGAgent(TorchPPOAgent):

    def __init__(self,
                 policy: torch.nn.Module,
                 model: torch.nn.Module,
                 policy_optimizer: torch.optim.Optimizer,
                 model_optimizer: torch.optim.Optimizer,
                 observation_space: Space,
                 action_space: Space,
                 logger: DataLogger) -> None:
        BaseAgent.__init__(self, observation_space, action_space, logger)
        self.policy = policy
        self.model = model
        self.optimizer = policy_optimizer
        self.model_optimizer = model_optimizer

    def update_parameters(self,
                          recent_sample: np.ndarray,
                          random_samples: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          policy_epochs: int,
                          model_epochs: int,
                          clip_value: float,
                          policy_batch_size: int,
                          policy_lr: float,
                          model_lr: float,
                          max_grad_norm: float,
                          normalize_advantage: bool
                          ) -> Dict[str, float]:
        self.update_model_parameters(samples=random_samples,
                                     max_grad_norm=max_grad_norm,
                                     lr=model_lr)
        self.update_policy_parameters(
            sample=recent_sample,
            value_coef=value_coef,
            ent_coef=ent_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            epochs=policy_epochs,
            lr=policy_lr,
            clip_value=clip_value,
            batch_size=policy_batch_size,
            max_grad_norm=max_grad_norm,
            normalize_advantage=normalize_advantage,
        )

    def update_policy_parameters(self, **kwargs) -> Dict[str, float]:
        super().update_parameters(**kwargs)

    def update_model_parameters(self,
                                samples: np.ndarray,
                                max_grad_norm,
                                lr: float,
                                ) -> None:
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = lr

        for index, sample in enumerate(samples):
            th_obs, th_acts, th_next_obs = self.flatten_time(
                self.to_torch([
                    sample["observation"],
                    sample["action"],
                    sample["next_observation"],
                ]))

            parameters = self.model(th_obs, th_acts)
            dist = self.model.dist(parameters)

            log_prob = dist.log_prob(th_next_obs)
            loss = -log_prob.mean(0)
            self.model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.model_optimizer.step()

            getattr(self.logger, "scalar/model/log_likelihood_loss").push(loss.item())
            getattr(self.logger, "scalar/model/mae").push(
                ((th_next_obs - dist.mean).abs().mean(dim=-1)).mean(0).item()
            )

    def _init_default_loggers(self) -> None:
        super()._init_default_loggers()
        loggers = {
            "scalar/model/log_likelihood_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/model/mae": ListDataLog(reduce_fn=lambda values: np.mean(values)),
        }
        self.logger.add_if_not_exists(loggers)

    def train_mode(self):
        self.policy.train(True)
        self.model.train(True)

    def eval_mode(self):
        self.policy.train(False)
        self.model.train(False)

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "policy_optimizer_state_dict": self.optimizer.state_dict(),
            "model_optimizer_state_dict": self.model_optimizer.state_dict(),
        },
            path)