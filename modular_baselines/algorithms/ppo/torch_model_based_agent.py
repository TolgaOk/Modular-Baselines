from typing import Tuple, Union, Dict, Generator, Optional, Any, List, Callable
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Space
from copy import deepcopy
from functools import partial

from modular_baselines.algorithms.ppo.torch_agent import TorchPPOAgent
from modular_baselines.algorithms.ppo.torch_lstm_agent import TorchLstmPPOAgent
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, ListDataLog
from modular_baselines.algorithms.ppo.torch_gae import calculate_diff_gae
from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.loggers.data_logger import SequenceNormDataLog


class TorchModelBasedAgent(TorchPPOAgent):

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


class TorchValueGradientAgent(TorchModelBasedAgent):

    def __init__(self,
                 policy: torch.nn.Module,
                 model: torch.nn.Module,
                 policy_optimizer: torch.optim.Optimizer,
                 model_optimizer: torch.optim.Optimizer,
                 observation_space: Space,
                 action_space: Space,
                 logger: DataLogger) -> None:
        super().__init__(policy,
                         model,
                         policy_optimizer,
                         model_optimizer,
                         observation_space,
                         action_space,
                         logger)
        self.target_policy = deepcopy(policy)

    def sample_action(self, observation: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return TorchModelBasedAgent.sample_action(self, observation)

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        return self.to_torch([
            sample["observation"],
            sample["action"],
            sample["next_observation"],
            sample["termination"],
            sample["reward"],
            sample["old_log_prob"],
            sample["reward_rms_var"],
            sample["obs_rms_mean"],
            sample["obs_rms_var"],
        ])

    def reparameterize(self,
                       th_obs: torch.Tensor,
                       th_acts: torch.Tensor,
                       th_next_obs: torch.Tensor,
                       th_dones: torch.Tensor,
                       ) -> Tuple[torch.Tensor]:
        rollout_len = th_obs.shape[1]
        obs_list = []
        act_list = []
        next_obs_list = []
        pi_param_list = []
        # value_list = []
        re_obs = th_obs[:, 0]
        re_obs.requires_grad = True

        th_pi_params, th_values = self.policy(th_obs)  # Use sampled obs

        for step in range(rollout_len):
            # obs = th_obs[:, step]
            act = th_acts[:, step]
            next_obs = th_next_obs[:, step]
            done = th_dones[:, step]
            pi_param = th_pi_params[:, step]

            # pi_param, value = self.policy(obs)
            pi_dist = self.policy.dist(pi_param)
            re_act = self._normal_reparam(act, pi_dist)

            model_param = self.model(re_obs, re_act)  # Use reparam obs and acts
            model_dist = self.model.dist(model_param)
            re_next_obs = self._normal_reparam(next_obs, model_dist)

            # For forward values
            getattr(self.logger, f"dict/time_sequence/forward_state").add(
                            step, re_obs.norm(dim=-1, p="fro").mean(0).item())
            # For gradients
            re_obs.register_hook(
                partial(lambda time, grad:
                        getattr(self.logger, f"dict/time_sequence/grad_state").add(time, grad.norm(dim=-1, p="fro").mean(0).item()), step))

            obs_list.append(re_obs)
            act_list.append(re_act)
            next_obs_list.append(re_next_obs)
            pi_param_list.append(pi_param)

            if step != (rollout_len - 1):
                th_obs[:, step + 1].requires_grad = True
                re_obs = (1 - done) * re_next_obs + done * th_obs[:, step + 1]

        return (*[
            torch.stack(tensor_list, dim=1) for tensor_list in
            (obs_list, act_list, next_obs_list)
        ], th_pi_params, th_values)

    @staticmethod
    def _normal_reparam(value: torch.Tensor, dist: torch.distributions.Distribution) -> torch.Tensor:
        z_value = (value - dist.mean) / dist.stddev
        return dist.mean + dist.stddev * z_value.detach()

    def rollout_loader(self,
                       sample: np.ndarray,
                       batch_size: int,
                       mini_rollout_size: int,
                       ) -> torch.Tensor:
        rollout_data = self.rollout_to_torch(sample)
        mini_rollout_data = TorchLstmPPOAgent._make_mini_rollout(self,
                                                                 rollout_data, mini_rollout_size=mini_rollout_size)
        perm_indices = torch.randperm(mini_rollout_data[0].shape[0])
        for indices in perm_indices.split(batch_size):
            yield TorchLstmPPOAgent._take_slice(self, mini_rollout_data, indices=indices)

    def target_policy_update(self):
        self.target_policy.load_state_dict(self.policy.state_dict())

    def calculate_reward(self,
                         reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                         obs: torch.Tensor,
                         act: torch.Tensor,
                         next_obs: torch.Tensor,
                         reward_rms_var: torch.Tensor,
                         obs_rms_mean: torch.Tensor,
                         obs_rms_var: torch.Tensor,
                         epsilon: float = 1e-8
                         ) -> torch.Tensor:
        obs_rms_std = torch.sqrt(obs_rms_var + epsilon)
        orig_obs = obs * obs_rms_std + obs_rms_mean
        orig_next_obs = next_obs * obs_rms_std + obs_rms_mean

        return reward_fn(orig_obs, act, orig_next_obs) / torch.sqrt(reward_rms_var + epsilon)

    def update_policy_parameters(self,
                                 sample: np.ndarray,
                                 reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                                 value_coef: float,
                                 ent_coef: float,
                                 gamma: float,
                                 gae_lambda: float,
                                 epochs: int,
                                 lr: float,
                                 clip_value: float,
                                 batch_size: int,
                                 mini_rollout_size: int,
                                 max_grad_norm: float,
                                 normalize_advantage: bool,
                                 check_reward_consistency: bool,
                                 use_log_likelihood: bool,
                                 check_gae: bool = False,
                                 use_reparameterization: bool = True
                                 ) -> Dict[str, float]:
        env_size, rollout_size = sample["observation"].shape[:2]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.target_policy_update()

        for epoch in range(epochs):
            for th_obs, th_act, th_next_obs, th_done, sampled_reward, th_old_log_p, reward_rms_var, obs_rms_mean, obs_rms_var in self.rollout_loader(
                    sample, batch_size=batch_size, mini_rollout_size=mini_rollout_size):

                if use_reparameterization:
                    re_obs, re_acts, re_next_obs, policy_params, values = self.reparameterize(
                        th_obs, th_act, th_next_obs, th_done)
                    rewards = self.calculate_reward(
                        reward_fn, re_obs, re_acts, re_next_obs, reward_rms_var, obs_rms_mean, obs_rms_var)
                    obs = re_obs
                    next_obs = re_next_obs
                else:
                    policy_params, values = self.policy(th_obs)
                    rewards = self.calculate_reward(
                        reward_fn, th_obs, th_act, th_next_obs, reward_rms_var, obs_rms_mean, obs_rms_var)
                    obs = th_obs
                    next_obs = th_next_obs


                _, old_values = self.target_policy(obs)  # Must contain the time dimension
                _, old_last_value = self.target_policy(next_obs[:, -1])

                if check_reward_consistency:
                    if not torch.allclose(rewards, sampled_reward, atol=1e-3):
                        raise RuntimeError("Mismatch between reward function and sampled rewards")

                advantages, returns = calculate_diff_gae(
                    rewards, th_done, old_values, old_last_value, gamma=gamma, gae_lambda=gae_lambda)

                if check_gae:
                    np_advantages, np_returns = calculate_gae(
                        *[tensor.cpu().detach().numpy()
                          for tensor in [sampled_reward, th_done, old_values, old_last_value]],
                        gamma=gamma, gae_lambda=gae_lambda
                    )
                    if not np.allclose(advantages.cpu().detach().numpy(), np_advantages, atol=1e-5):
                        raise RuntimeError("Mismatch advantages!")
                    if not np.allclose(returns.cpu().detach().numpy(), np_returns, atol=1e-5):
                        raise RuntimeError("Mismatch returns!")

                flat_act, flat_old_log_prob, flat_advantages, flat_returns, flat_param, flat_value = self.flatten_time(
                    [th_act, th_old_log_p, advantages, returns, policy_params, values]
                )

                policy_dist = self.policy.dist(flat_param)
                log_probs = policy_dist.log_prob(flat_act).unsqueeze(-1)
                entropies = policy_dist.entropy().unsqueeze(-1)
                ratio = torch.exp(log_probs - flat_old_log_prob.detach())

                value_loss = torch.nn.functional.mse_loss(flat_value, flat_returns)
                entropy_loss = -entropies.mean()

                if use_log_likelihood:
                    policy_loss = self.log_likelihood_loss(
                        ratio=ratio,
                        advantages=flat_advantages,
                        normalize_advantage=normalize_advantage,
                        clip_value=clip_value,
                    )
                else:
                    policy_loss = self.value_gradient_loss(
                        ratio=ratio,
                        returns=flat_returns,
                        clip_value=clip_value,
                    )
                entropy_loss = -entropies.mean()
                loss = policy_loss + value_loss * value_coef + entropy_loss * ent_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                getattr(self.logger, "scalar/agent/value_loss").push(value_loss.item())
                getattr(self.logger, "scalar/agent/policy_loss").push(policy_loss.item())
                getattr(self.logger, "scalar/agent/entropy_loss").push(entropy_loss.item())
                getattr(self.logger, "scalar/agent/approxkl").push(((ratio - 1) - ratio.log()).mean().item())
                getattr(self.logger, "scalar/agent/ratio").push(ratio.mean().item())
        getattr(self.logger, "scalar/agent/clip_range").push(clip_value)
        getattr(self.logger, "scalar/agent/learning_rate").push(lr)

    def value_gradient_loss(self,
                            ratio: torch.Tensor,
                            returns: torch.Tensor,
                            clip_value: float,
                            ) -> torch.Tensor:
        # maximize return that is formed using rewards and re-parameterized states

        ratio = ratio.detach()
        surrogate_loss_1 = returns * ratio
        surrogate_loss_2 = torch.clamp(returns * ratio,
                                       (1 - clip_value) * returns.detach(),
                                       (1 + clip_value) * returns.detach())
        policy_loss = -torch.minimum(surrogate_loss_1, surrogate_loss_2).mean()
        return policy_loss

    def log_likelihood_loss(self,
                            ratio: torch.Tensor,
                            advantages: torch.Tensor,
                            normalize_advantage: bool,
                            clip_value: float,
                            ) -> torch.Tensor:
        advantages = advantages.detach()

        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

        surrogate_loss_1 = advantages * ratio
        surrogate_loss_2 = advantages * torch.clamp(ratio, 1 - clip_value, 1 + clip_value)
        policy_loss = -torch.minimum(surrogate_loss_1, surrogate_loss_2).mean()
        return policy_loss

    def _init_default_loggers(self) -> None:
        super()._init_default_loggers()
        loggers = {
            f"dict/time_sequence/{group}_state": SequenceNormDataLog()
            for group in ("grad", "forward")
        }
        loggers["scalar/agent/ratio"] = ListDataLog(reduce_fn=lambda values: np.mean(values))
        self.logger.add_if_not_exists(loggers)