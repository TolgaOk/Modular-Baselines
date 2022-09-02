from typing import Tuple, Union, Dict, Generator, Optional, Any, List, Callable
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Space
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

from modular_baselines.algorithms.ppo.torch_agent import TorchPPOAgent
from modular_baselines.algorithms.ppo.torch_lstm_agent import TorchLstmPPOAgent
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger, ListDataLog
from modular_baselines.algorithms.ppo.torch_gae import calculate_diff_gae
from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.loggers.data_logger import SequenceNormDataLog, ParamHistDataLog


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

            embed_obs = self.model.immersion(th_obs)
            parameters = self.model(embed_obs, th_acts)
            dist = self.model.dist(parameters)

            log_prob = dist.log_prob(self.model.immersion(th_next_obs))
            loss = -log_prob.mean(0)
            self.model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.model_optimizer.step()

            getattr(self.logger, "scalar/model/log_likelihood_loss").push(loss.item())
            getattr(self.logger, "scalar/model/mae").push(
                ((th_next_obs - self.model.submersion(dist.mean)).abs().mean(dim=-1)).mean(0).item()
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

    def param_dict_as_numpy(self) -> Dict[str, np.ndarray]:
        return {f"{prefix}_{name}": param.detach().cpu().numpy()
                for prefix, parameters in (("policy", self.policy.named_parameters()), ("model", self.model.named_parameters()))
                for name, param in parameters}

    def grad_dict_as_numpy(self) -> Dict[str, np.ndarray]:
        return {f"{prefix}_{name}": param.grad.cpu().numpy() if param.grad is not None else None
                for prefix, parameters in (("policy", self.policy.named_parameters()), ("model", self.model.named_parameters()))
                for name, param in parameters}


@dataclass
class VgRollout:
    obs: torch.Tensor
    act: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    reward: torch.Tensor
    old_log_p: torch.Tensor
    reward_rms_var: torch.Tensor
    obs_rms_mean: torch.Tensor
    obs_rms_var: torch.Tensor
    next_obs_rms_mean: torch.Tensor
    next_obs_rms_var: torch.Tensor


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

    def rollout_to_torch(self, sample: np.ndarray, is_vec_normalize: bool) -> Tuple[torch.Tensor]:

        if is_vec_normalize:
            norm_sample = [
                sample["reward_rms_var"],
                sample["obs_rms_mean"],
                sample["obs_rms_var"],
                sample["next_obs_rms_mean"],
                sample["next_obs_rms_var"]
            ]
        else:
            norm_sample = [
                np.ones_like(sample["reward"]),
                np.zeros_like(sample["observation"]),
                np.ones_like(sample["observation"]),
                np.zeros_like(sample["observation"]),
                np.ones_like(sample["observation"]),
            ]

        return self.to_torch([
            sample["observation"],
            sample["action"],
            sample["next_observation"],
            sample["termination"],
            sample["reward"],
            sample["old_log_prob"],
            *norm_sample
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
        # pi_param_list = []
        # value_list = []
        re_obs = th_obs[:, 0]
        re_obs.requires_grad = True
        re_obs_embed = self.model.immersion(re_obs)
        # re_obs_embed.requires_grad = True

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

            model_param = self.model(re_obs_embed, re_act)  # Use reparam obs and acts
            model_dist = self.model.dist(model_param)
            next_obs_embed = self.model.immersion(next_obs)
            re_next_obs_embed = self._normal_reparam(next_obs_embed, model_dist)
            re_next_obs = self.model.submersion(re_next_obs_embed)

            # Log forward values
            getattr(self.logger, f"dict/time_sequence/forward_state").add(
                step, re_obs.norm(dim=-1, p="fro").mean(0).item())
            # Log gradients w.r.t embed
            re_obs_embed.register_hook(
                partial(lambda time, grad:
                        getattr(self.logger, f"dict/time_sequence/grad_embed").add(
                            time, grad.norm(dim=-1, p="fro").mean(0).item()), step))
            # Log gradients w.r.t observation
            re_obs.register_hook(
                partial(lambda time, grad:
                        getattr(self.logger, f"dict/time_sequence/grad_state").add(
                            time, grad.norm(dim=-1, p="fro").mean(0).item()), step))
            # Log gradient w.r.t action
            re_act.register_hook(
                partial(lambda time, grad:
                        getattr(self.logger, f"dict/time_sequence/grad_action").add(
                            time, grad.norm(dim=-1, p="fro").mean(0).item()), step))

            obs_list.append(re_obs)
            act_list.append(re_act)
            next_obs_list.append(re_next_obs)

            if step != (rollout_len - 1):
                reset_obs = th_obs[:, step + 1]
                reset_obs.requires_grad = True
                reset_obs_embed = self.model.immersion(reset_obs)
                re_obs_embed = (1 - done) * re_next_obs_embed + done * reset_obs_embed
                re_obs = (1 - done) * self.model.submersion(re_next_obs_embed) + done * reset_obs
                # re_obs = self.model.submersion(re_obs_embed)

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
                       is_vec_normalize: bool
                       ) -> VgRollout:
        rollout_data = self.rollout_to_torch(sample, is_vec_normalize=is_vec_normalize)
        mini_rollout_data = TorchLstmPPOAgent._make_mini_rollout(self,
                                                                 rollout_data, mini_rollout_size=mini_rollout_size)
        # perm_indices = torch.randperm(mini_rollout_data[0].shape[0])
        perm_indices = torch.arange(mini_rollout_data[0].shape[0])
        for indices in perm_indices.split(batch_size):
            yield VgRollout(*TorchLstmPPOAgent._take_slice(self, mini_rollout_data, indices=indices))

    def target_policy_update(self):
        self.target_policy.load_state_dict(self.policy.state_dict())

    def calculate_reward(self,
                         reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                         obs: torch.Tensor,
                         act: torch.Tensor,
                         next_obs: torch.Tensor,
                         obs_rms_var: torch.Tensor,
                         obs_rms_mean: torch.Tensor,
                         next_obs_rms_var: torch.Tensor,
                         next_obs_rms_mean: torch.Tensor,
                         reward_rms_var: torch.Tensor,
                         epsilon: float = 1e-8
                         ) -> torch.Tensor:

        orig_obs = obs * torch.sqrt(obs_rms_var + epsilon) + obs_rms_mean
        orig_next_obs = next_obs * torch.sqrt(next_obs_rms_var + epsilon) + next_obs_rms_mean
        # TODO!!!: Clipping actions remove gradient for clipped values!
        orig_act = torch.clamp(act, -1, 1)
        return reward_fn(orig_obs, orig_act, orig_next_obs) / torch.sqrt(reward_rms_var + epsilon)

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
                                 is_vec_normalize: bool,
                                 check_gae: bool = True,
                                 use_reparameterization: bool = True,
                                 policy_loss_beta: float = 1.0,
                                 ) -> Dict[str, float]:

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.target_policy_update()

        for epoch in range(epochs):
            for mini_rollout in self.rollout_loader(sample, batch_size=batch_size, mini_rollout_size=mini_rollout_size, is_vec_normalize=is_vec_normalize):

                if use_reparameterization:
                    re_obs, re_acts, re_next_obs, policy_params, values = self.reparameterize(
                        mini_rollout.obs, mini_rollout.act, mini_rollout.next_obs, mini_rollout.done)
                    rewards = self.calculate_reward(
                        reward_fn, re_obs, re_acts, re_next_obs,
                        mini_rollout.obs_rms_var,
                        mini_rollout.obs_rms_mean,
                        mini_rollout.next_obs_rms_var,
                        mini_rollout.next_obs_rms_mean,
                        mini_rollout.reward_rms_var)
                    obs = re_obs
                    next_obs = re_next_obs
                else:
                    policy_params, values = self.policy(mini_rollout.obs)
                    rewards = mini_rollout.reward
                    obs = mini_rollout.obs
                    next_obs = mini_rollout.next_obs

                _, old_values = self.target_policy(obs)  # Must contain the time dimension
                _, old_last_value = self.target_policy(next_obs[:, -1])

                if check_reward_consistency:
                    if not torch.allclose(obs, mini_rollout.obs, atol=1e-6):
                        mismatch = (obs - mini_rollout.obs).abs().max().item()
                        raise RuntimeError(
                            f"Mismatch between reparameterized and sampled observations: {mismatch}")
                    if not torch.allclose(next_obs, mini_rollout.next_obs, atol=1e-6):
                        mismatch = (next_obs - mini_rollout.next_obs).abs().max().item()
                        raise RuntimeError(
                            f"Mismatch between reparameterized and sampled next observations: {mismatch}")

                if check_reward_consistency:
                    if not torch.allclose(rewards, mini_rollout.reward, atol=1e-4):
                        mismatch = (rewards - mini_rollout.reward).abs().mean()
                        raise RuntimeError(
                            f"Mismatch between reward function and sampled rewards: {mismatch}")

                advantages, returns = calculate_diff_gae(
                    rewards, mini_rollout.done, old_values, old_last_value, gamma=gamma, gae_lambda=gae_lambda)

                if check_reward_consistency:
                    np_advantages, np_returns = calculate_gae(
                        *[tensor.cpu().detach().numpy()
                          for tensor in [mini_rollout.reward, mini_rollout.done,
                          old_values,
                          old_last_value]],
                        gamma=gamma, gae_lambda=gae_lambda
                    )
                    if not np.allclose(advantages.cpu().detach().numpy(), np_advantages, atol=1e-5):
                        raise RuntimeError(f"Mismatch advantages! Mismatch: {np.abs(advantages.cpu().detach().numpy() - np_advantages).max()}")
                    if not np.allclose(returns.cpu().detach().numpy(), np_returns, atol=1e-5):
                        raise RuntimeError("Mismatch returns!")

                flat_act, flat_old_log_prob, flat_advantages, flat_returns, flat_param, flat_value = self.flatten_time(
                    [mini_rollout.act, mini_rollout.old_log_p,
                        advantages, returns, policy_params, values]
                )

                policy_dist = self.policy.dist(flat_param)
                log_probs = policy_dist.log_prob(flat_act).unsqueeze(-1)
                entropies = policy_dist.entropy().unsqueeze(-1)
                ratio = torch.exp(log_probs - flat_old_log_prob.detach())

                value_loss = torch.nn.functional.mse_loss(flat_value, flat_returns.detach())
                entropy_loss = -entropies.mean()

                log_policy_loss = self.log_likelihood_loss(
                    ratio=ratio,
                    advantages=flat_advantages,
                    normalize_advantage=normalize_advantage,
                    clip_value=clip_value,
                )
                vg_loss = self.value_gradient_loss(
                    ratio=ratio,
                    advantages=flat_advantages,
                    normalize_advantage=normalize_advantage,
                    clip_value=clip_value,
                )

                if use_log_likelihood:
                    policy_loss = log_policy_loss
                else:
                    policy_loss = policy_loss_beta * log_policy_loss + \
                        (1 - policy_loss_beta) * vg_loss
                entropy_loss = -entropies.mean()
                loss = policy_loss + value_loss * value_coef + entropy_loss * ent_coef

                pre_update_model_params = {
                    name: param.detach().cpu().clone() for name, param in self.model.named_parameters()
                }
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                # Check if model parameters are updated
                post_update_model_params = {
                    name: param.detach().cpu() for name, param in self.model.named_parameters()
                }
                for key in pre_update_model_params.keys():
                    pre_param = pre_update_model_params[key]
                    post_param = post_update_model_params[key]
                    difference = (pre_param - post_param).abs().mean().item()
                    if difference > 0:
                        raise RuntimeError(key, (pre_param - post_param).abs().mean().item())

                getattr(self.logger, "scalar/agent/value_loss").push(value_loss.item())
                getattr(self.logger, "scalar/agent/policy_loss").push(policy_loss.item())
                getattr(self.logger, "scalar/agent/entropy_loss").push(entropy_loss.item())
                getattr(self.logger, "scalar/agent/approxkl").push(((ratio - 1) - ratio.log()).mean().item())
                getattr(self.logger, "scalar/agent/ratio").push(ratio.mean().item())
                getattr(self.logger, "scalar/agent/clip_percentage(%)").push(
                    (torch.clamp(ratio, 1 - clip_value, 1 + clip_value) == ratio).float().mean().item() * 100)
        getattr(self.logger, "scalar/agent/clip_range").push(clip_value)
        getattr(self.logger, "scalar/agent/learning_rate").push(lr)
        getattr(self.logger, "scalar/agent/policy_loss_beta").push(policy_loss_beta)

        getattr(self.logger, "dict/histogram/params").push(self.param_dict_as_numpy)
        getattr(self.logger, "dict/histogram/grads").push(self.grad_dict_as_numpy)

    def value_gradient_loss(self,
                            ratio: torch.Tensor,
                            advantages: torch.Tensor,
                            normalize_advantage: bool,
                            clip_value: float,
                            ) -> torch.Tensor:
        # maximize return that is formed using rewards and re-parameterized states
        ratio = ratio.detach()

        # if normalize_advantage:
        #     advantages = (advantages - advantages.mean()) / \
        #         (advantages.std() + 1e-8)

        surrogate_loss_1 = advantages * ratio
        surrogate_loss_2 = torch.clamp(advantages * ratio,
                                       (1 - clip_value) * advantages.detach(),
                                       (1 + clip_value) * advantages.detach())
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
            f"dict/time_sequence/{name}": SequenceNormDataLog()
            for name in ("grad_state", "forward_state", "grad_action", "grad_embed")
        }
        loggers["scalar/agent/policy_loss_beta"] = ListDataLog(
            reduce_fn=lambda values: np.max(values))
        loggers["scalar/agent/ratio"] = ListDataLog(reduce_fn=lambda values: np.mean(values))
        loggers["scalar/agent/ratio"] = ListDataLog(reduce_fn=lambda values: np.mean(values))
        loggers["scalar/agent/clip_percentage(%)"] = ListDataLog(
            reduce_fn=lambda values: np.mean(values))
        loggers["dict/histogram/params"] = ParamHistDataLog(n_bins=15)
        loggers["dict/histogram/grads"] = ParamHistDataLog(n_bins=15)

        self.logger.add_if_not_exists(loggers)
