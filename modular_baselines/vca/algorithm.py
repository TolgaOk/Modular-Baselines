import torch
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv

from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.vca.modules import (BaseTransitionModule,
                                           CategoricalPolicyModule)
from modular_baselines.vca.buffer import Buffer
from modular_baselines.vca.collector import NStepCollector


class VCA(OnPolicyAlgorithm):

    def __init__(self,
                 policy_module: CategoricalPolicyModule,
                 transition_module: BaseTransitionModule,
                 buffer: Buffer,
                 collector: NStepCollector,
                 rollout_len: int,
                 env: VecEnv,
                 reward_vals: np.ndarray,
                 trans_opt: torch.optim.Optimizer,
                 policy_opt: torch.optim.Optimizer,
                 batch_size: int = 32,
                 entropy_coef: float = 0.01,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu",
                 verbose_playback: bool = False,
                 grad_norm: bool = False,
                 grad_clip: bool = False,
                 pg_optimization: bool = False,
                 gamma=1,
                 alpha=0.001):
        super().__init__(
            policy_module,
            buffer,
            collector,
            env,
            rollout_len,
            device=device,
            callbacks=callbacks,
        )

        self.policy_module = policy_module.to(self.device)
        self.transition_module = transition_module.to(self.device)
        self.trans_opt = trans_opt
        self.policy_opt = policy_opt
        self.batch_size = batch_size
        self.ent_coef = entropy_coef
        self.gamma = gamma
        self.alpha = alpha

        self.reward_vals = reward_vals
        self.policy = torch.nn.ModuleList([self.policy_module,
                                           self.transition_module])

        self.verbose_playback = verbose_playback
        self.grad_norm = grad_norm
        self.grad_clip = grad_clip
        self.pg_optimization = pg_optimization

    def train(self):
        self.train_policy()
        self.train_trans()

    def train_policy(self):
        episode = self.buffer.sample_last_episode()
        if episode is None:
            return

        assert torch.all(episode.dones.reshape(-1)[:-1] == 0)
        assert (episode.dones.reshape(-1)[-1] == 1)
        assert episode.rewards.reshape(-1)[-1] != 0

        entropies = []
        expected_rewards = []
        log_probs = []
        r_acts = []
        r_obs = []
        gate_mean = []
        gate_std = []

        r_state = self._init_soft_state(episode.observations[0].unsqueeze(0)).to(self.device)
        r_state.requires_grad = True

        for ix in range(len(episode.dones)):

            action = self._action_onehot(episode.actions[ix].unsqueeze(0).to(self.device))
            next_state = self._process_state(
                episode.next_observations[ix].unsqueeze(0).to(self.device))

            log_probs.append(self.policy_module.dist(
                r_state.detach()).log_prob(episode.actions[ix].to(self.device)))

            r_action, entropy = self.policy_module.reparam_act(r_state, action)

            if self.grad_norm:
                r_state = GradNormalizer.apply(r_state)
                r_action = GradNormalizer.apply(r_action)

            if self.grad_clip:
                r_state = GradClip.apply(r_state)
                r_action = GradClip.apply(r_action)

            r_action.retain_grad()
            r_acts.append(r_action)

            if "gate_output" in dir(self.transition_module):
                gate_out = self.transition_module.gate_output(
                    r_state, r_action).abs()
                gate_mean.append(gate_out.mean().item())
                gate_std.append(gate_out.std().item())

            dist = self.transition_module.dist(r_state, r_action)
            r_next_state = self.transition_module.reparam(
                obs=next_state,
                probs=dist)

            r_next_state.retain_grad()
            r_obs.append(r_next_state)

            expected_reward = self._expected_reward(
                self.reward_vals, r_next_state)
            expected_rewards.append(expected_reward)
            entropies.append(entropy)

            r_state = r_next_state

        reward_sum = sum(expected_rewards).sum()
        # reward_sum = expected_rewards[-1].sum()
        entropy_sum = sum(entropies).sum()
        self.policy_opt.zero_grad()

        eps_reward = episode.rewards
        if self.pg_optimization:
            log_prob_term = sum([logp * self.gamma**(ix)
                                 for ix, logp in enumerate(reversed(log_probs))])
            (-log_prob_term * eps_reward.sum().item() -
             entropy_sum * self.ent_coef).backward()
        else:
            (-sum(log_probs) * reward_sum.item() * self.alpha -
             reward_sum - entropy_sum * self.ent_coef).backward()

        for param in self.policy_module.parameters():
            if torch.any(torch.isnan(param.grad)):
                # print(param.grad)
                # print(param)
                for ix, r_act in enumerate(r_acts):
                    print(ix, r_act.grad)
                print(param.grad)
                raise RuntimeError("Nan observed!")

        if np.random.uniform() < 0.01 and self.verbose_playback:
            for ix, (r_act, r_ob) in enumerate(zip(r_acts, r_obs)):
                print(ix,
                      r_act.grad.abs().mean().item(),
                      r_ob.grad.abs().mean().item())

        self.policy_opt.step()

        if self.pg_optimization is False:
            logger.record_mean("playback/state_grad",
                               np.mean([r_ob.grad.abs().mean().item() for r_ob in r_obs]))
            logger.record_mean("playback/action_grad",
                               np.mean([r_act.grad.abs().mean().item() for r_act in r_acts]))
        if "gate_output" in dir(self.transition_module):
            logger.record_mean("playback/gate_mean", np.mean(gate_mean))
            logger.record_mean("playback/gate_std", np.std(gate_std))

        logger.record_mean("train/log_probs", sum(log_probs).item())
        logger.record_mean("train/E[R]", reward_sum.item())
        logger.record_mean(
            "train/entropy", (entropy_sum.item() / len(entropies)))

    def train_trans(self):

        sample = self.buffer.sample(self.batch_size)
        if sample is None:
            return

        state = self._process_state(sample.observations.to(self.device))
        next_state = self._process_state(sample.next_observations.to(self.device))
        action = self._action_onehot(sample.actions.to(self.device))

        trans_loss = self.transition_loss(state, action, next_state)
        self.trans_opt.zero_grad()
        trans_loss.backward()
        self.trans_opt.step()

        logger.record_mean("train/Transition MSE",
                           torch.nn.functional.mse_loss(
                               self.transition_module(state, action),
                               next_state
                           ).item())
        logger.record_mean("train/Transition loss", trans_loss.item())

    @abstractmethod
    def transition_loss(self, *args):
        pass

    @abstractmethod
    def _init_soft_state(self, *args):
        pass

    @abstractmethod
    def _process_state(self, *args):
        pass

    @abstractmethod
    def _expected_reward(self, *args):
        pass

    def _action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        assert len(action.shape) == 2, ""
        action_set = torch.arange(self.env.action_space.n).to(self.device)
        return (action_set.reshape(1, -1) == action).float()


class DiscreteStateVCA(VCA):

    def _init_soft_state(self, observations: torch.Tensor) -> torch.Tensor:
        soft_state = torch.ones(
            (1, self.transition_module.insize)).to(self.device)
        state = self._process_state(observations)
        r_state = self.transition_module.reparam(state, soft_state)
        return r_state

    def _process_state(self, state: torch.Tensor) -> torch.Tensor:
        assert len(state.shape) == 2, ""
        return (self.transition_module.state_set.reshape(1, -1) == state).float()

    def _expected_reward(self,
                         reward_arr: np.ndarray,
                         next_state_prob: torch.Tensor) -> torch.Tensor:
        assert len(next_state_prob.shape) == 2, ""
        reward_tens = torch.from_numpy(reward_arr).to(self.device)
        return (reward_tens * next_state_prob).sum(1).mean(0)

    def transition_loss(self,
                        state: torch.Tensor,
                        action: torch.Tensor,
                        next_state: torch.Tensor) -> torch.Tensor:
        target_next_state = next_state.argmax(dim=1)
        next_state_pred = self.transition_module(state, action)
        trans_loss = torch.nn.functional.cross_entropy(
            next_state_pred, target_next_state)
        return trans_loss


class ContinuousStateVCA(VCA):

    def _init_soft_state(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def _process_state(self, state: torch.Tensor):
        return state

    # def _expected_reward(self,
    #                      reward_info: Dict,
    #                      r_next_state: torch.Tensor):

    #     index = reward_info["index"]
    #     upper_threshold = reward_info["upper_threshold"]
    #     lower_threshold = reward_info["lower_threshold"]

    #     value = 0.0
    #     value = -1.0 if r_next_state[:, index] > upper_threshold else value
    #     value = 1.0 if r_next_state[:, index] < lower_threshold else value

    #     multiplier = torch.zeros_like(r_next_state)
    #     multiplier[:, index] = value

    #     return (r_next_state * multiplier).sum(1).mean(0)

    # def _expected_reward(self,
    #                      reward_info: Dict,
    #                      r_next_state: torch.Tensor):

    #     loss = -((r_next_state[:, 0] - r_next_state[:, 3])**2).mean(0)
    #     return loss

    def _expected_reward(self,
                         reward_info: Dict,
                         r_next_state: torch.Tensor):

        loss = -((r_next_state[:, 2])**2).mean()
        # loss = r_next_state[: 0]
        return loss

    def transition_loss(self,
                        state: torch.Tensor,
                        action: torch.Tensor,
                        target_state: torch.Tensor):
        dist = self.transition_module.dist(state, action)
        return -dist.log_prob(target_state).mean()


class ChannelStateVCA(ContinuousStateVCA):

    def _expected_reward(self,
                         reward_info: Dict,
                         r_next_state: torch.Tensor) -> torch.Tensor:
        index = reward_info["target_index"]
        target = torch.from_numpy(reward_info["target"]).unsqueeze(0).to(self.device)
        return (r_next_state[:, index] * target).mean(0).sum()

    def transition_loss(self,
                        state: torch.Tensor,
                        action: torch.Tensor,
                        next_state: torch.Tensor) -> torch.Tensor:
        next_state_pred = self.transition_module(state, action)
        target_state = torch.abs(state - next_state).detach()
        trans_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            next_state_pred, target_state)
        return trans_loss


class GradNormalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        norm = torch.norm(grad_output, dim=1, keepdim=True)
        return grad_output / norm


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        norm = torch.norm(grad_output, dim=1, keepdim=True)
        norm = torch.maximum(norm, torch.ones_like(norm))
        return grad_output / norm
