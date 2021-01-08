import torch
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv

from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.vca.modules import (BaseTransitionModule,
                                           CategoricalRewardModule,
                                           CategoricalPolicyModule)
from modular_baselines.vca.buffer import Buffer
from modular_baselines.vca.collector import NStepCollector


class VCA(OnPolicyAlgorithm):

    def __init__(self,
                 policy_module: CategoricalRewardModule,
                 transition_module: BaseTransitionModule,
                 reward_module: CategoricalRewardModule,
                 buffer: Buffer,
                 collector: NStepCollector,
                 rollout_len: int,
                 env: VecEnv,
                 reward_vals: np.ndarray,
                 trans_opt: torch.optim.Optimizer,
                 policy_opt: torch.optim.Optimizer,
                 reward_opt: torch.optim.Optimizer,
                 use_reward_module: bool = False,
                 batch_size: int = 32,
                 entropy_coef: float = 0.01,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu",
                 verbose_playback: bool = False,
                 grad_norm: bool = False):
        super().__init__(
            policy_module,
            buffer,
            collector,
            env,
            rollout_len,
            device=device,
            callbacks=callbacks,
        )

        self.policy_module = policy_module
        self.transition_module = transition_module
        self.reward_module = reward_module
        self.trans_opt = trans_opt
        self.policy_opt = policy_opt
        self.reward_opt = reward_opt
        self.batch_size = batch_size
        self.ent_coef = entropy_coef
        self.use_reward_module = use_reward_module

        self.reward_vals = reward_vals
        self.policy = torch.nn.ModuleList([self.policy_module,
                                           self.transition_module,
                                           self.reward_module])

        self.verbose_playback = verbose_playback
        self.grad_norm = grad_norm

    def train(self):
        self.train_policy()
        self.train_trans_and_reward()

    def train_policy(self):
        episode = self.buffer.sample_last_episode()
        if episode is None:
            return

        assert torch.all(episode.dones.reshape(-1)[:-1] == 0)
        assert (episode.dones.reshape(-1)[-1] == 1)
        assert episode.rewards.reshape(-1)[-1] != 0

        entropies = []
        expected_rewards = []
        r_acts = []
        r_obs = []

        r_state = self._init_soft_state(episode.observations[0].unsqueeze(0))
        r_state.requires_grad = True

        for ix in range(len(episode.dones)):
            action = self._action_onehot(episode.actions[ix].unsqueeze(0))
            next_state = self._process_state(
                episode.next_observations[ix].unsqueeze(0))

            r_action, entropy = self.policy_module.reparam_act(r_state, action)

            if self.grad_norm:
                r_state = GradNormalizer.apply(r_state)
                r_action = GradNormalizer.apply(r_action)

            r_action.retain_grad()
            r_acts.append(r_action)
            r_state.retain_grad()
            r_obs.append(r_state)

            logits = self.transition_module(r_state, r_action)


            r_next_state = self.transition_module.reparam(
                next_state, logits)

            if self.use_reward_module:
                expected_reward = self.reward_module.expected(r_next_state)
            else:
                expected_reward = self._expected_reward(
                    self.reward_vals, r_next_state)
            expected_rewards.append(expected_reward)
            entropies.append(entropy)

            r_state = r_next_state

        # reward_sum = sum(expected_rewards).sum()
        reward_sum = expected_rewards[-1].sum()
        entropy_sum = sum(entropies).sum()
        self.policy_opt.zero_grad()
        (-reward_sum - entropy_sum * self.ent_coef).backward()
        
        for param in self.policy_module.parameters():
            if torch.any(torch.isnan(param.grad)):
                # print(param.grad)
                # print(param)
                for ix, r_act in enumerate(r_acts):
                    print(ix, r_act.grad)
                raise RuntimeError("Nan observed!")

        if np.random.uniform() < 0.01 and self.verbose_playback:
            for ix, (r_act, r_ob) in enumerate(zip(r_acts, r_obs)):
                print(ix,
                      r_act.grad.abs().mean().item(),
                      r_ob.grad.abs().mean().item())

        self.policy_opt.step()
        logger.record_mean("train/E[R]", reward_sum.item())
        logger.record_mean("train/entropy", entropy_sum.item())

    def train_trans_and_reward(self):

        sample = self.buffer.sample(self.batch_size)
        if sample is None:
            return

        state = self._process_state(sample.observations)
        next_state = self._process_state(sample.next_observations)
        action = self._action_onehot(sample.actions)

        trans_loss = self.transition_loss(state, action, next_state)
        self.trans_opt.zero_grad()
        trans_loss.backward()
        self.trans_opt.step()

        target_reward = self._reward_target(sample.rewards)
        reward_pred = self.reward_module(next_state).logits
        reward_loss = torch.nn.functional.cross_entropy(
            reward_pred, target_reward)
        self.reward_opt.zero_grad()
        reward_loss.backward()
        self.reward_opt.step()

        logger.record_mean("train/Transition loss", trans_loss.item())
        logger.record_mean("train/Reward loss", reward_loss.item())

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
        action_set = torch.arange(self.env.action_space.n)
        return (action_set.reshape(1, -1) == action).float()

    def _reward_target(self, reward: torch.Tensor) -> torch.Tensor:
        assert len(reward.shape) == 2, ""
        onehot = (self.reward_module.reward_set.reshape(1, -1) == reward)
        return onehot.float().argmax(dim=1)


class DiscerteStateVCA(VCA):

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
        reward_tens = torch.from_numpy(reward_arr)
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


class GradNormalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        norm = torch.norm(grad_output, dim=1, keepdim=True)
        return grad_output / norm
