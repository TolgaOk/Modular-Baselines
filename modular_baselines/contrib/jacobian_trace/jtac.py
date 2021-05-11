"""A2C based actor critic implementation with Jacobian trace """
import numpy as np
import torch
import time
import gym
from collections import namedtuple
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple

from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean, configure_logger
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.collectors.callbacks import BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.algorithms.a2c import A2C
from stable_baselines3.a2c.a2c import A2C as SB3_A2C

from modular_baselines.contrib.jacobian_trace.buffer import JTACBuffer
from modular_baselines.contrib.jacobian_trace.utils import FreezeParameters
from modular_baselines.contrib.jacobian_trace.type_aliases import (DiscreteLatentTuple,
                                                                   TransitionTuple,
                                                                   SequentialRolloutSamples)


class JTAC(A2C):
    """ JTAC algorithm.

    Args:
        policy (torch.nn.Module): Policy module
        rollout_buffer (RolloutBuffer): Rollout buffer
        collector (OnPolicyCollector): Experience Collector
        env (VecEnv): Vectorized environment
        rollout_len (int): Length of the rollout
        ent_coef (float): Entropy coefficient/multiplier
        vf_coef (float): Value loss coefficient/multiplier
        max_grad_norm (float): Maximum allowed gradient norm
            advantage or not. Defaults to False.
        callbacks (List[BaseAlgorithmCallback], optional): Algorithm callbacks.
            Defaults to [].
        device (str, optional): Torch device. Defaults to "cpu".

    Raises:
        ValueError: Policy class must have an "optimizer"
            members
    """

    def __init__(self,
                 policy: torch.nn.Module,
                 rollout_buffer: JTACBuffer,
                 collector: OnPolicyCollector,
                 env: VecEnv,
                 rollout_len: int,
                 ent_coef: float,
                 vf_coef: float,
                 gamma: float,
                 gae_lambda: float,
                 model_loss_coef: float,
                 prior_kl_coef: float,
                 trans_kl_coef: float,
                 model_iteration_per_update: int,
                 model_batch_size: int,
                 model_horizon: int,
                 policy_iteration_per_update: int,
                 policy_batch_size: int,
                 policy_nstep: int,
                 max_grad_norm: float,
                 reward_clip: Optional[float] = None,
                 enable_jtac: bool = False,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):

        self.model_loss_coef = model_loss_coef
        self.prior_kl_coef = prior_kl_coef
        self.trans_kl_coef = trans_kl_coef
        self.model_iteration_per_update = model_iteration_per_update
        self.model_batch_size = model_batch_size
        self.model_horizon = model_horizon
        self.policy_iteration_per_update = policy_iteration_per_update
        self.policy_batch_size = policy_batch_size
        self.policy_nstep = policy_nstep
        self.reward_clip = reward_clip
        self.enable_jtac = enable_jtac
        A2C.__init__(self,
                     policy=policy,
                     rollout_buffer=rollout_buffer,
                     collector=collector,
                     env=env,
                     rollout_len=rollout_len,
                     ent_coef=ent_coef,
                     vf_coef=vf_coef,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     max_grad_norm=max_grad_norm,
                     callbacks=callbacks,
                     device=device)

    def train(self) -> None:

        for iteration in range(self.model_iteration_per_update):
            self.policy.model_optimizer.zero_grad()
            sample = self.buffer.get_sequential_rollout(self.model_horizon, self.model_batch_size)
            model_loss = self.discrete_model_loss(sample)
            transition_loss, recon_loss, q_latent_loss, e_latent_loss, perplexity = model_loss
            (transition_loss * self.trans_kl_coef
             + recon_loss
             + q_latent_loss
             + e_latent_loss).backward()

            torch.nn.utils.clip_grad_norm_(
                chain(
                    self.policy.encoder.parameters(),
                    self.policy.decoder.parameters(),
                    self.policy.transition_dist.parameters(),
                ),
                self.max_grad_norm
            )

            self.policy.model_optimizer.step()

        for iteration in range(self.policy_iteration_per_update):
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            sample = self.buffer.get_sequential_rollout(self.model_horizon,
                                                        self.model_batch_size,
                                                        maximum_horizon=self.rollout_len)
            policy_loss, entropy_loss, value_loss = self.jtac_loss(sample)
            loss = policy_loss + entropy_loss * self.ent_coef + value_loss * self.vf_coef
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(
                    self.policy.critic.parameters(),
                    self.policy.actor.parameters()),
                self.max_grad_norm)

            self.policy.actor_optimizer.step()
            self.policy.critic_optimizer.step()

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._n_updates += 1

    def discrete_model_loss(self, sample: SequentialRolloutSamples):
        squeezed_obs, meta_data = SampleHandler.squeeze(sample)
        squeezed_latent, q_latent_loss, e_latent_loss, perplexity = self.policy.get_latent(
            squeezed_obs)
        squeezed_pred_obs_dist = self.policy.decoder(squeezed_latent)
        recon_loss = -squeezed_pred_obs_dist.log_prob(squeezed_obs).reshape(
            squeezed_obs.shape[0], -1).sum(1).mean(0)

        latent_tuple, next_latent_tuple = [DiscreteLatentTuple(*half_tensor) for half_tensor in zip(
            *[SampleHandler.unsqueeze(tensor, meta_data) for tensor in squeezed_latent])]

        transition_losses = []
        transition_dist_logits = torch.zeros_like(latent_tuple.encoding[:, 0])
        for index in range(meta_data.horizon):
            step_tuple, next_step_tuple = (DiscreteLatentTuple(*(tensor[:, index] for tensor in _tuple))
                                           for _tuple in (latent_tuple, next_latent_tuple))

            transition_dist = self.policy.transition_dist(step_tuple,
                                                          sample.actions[:, index].flatten(),
                                                          transition_dist_logits)
            transition_losses.append(
                -transition_dist.log_prob(next_step_tuple.encoding).reshape(
                    sample.actions.shape[0], -1).sum(1).mean(0))

        transition_loss = sum(transition_losses) / meta_data.horizon

        logger.record_mean("train/model/transition_loss", transition_loss.item())
        logger.record_mean("train/model/e_latent_loss", e_latent_loss.item())
        logger.record_mean("train/model/q_latent_loss", q_latent_loss.item())
        logger.record_mean("train/model/perplexity", perplexity.item())
        logger.record_mean("train/model/reconstruction", recon_loss.item())

        return transition_loss, recon_loss, q_latent_loss, e_latent_loss, perplexity

    def jtac_loss(self,
                  sample: SequentialRolloutSamples,
                  r_encoding: Optional[torch.Tensor] = None,
                  prev_terminations: Optional[torch.Tensor] = None):
        local_log = namedtuple("Log", "policy entropy next_state_logp action")([], [], [], [])

        squeezed_obs, meta_data = SampleHandler.squeeze(sample)
        squeezed_latent, *_ = self.policy.get_latent(squeezed_obs)

        latent_tuple, next_latent_tuple = [DiscreteLatentTuple(*half_tensor) for half_tensor in zip(
            *[SampleHandler.unsqueeze(tensor, meta_data) for tensor in squeezed_latent])]

        values, next_values = SampleHandler.unsqueeze(
            self.policy.critic(squeezed_latent), meta_data)
        advantages, returns, td_errors = self.calculate_gae(values.squeeze(2).detach(),
                                                            next_values.squeeze(2).detach(),
                                                            sample.terminations,
                                                            sample.rewards,
                                                            self.reward_clip)

        # Value loss using the TD(gae_lambda) target
        value_loss = torch.nn.functional.smooth_l1_loss(values.flatten(), returns.flatten())

        if self.enable_jtac:
            # Initial transition dist logits
            transition_dist_logits = torch.zeros_like(latent_tuple.encoding[:, 0])

            for index in range(meta_data.horizon):
                step_tuple, next_step_tuple = (DiscreteLatentTuple(*(tensor[:, index] for tensor in _tuple))
                                               for _tuple in (latent_tuple, next_latent_tuple))

                # Check termination condition for encodings
                if r_encoding is not None:
                    r_encoding = (torch.einsum("b...,b->b...",
                                               r_encoding,
                                               (1 - prev_terminations))
                                  + torch.einsum("b...,b->b...",
                                                 step_tuple.encoding,
                                                 prev_terminations))
                    step_tuple = DiscreteLatentTuple(
                        step_tuple.logit, step_tuple.quantized, r_encoding)

                # Policy reparametrization
                act_dist = self.policy.actor(step_tuple, onehot=True)
                onehot_action = self.policy.actor.make_onehot(
                    sample.actions[:, index].flatten()).float()
                # Measure action gradients
                act_dist.probs.retain_grad()
                raction = self.policy.actor.reparam(act_dist, onehot_action)

                # Transition reparametrization
                with FreezeParameters([self.policy.transition_dist]):
                    transition_dist = self.policy.transition_dist(
                        step_tuple, raction, transition_dist_logits)
                r_encoding = self.policy.transition_dist.reparam(
                    transition_dist, next_step_tuple.encoding.detach())
                prev_terminations = sample.terminations[:, index]
                transition_dist_logits = transition_dist.logits

                # Policy loss
                next_state_log_prob = transition_dist.log_prob(
                    next_latent_tuple.encoding[:, index].detach()).reshape(td_errors.shape[0], -1).sum(1)
                policy_loss = -(next_state_log_prob * td_errors[:, index].detach()).mean(0)

                # Actor entropy loss
                act_entropy = act_dist.entropy()
                entropy_loss = -torch.mean(act_entropy)

                local_log.entropy.append(entropy_loss)
                local_log.action.append(act_dist.probs)
                local_log.policy.append(policy_loss)
                local_log.next_state_logp.append(next_state_log_prob.mean(0).item())

            policy_loss = sum(local_log.policy) / meta_data.horizon
            entropy_loss = sum(local_log.entropy) / meta_data.horizon

            logger.record_mean("train/next_state_logp", next_state_logp.item())
        else:
            latent_tuple = DiscreteLatentTuple(
                *(tensor.reshape(np.product(tensor.shape[:2]), *tensor.shape[2:]) for tensor in latent_tuple))
            actor_dist = self.policy.actor(latent_tuple)
            log_prob = actor_dist.log_prob(sample.actions.flatten())

            a2c_loss = - log_prob * advantages.flatten().detach()
            policy_loss = a2c_loss.mean()
            entropy_loss = -actor_dist.entropy().mean()

        logger.record_mean("train/entropy_loss", entropy_loss.item())
        logger.record_mean("train/policy_loss", policy_loss.item())
        logger.record_mean("train/value_loss", value_loss.item())

        return policy_loss, entropy_loss, value_loss

    def calculate_gae(self,
                      values: torch.Tensor,
                      next_values: torch.Tensor,
                      terminations: torch.Tensor,
                      rewards: torch.Tensor,
                      reward_clip: Optional[float] = None):
        assert len(rewards.shape) == 2
        assert len(terminations.shape) == 2
        assert len(values.shape) == 2
        assert len(next_values.shape) == 2

        if reward_clip:
            rewards = torch.clamp(rewards, min=-reward_clip, max=reward_clip)
        td_targets = next_values * (1 - terminations) * self.gamma + rewards
        td_errors = td_targets - values

        returns = torch.zeros_like(td_errors)
        advantages = torch.zeros_like(td_errors)

        advantage = torch.zeros_like(td_errors[:, 0])
        for index in reversed(range(td_targets.shape[1])):
            advantage = td_errors[:, index] + advantage * \
                self.gae_lambda * self.gamma * (1 - terminations[:, index])
            advantages[:, index] = advantage
            returns[:, index] = advantage + values[:, index]

        return advantages, returns, td_errors


class SquuezedSampleMeta(NamedTuple):
    batch_size: int
    horizon: int
    termination_indexes: Tuple[torch.Tensor]


class SampleHandler():

    @staticmethod
    def squeeze(sample: SequentialRolloutSamples):
        assert len(sample.terminations.shape) == 2
        batch_size, horizon, *shape = sample.observations.shape

        termination_indexes = sample.terminations[:-1].nonzero(as_tuple=True)
        squeezed_observation = torch.cat([
            sample.observations.reshape(batch_size * horizon, *shape),
            sample.next_observations[:, -1].reshape(batch_size, *shape),
            sample.next_observations[termination_indexes]
        ])
        return squeezed_observation, SquuezedSampleMeta(
            batch_size, horizon, termination_indexes)

    @staticmethod
    def unsqueeze(squeezed: torch.Tensor, meta_data: SquuezedSampleMeta):
        batch_size, horizon, termination_indexes = meta_data
        shape = squeezed.shape[1:]
        upper_index = batch_size * horizon
        observation = squeezed[:upper_index]
        observation = observation.reshape(batch_size, horizon, *shape)

        next_upper_index = upper_index + batch_size
        next_observation = torch.cat([
            observation[:, 1:],
            squeezed[upper_index: next_upper_index].reshape(batch_size, 1, *shape)
        ], dim=1)
        next_observation[termination_indexes] = squeezed[next_upper_index:]

        return observation, next_observation
