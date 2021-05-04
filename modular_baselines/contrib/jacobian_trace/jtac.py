"""A2C based actor critic implementation with Jacobian trace """
import numpy as np
import torch
import time
import gym
from collections import namedtuple
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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
from modular_baselines.contrib.jacobian_trace.type_aliases import (LatentTuple,
                                                                   TransitionTuple)


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
                 model_batch_size: int,
                 max_grad_norm: float,
                 enable_jtac: bool = False,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):

        self.model_loss_coef = model_loss_coef
        self.prior_kl_coef = prior_kl_coef
        self.trans_kl_coef = trans_kl_coef
        self.model_batch_size = model_batch_size
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
        self.policy.ac_optimizer.zero_grad()
        self.policy.model_optimizer.zero_grad()

        model_loss = self.model_loss()
        actor_critic_loss = self.jtac_loss() if self.enable_jtac else self.a2c_loss()
        loss = model_loss * self.model_loss_coef + actor_critic_loss
        loss.backward()

        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(
            chain(self.policy.critic.parameters(),
                  self.policy.actor.parameters()),
            self.max_grad_norm)

        self.policy.model_optimizer.step()
        self.policy.ac_optimizer.step()

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._n_updates += 1

    def model_loss(self):

        sample = self.buffer.sample(self.model_batch_size)

        actions = sample.actions
        combined_obs = torch.cat([sample.observations, sample.next_observations], dim=0)

        combined_latent_tuple = self.policy.get_latent(combined_obs)
        pred_next_obs_dist = self.policy.decoder(combined_latent_tuple)

        recon_loss = -pred_next_obs_dist.log_prob(combined_obs).sum(1).mean(0)
        prior_state_dist = self.policy.make_normal_prior(combined_latent_tuple.dist)
        prior_kl_loss = torch.distributions.kl_divergence(
            combined_latent_tuple.dist, prior_state_dist).sum(1).mean(0)

        prev_state_dist, next_state_dist = self.policy.batch_chunk_distribution(
            combined_latent_tuple.dist)
        prev_embedding, _ = torch.chunk(combined_latent_tuple.embedding, 2, dim=0)
        prev_rstate, _ = torch.chunk(combined_latent_tuple.rsample, 2, dim=0)
        pred_next_state_dist = self.policy.transition_dist(
            LatentTuple(prev_embedding, prev_state_dist, prev_rstate), actions)
        transition_kl_loss = torch.distributions.kl_divergence(
            pred_next_state_dist,
            self.policy.transition_dist.detach_dist(next_state_dist)).sum(1).mean(0)

        model_loss = transition_kl_loss * self.trans_kl_coef + \
            prior_kl_loss * self.prior_kl_coef + recon_loss

        logger.record_mean("train/loss/model/transition_kl", transition_kl_loss.item())
        logger.record_mean("train/loss/model/prior_kl", prior_kl_loss.item())
        logger.record_mean("train/loss/model/reconstruction", recon_loss.item())

        return model_loss

    def a2c_loss(self):
        self.buffer.compute_returns_and_advantage(
            initial_pos_indices=(np.ones(self.buffer.n_envs, dtype=np.int32) *
                                 (self.buffer.pos - 1 - self.rollout_len)),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            rollout_size=self.rollout_len)

        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        logger.record("train/a2c/entropy_loss", entropy_loss.item())
        logger.record("train/a2c/policy_loss", policy_loss.item())
        logger.record("train/a2c/value_loss", value_loss.item())

        return loss

    def jtac_loss(self):
        rstate = None
        loss_list = namedtuple("Loss", "total policy value entropy")([], [], [], [])
        for rollout_data in self.rollout_buffer.get_sequential_rollout(self.rollout_len):

            actions = rollout_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            with torch.no_grad():
                latent_tuple = self.policy.get_latent(rollout_data.observations)
                next_latent_tuple = self.policy.get_latent(rollout_data.next_observations)
            if rstate is not None:
                rstate = latent_tuple.rsample * prev_terminations + (1 - prev_terminations) * rstate
            else:
                rstate = latent_tuple.rsample
            latent_tuple = LatentTuple(latent_tuple.embedding, latent_tuple.dist, rstate)

            values = self.policy.critic(latent_tuple)
            next_values = self.policy.critic(next_latent_tuple)
            td_target = (next_values * (1 - rollout_data.terminations.unsqueeze(1)) * self.gamma
                         + rollout_data.rewards.unsqueeze(1)).detach()
            td_error = td_target - values

            act_dist = self.policy.actor(latent_tuple, onehot=True)
            raction = self.policy.actor.reparam(act_dist, actions)

            with FreezeParameters([self.policy.transition_dist]):
                transition_dist = self.policy.transition_dist(latent_tuple, raction)

            # Do reparam
            rstate = self.policy.transition_dist.reparam(
                transition_dist, next_latent_tuple.rsample.detach())
            prev_terminations = rollout_data.terminations.unsqueeze(1)

            # Policy gradient loss
            next_state_log_prob = transition_dist.log_prob(next_latent_tuple.rsample.detach())
            policy_loss = -(next_state_log_prob * td_error.detach()).sum(1).mean(0)

            # Value loss using the TD(gae_lambda) target
            values = values
            value_loss = torch.nn.functional.l1_loss(values, td_target)

            # Actor entropy loss
            act_entropy = act_dist.entropy()
            entropy_loss = -torch.mean(act_entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss_list.total.append(loss)
            loss_list.entropy.append(entropy_loss)
            loss_list.value.append(value_loss)
            loss_list.policy.append(policy_loss)

        logger.record_mean("train/loss/ac/entropy_loss",
                           sum(loss_list.entropy).item() / self.rollout_len)
        logger.record_mean("train/loss/ac/policy", sum(loss_list.policy).item() / self.rollout_len)
        logger.record_mean("train/loss/ac/value", sum(loss_list.value).item() / self.rollout_len)

        return sum(loss_list.total) / self.rollout_len
