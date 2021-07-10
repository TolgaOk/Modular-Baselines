"""A2C based actor critic implementation with Jacobian trace """
import numpy as np
import torch
import time
import gym
from collections import namedtuple
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple

from stable_baselines3.common import logger
from modular_baselines.contrib.jacobian_trace.utils import FreezeParameters
from modular_baselines.contrib.jacobian_trace.type_aliases import (DiscreteLatentTuple,
                                                                   SequentialRolloutSamples)
from modular_baselines.contrib.jacobian_trace.jtac import (JTAC,
                                                           SampleHandler)


class EncodedJTAC(JTAC):
    """ Encoded JTAC algorithm.
    """

    def train(self) -> None:
        for _ in range(self.policy_iteration_per_update):
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            sample = self.buffer.get_sequential_rollout(self.policy_nstep,
                                                        self.policy_batch_size,
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

        horizon = sample.rewards.shape[1]
        latent_tuple = DiscreteLatentTuple(None, None, sample.observations)
        next_latent_tuple = DiscreteLatentTuple(None, None, sample.next_observations)

        transition_losses = []
        # transition_dist_logits = torch.zeros_like(latent_tuple.encoding[:, 0])
        # transition_hidden = self.policy.transition_dist.model.init_hidden(
        # sample.observations.shape[0])
        for index in range(horizon):
            step_tuple = DiscreteLatentTuple(None, None, latent_tuple.encoding[:, index])
            next_step_tuple = DiscreteLatentTuple(None, None, next_latent_tuple.encoding[:, index])

            transition_dist = self.policy.transition_dist(
                step_tuple,
                sample.actions[:, index].unbind(-1),
                # transition_hidden
            )
            # transition_dist_logits = transition_dist.logits
            transition_losses.append(
                -transition_dist.log_prob(next_step_tuple.encoding).reshape(
                    sample.actions.shape[0], -1).sum(1).mean(0))

        transition_loss = sum(transition_losses) / horizon

        logger.record_mean("train/model/transition_loss", transition_loss.item())

        return transition_loss

    def jtac_loss(self,
                  sample: SequentialRolloutSamples,
                  r_encoding: Optional[torch.Tensor] = None,
                  prev_terminations: Optional[torch.Tensor] = None):
        local_log = namedtuple("Log", "policy entropy next_state_logp action")([], [], [], [])

        squeezed_obs, meta_data = SampleHandler.squeeze(sample)
        squeezed_latent = DiscreteLatentTuple(None, None, squeezed_obs)

        latent_tuple, next_latent_tuple = [DiscreteLatentTuple(None, None, tensor) for tensor in
                                           SampleHandler.unsqueeze(squeezed_obs, meta_data)]

        values, next_values = SampleHandler.unsqueeze(
            self.policy.critic(squeezed_latent), meta_data)
        advantages, returns, td_errors = self.calculate_gae(values.squeeze(2).detach(),
                                                            next_values.squeeze(2).detach(),
                                                            sample.terminations,
                                                            sample.rewards,
                                                            self.reward_clip)
        # Value loss using the TD(gae_lambda) target
        value_loss = torch.nn.functional.mse_loss(values.flatten(), returns.flatten())

        if self.enable_jtac:
            # Initial transition dist logits
            # transition_dist_logits = torch.zeros_like(latent_tuple.encoding[:, 0])
            # transition_hidden = self.policy.transition_dist.model.init_hidden(
            # sample.observations.shape[0])
            if not hasattr(self, "moving_avg_td"):
                self.moving_avg_td = 1.0
            self.moving_avg_td += 0.05 * (td_errors.abs().mean().item() - self.moving_avg_td)

            for index in range(meta_data.horizon):
                step_tuple = DiscreteLatentTuple(None, None, latent_tuple.encoding[:, index])
                next_step_tuple = DiscreteLatentTuple(
                    None, None, next_latent_tuple.encoding[:, index])

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
                act_dists = self.policy.actor(step_tuple, onehot=True)
                onehot_actions = self.policy.actor.make_onehot(
                    torch.unbind(sample.actions[:, index], dim=-1))
                onehot_actions = [action.float() for action in onehot_actions]
                ractions = self.policy.actor.reparam(act_dists, onehot_actions)

                # Transition reparametrization
                # with FreezeParameters([self.policy.transition_dist]):
                transition_dist = self.policy.transition_dist(
                    step_tuple,
                    ractions,
                    # transition_hidden
                    # transition_dist_logits
                )
                
                radius = self.spectral_radius(step_tuple, ractions, sample_size=10)
                r_encoding = self.policy.transition_dist.reparam(
                    transition_dist, next_step_tuple.encoding.detach())
                normalized_r_encoding = torch.einsum("b...,b->...", r_encoding, 1/radius)
                r_encoding = normalized_r_encoding + (r_encoding - normalized_r_encoding).detach()

                prev_terminations = sample.terminations[:, index]
                # transition_dist_logits = transition_dist.logits

                # Policy loss
                # next_state_log_prob = transition_dist.log_prob(
                #     next_step_tuple.encoding.detach()).reshape(td_errors.shape[0], -1).sum(1)
                next_state_prob = transition_dist.probs * next_step_tuple.encoding.detach()
                next_state_prob = next_state_prob.reshape(next_state_prob.shape[0], -1).sum(-1)
                next_state_log_prob = next_state_prob / (next_state_prob + 1e-5).detach()

                policy_loss = -(next_state_log_prob / self.moving_avg_td *
                                td_errors[:, index].detach()).mean(0)

                # Actor entropy loss
                entropy_loss = -sum(dist.entropy().mean() for dist in act_dists)

                local_log.entropy.append(entropy_loss)
                local_log.policy.append(policy_loss)
                local_log.next_state_logp.append(next_state_log_prob.mean(0).item())

            policy_loss = sum(local_log.policy) / meta_data.horizon
            entropy_loss = sum(local_log.entropy) / meta_data.horizon

            logger.record_mean("train/next_state_logp",
                               sum(local_log.next_state_logp) / meta_data.horizon)
            logger.record_mean("train/average_td", self.moving_avg_td)
        else:
            latent_tuple = DiscreteLatentTuple(
                None,
                None,
                latent_tuple.encoding.reshape(np.product(
                    latent_tuple.encoding.shape[:2]),
                    *latent_tuple.encoding.shape[2:])
            )
            actor_dists = self.policy.actor(latent_tuple)
            actions = [act.flatten() for act in torch.unbind(sample.actions, dim=-1)]
            log_prob = sum(dist.log_prob(act) for dist, act in zip(actor_dists, actions))
            a2c_loss = - log_prob * advantages.flatten().detach()
            policy_loss = a2c_loss.mean()
            entropy_loss = -sum(dist.entropy().mean() for dist in actor_dists)

        logger.record_mean("train/entropy_loss", entropy_loss.item())
        logger.record_mean("train/policy_loss", policy_loss.item())
        logger.record_mean("train/value_loss", value_loss.item())

        return policy_loss, entropy_loss, value_loss

    def spectral_radius(self, step_tuple, ractions, sample_size: int = 100, multiplier: float = 2):
        pre_shape = step_tuple.encoding.shape
        step_tuple = DiscreteLatentTuple(
            None,
            None,
            step_tuple.encoding.detach().repeat(sample_size, *[1]*len(step_tuple.encoding.shape[1:]))
        )
        step_tuple.encoding.requires_grad = True
        step_tuple.encoding.retain_grad()

        ractions = [raction.detach().repeat(sample_size, *[1]*len(raction.shape[1:]))
                    for raction in ractions]

        vectors = torch.randn(*step_tuple.encoding.shape)
        flatten_vector = vectors.reshape(vectors.shape[0], -1)
        flatten_vector = flatten_vector / torch.sqrt((flatten_vector**2).sum(1, keepdims=True))
        vectors = flatten_vector.reshape(vectors.shape)

        transition_dist = self.policy.transition_dist(
            step_tuple,
            ractions)

        (transition_dist.probs * vectors).sum().backward()
        grads = step_tuple.encoding.grad.reshape(sample_size, *pre_shape)
        grads = grads.reshape(sample_size, pre_shape[0], np.product(pre_shape[1:])).norm(dim=-1)
        approx_eigen_vals, _ = grads.max(dim=0)
        return (approx_eigen_vals * multiplier).detach()
