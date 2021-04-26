"""A2C based actor critic implementation with Jacobian trace """
import numpy as np
import torch
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean, configure_logger
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.collectors.callbacks import BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.buffers.buffer import GeneralBuffer
from modular_baselines.algorithms.a2c import A2C


class JTAC(A2C):
    """ JTAC algorithm.
        Based on the implementation given within Stable-Baselines3

    Args:
        policy (torch.nn.Module): Policy module
        rollout_buffer (RolloutBuffer): Rollout buffer
        collector (OnPolicyCollector): Experience Collector
        env (VecEnv): Vectorized environment
        rollout_len (int): Length of the rollout
        ent_coef (float): Entropy coefficient/multiplier
        vf_coef (float): Value loss coefficient/multiplier
        max_grad_norm (float): Maximum allowed gradient norm
        normalize_advantage (bool, optional): Whether to normalize the
            advantage or not. Defaults to False.
        callbacks (List[BaseAlgorithmCallback], optional): Algorithm callbacks.
            Defaults to [].
        device (str, optional): Torch device. Defaults to "cpu".

    Raises:
        ValueError: Policy class must have "optimizer" and "evaluate_actions"
            members
    """

    def __init__(self,
                 policy: torch.nn.Module,
                 rollout_buffer: GeneralBuffer,
                 collector: OnPolicyCollector,
                 env: VecEnv,
                 rollout_len: int,
                 ent_coef: float,
                 vf_coef: float,
                 gamma: float,
                 gae_lambda: float,
                 prior_kl_coef: float,
                 trans_kl_coef: float,
                 model_batch_size: int,
                 max_grad_norm: float,
                 batch_size: int = None,
                 normalize_advantage: bool = False,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):

        self.prior_kl_coef = prior_kl_coef
        self.trans_kl_coef = trans_kl_coef
        self.model_batch_size = model_batch_size
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
                     batch_size=batch_size,
                     normalize_advantage=normalize_advantage,
                     callbacks=callbacks,
                     device=device)

    def train(self) -> None:
        super().train()
        self.policy.model.optimizer.zero_grad()
        self.model_loss().backward()
        self.policy.model.optimizer.step()

    def model_loss(self):
        losses = {
            "transition_kl_loss": [],
            "prior_kl_loss": [],
            "reconstruction_loss": [],
            "model_loss": []
        }
        sample = self.buffer.sample(self.model_batch_size)

        actions = sample.actions.to(self.device)
        observation = self.policy._preprocess(sample.observations.to(self.device))
        next_observation = self.policy._preprocess(sample.next_observations.to(self.device))
        combined_obs = torch.cat([observation, next_observation], dim=0)

        logits, state_dist, rstate = self.policy.model(combined_obs)

        decoder_features = self.policy.model.make_feature(logits, rstate)
        pred_next_obs_dist = self.policy.model.decoder(decoder_features)

        recon_loss = -pred_next_obs_dist.log_prob(combined_obs).mean()
        prior_state_dist = self.policy.model.make_normal_prior(state_dist)
        prior_kl_loss = torch.distributions.kl_divergence(
            state_dist, prior_state_dist).mean()

        _, next_state_dist = self.policy.model.batch_chunk_distribution(state_dist)
        prev_logits, _ = torch.chunk(logits, 2, dim=0)
        prev_rstate, _ = torch.chunk(rstate, 2, dim=0)
        pred_next_state_dist = self.policy.model.make_transition_dist(
            prev_logits, prev_rstate, actions)

        transition_kl_loss = torch.distributions.kl_divergence(
            pred_next_state_dist, next_state_dist).mean()

        model_loss = transition_kl_loss * self.trans_kl_coef + \
            prior_kl_loss * self.prior_kl_coef + recon_loss

        losses["transition_kl_loss"].append(transition_kl_loss.item())
        losses["prior_kl_loss"].append(prior_kl_loss.item())
        losses["reconstruction_loss"].append(recon_loss.item())
        losses["model_loss"].append(model_loss.item())

        self._log_losses(losses)

        return model_loss

    def _log_losses(self, loss_container: Dict):
        for name, losses in loss_container.items():
            if len(losses) > 0:
                logger.record("train/{}".format(name), np.mean(losses))
