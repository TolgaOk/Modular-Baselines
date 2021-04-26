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


class LatentA2C(A2C):
    """ Latent A2C algorithm.
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
                 kl_beta_coef: float,
                 model_update_freq: float,
                 model_batch_size: int,
                 max_grad_norm: float,
                 batch_size: int = None,
                 normalize_advantage: bool = False,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):

        self.kl_beta_coef = kl_beta_coef
        self.model_update_freq = model_update_freq
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

        update_freq = self._calculate_update_freq()
        losses = {
            "vae_kl_loss": [],
            "reconstruction_loss": []
        }
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)

            observation = sample.observations.to(self.device)
            observation = self.policy._preprocess(observation)
            _, prediction, dist = self.policy.vae(observation)
            recon_loss, kl_loss = self.policy.vae.loss(observation, prediction, dist)

            self.policy.optimizer.zero_grad()
            (recon_loss + self.kl_beta_coef * kl_loss).backward()
            self.policy.optimizer.step()

            losses["vae_kl_loss"].append(kl_loss.item())
            losses["reconstruction_loss"].append(recon_loss.item())

        self._log_losses(losses)

    def _calculate_update_freq(self):
        floor_freq = int(self.model_update_freq)
        return (np.random.random() < (self.model_update_freq - floor_freq)) + floor_freq

    def _log_losses(self, loss_container: Dict):
        for name, losses in loss_container.items():
            if len(losses) > 0:
                logger.record("train/{}".format(name), np.mean(losses))


class TransitionModelA2C(LatentA2C):

    def train(self) -> None:
        A2C.train(self)

        update_freq = self._calculate_update_freq()
        losses = {
            "vae_kl_loss": [],
            "reconstruction_loss": []
        }
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)

            actions = sample.actions.to(self.device)
            assert len(actions.shape) == 2

            observation = self.policy._preprocess(sample.observations.to(self.device))
            next_observation = self.policy._preprocess(sample.next_observations.to(self.device))

            _, prediction, dist = self.policy.vae(observation, actions)
            recon_loss, kl_loss = self.policy.vae.loss(next_observation, prediction, dist)

            self.policy.optimizer.zero_grad()
            (recon_loss + self.kl_beta_coef * kl_loss).backward()
            self.policy.optimizer.step()

            losses["vae_kl_loss"].append(kl_loss.item())
            losses["reconstruction_loss"].append(recon_loss.item())

        self._log_losses(losses)


class LatentActionPredA2C(LatentA2C):

    def train(self) -> None:
        A2C.train(self)

        update_freq = self._calculate_update_freq()
        losses = {
            "vae_kl_loss": [],
            "reconstruction_loss": []
        }
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)

            actions = sample.actions.to(self.device)
            assert len(actions.shape) == 2

            observation = self.policy._preprocess(sample.observations.to(self.device))
            next_observation = self.policy._preprocess(sample.next_observations.to(self.device))

            prediction, dists = self.policy.vae(observation, next_observation)
            recon_loss, kl_loss = self.policy.vae.loss(actions, prediction, dists)

            self.policy.optimizer.zero_grad()
            (recon_loss + self.kl_beta_coef * kl_loss).backward()
            self.policy.optimizer.step()

            losses["vae_kl_loss"].append(kl_loss.item())
            losses["reconstruction_loss"].append(recon_loss.item())

        self._log_losses(losses)


class LatentTransitionPredA2C(LatentA2C):

    def train(self) -> None:
        A2C.train(self)

        update_freq = self._calculate_update_freq()
        losses = {
            "vae_kl_loss": [],
            "reconstruction_loss": []
        }
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)

            actions = sample.actions.to(self.device)
            assert len(actions.shape) == 2

            observation = self.policy._preprocess(sample.observations.to(self.device))
            next_observation = self.policy._preprocess(sample.next_observations.to(self.device))

            next_observation_latent, prediction, dists = self.policy.vae(
                observation, next_observation, actions)
            recon_loss, kl_loss = self.policy.vae.loss(next_observation_latent, prediction, dists)

            self.policy.optimizer.zero_grad()
            (recon_loss + self.kl_beta_coef * kl_loss).backward()
            self.policy.optimizer.step()

            losses["vae_kl_loss"].append(kl_loss.item())
            losses["reconstruction_loss"].append(recon_loss.item())

        self._log_losses(losses)


class FullLatentA2C(LatentA2C):

    def train(self):

        A2C.train(self)
        update_freq = self._calculate_update_freq()
        losses = {
            "vae_kl_loss": [],
            "reconstruction_loss": [],
            "action_recon_loss": [],
            "trans_recon_loss": [],
        }
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)

            actions = sample.actions.to(self.device)
            assert len(actions.shape) == 2

            observation = self.policy._preprocess(sample.observations.to(self.device))
            next_observation = self.policy._preprocess(sample.next_observations.to(self.device))

            next_obs_dist, prediction_dist, dists, action_pred = self.policy.vae(
                observation, next_observation, actions)
            action_recon_loss, trans_recon_loss, kl_loss = self.policy.vae.loss(
                actions, action_pred, next_obs_dist, prediction_dist, dists)

            self.policy.optimizer.zero_grad()
            (action_recon_loss + trans_recon_loss + self.kl_beta_coef * kl_loss).backward()
            self.policy.optimizer.step()

            losses["vae_kl_loss"].append(kl_loss.item())
            losses["reconstruction_loss"].append(
                action_recon_loss.item() + trans_recon_loss.item())
            losses["action_recon_loss"].append(action_recon_loss.item())
            losses["trans_recon_loss"].append(trans_recon_loss.item())

        self._log_losses(losses)
