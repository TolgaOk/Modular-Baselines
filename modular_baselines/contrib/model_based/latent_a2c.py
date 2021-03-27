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

        floor_freq = int(self.model_update_freq)
        update_freq = (np.random.random() < (self.model_update_freq - floor_freq)) + floor_freq
        kl_losses = []
        recon_losses = []
        for _ in range(update_freq):
            sample = self.buffer.sample(self.model_batch_size)
            
            observation = sample.observations.to(self.device)
            observation = self.policy._preprocess(observation)
            _, prediction, dist = self.policy.vae(observation)
            recon_loss, kl_loss = self.policy.vae.loss(observation, prediction, dist)

            self.policy.optimizer.zero_grad()
            (recon_loss + self.kl_beta_coef * kl_loss).backward
            self.policy.optimizer.step()

            kl_losses.append(kl_loss.item())
            recon_losses.append(recon_loss.item())

        if len(kl_losses) > 0:
            logger.record("train/reconstruction_loss", np.mean(recon_losses))
            logger.record("train/vae_kl_loss", np.mean(kl_losses))
