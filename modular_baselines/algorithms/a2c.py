import numpy as np
import torch
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean, configure_logger
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.a2c.a2c import A2C as SB3_A2C

from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.collectors.callbacks import BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm
from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.buffers.buffer import GeneralBuffer


class A2C(OnPolicyAlgorithm, SB3_A2C):
    """ A2C on policy algorithm.
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
                 max_grad_norm: float,
                 batch_size: int = None,
                 normalize_advantage: bool = False,
                 callbacks: List[BaseAlgorithmCallback] = [],
                 device: str = "cpu"):

        # for name in ("optimizer", "evaluate_actions"):
        #     if not hasattr(policy, name):
        #         raise ValueError("Policy has no attribute {}".format(name))

        OnPolicyAlgorithm.__init__(self,
                                   policy=policy,
                                   buffer=rollout_buffer,
                                   collector=collector,
                                   env=env,
                                   rollout_len=rollout_len,
                                   callbacks=callbacks,
                                   device=device)
        self.rollout_buffer = RolloutBufferWrapper(
            self.buffer, rollout_len, batch_size)
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._n_updates = 0

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

    def train(self) -> None:
        """ Train method of the SB3 """
        self.buffer.compute_returns_and_advantage(
            rollout_size=self.rollout_len,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda)
        SB3_A2C.train(self)

    def _update_learning_rate(self, *args):
        pass


class RolloutBufferWrapper():

    def __init__(self, buffer: GeneralBuffer, rollout_size: int, batch_size: int):
        self.buffer = buffer
        self.rollout_size = rollout_size
        self.batch_size = batch_size

    def __getattr__(self, name: str):
        return getattr(self.buffer, name)

    def get(self, **kwargs):
        return self.buffer.get_rollout(self.rollout_size, self.batch_size)
