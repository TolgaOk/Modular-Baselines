from typing import List, Optional, Union, Dict
from abc import abstractmethod
import os
from gym import spaces
import numpy as np
from dataclasses import dataclass

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing
from modular_baselines.loggers.data_logger import DataLogger


@dataclass(frozen=True)
class PPOArgs():
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    epochs: int
    lr: Coefficient
    clip_value: Coefficient
    batch_size: int
    max_grad_norm: float
    normalize_advantage: bool


class PPO(OnPolicyAlgorithm):

    def __init__(self,
                 agent: BaseAgent,
                 collector: RolloutCollector,
                 args: PPOArgs,
                 logger: DataLogger,
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        super().__init__(agent=agent,
                         collector=collector,
                         rollout_len=args.rollout_len,
                         logger=logger,
                         callbacks=callbacks)
        self.args = args

    def train(self) -> Dict[str, float]:
        """ One step training. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.agent.train_mode()
        sample = self.buffer.sample(batch_size=self.num_envs,
                                    rollout_len=self.args.rollout_len,
                                    sampling_length=self.args.rollout_len)
        return self.agent.update_parameters(
            sample,
            value_coef=self.args.value_coef,
            ent_coef=self.args.ent_coef,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            epochs=self.args.epochs,
            lr=next(self.args.lr),
            clip_value=next(self.args.clip_value),
            batch_size=self.args.batch_size,
            max_grad_norm=self.args.max_grad_norm,
            normalize_advantage=self.args.normalize_advantage)

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: PPOArgs,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "PPO":
        observation_space, action_space, action_dim = PPO._setup(env)

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
        ])
        buffer = Buffer(struct, args.rollout_len, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, agent, data_logger, collector_callbacks)
        return PPO(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            callbacks=algorithm_callbacks
        )
