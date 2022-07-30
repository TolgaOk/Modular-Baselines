from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.algorithms.ppo.ppo import PPO
from modular_baselines.utils.annealings import Coefficient
from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger


@dataclass(frozen=True)
class VgPPOArgs():
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    policy_epochs: int
    model_epochs: int
    buffer_size: int
    max_grad_norm: float
    normalize_advantage: bool
    clip_value: Coefficient
    policy_batch_size: int
    model_batch_size: int
    policy_lr: Coefficient
    model_lr: Coefficient


class VgPPO(PPO):

    def train(self) -> Dict[str, float]:
        """ One step training. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.agent.train_mode()
        recent_sample = self.buffer.sample(
            batch_size=self.num_envs,
            rollout_len=self.args.rollout_len,
            sampling_length=self.args.rollout_len)
        random_samples = [self.buffer.sample(
            batch_size=self.args.model_batch_size,
            rollout_len=1,
            sampling_length=None)
            for _ in range(self.args.model_epochs)]
        return self.agent.update_parameters(
            recent_sample=recent_sample,
            random_samples=random_samples,
            value_coef=self.args.value_coef,
            ent_coef=self.args.ent_coef,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            policy_epochs=self.args.policy_epochs,
            model_epochs=self.args.model_epochs,
            clip_value=next(self.args.clip_value),
            policy_batch_size=self.args.policy_batch_size,
            policy_lr=next(self.args.policy_lr),
            model_lr=next(self.args.model_lr),
            max_grad_norm=self.args.max_grad_norm,
            normalize_advantage=self.args.normalize_advantage,
        )

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: VgPPOArgs,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "VgPPO":
        observation_space, action_space, action_dim = PPO._setup(env)
        if args.buffer_size < args.rollout_len:
            raise ValueError(
                f"Buffer size: {args.buffer_size} can not be smaller than the rollout length: {args.rollout_len}")

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
        ])
        buffer = Buffer(struct, args.buffer_size, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, agent, data_logger, collector_callbacks)
        return VgPPO(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            callbacks=algorithm_callbacks
        )
