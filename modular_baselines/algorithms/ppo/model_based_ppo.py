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
class ModelBasedPPOArgs():
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
    use_vec_normalization: bool


class ModelBasedPPO(PPO):

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

        self.agent.update_model_parameters(samples=random_samples,
                                           max_grad_norm=self.args.max_grad_norm,
                                           lr=next(self.args.model_lr))
        self.agent.update_policy_parameters(
            sample=recent_sample,
            value_coef=self.args.value_coef,
            ent_coef=self.args.ent_coef,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            epochs=self.args.policy_epochs,
            lr=next(self.args.policy_lr),
            clip_value=next(self.args.clip_value),
            batch_size=self.args.policy_batch_size,
            max_grad_norm=self.args.max_grad_norm,
            normalize_advantage=self.args.normalize_advantage,
        )

    @classmethod
    def setup(cls,
              env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: ModelBasedPPOArgs,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "ModelBasedPPO":
        observation_space, action_space, action_dim = PPO._setup(env)
        if args.buffer_size < args.rollout_len:
            raise ValueError(
                f"Buffer size: {args.buffer_size} can not be smaller than the rollout length: {args.rollout_len}")

        normalizer_struct = []
        if args.use_vec_normalization:
            normalizer_struct = [
                ("reward_rms_var", np.float32, (1,)),
                ("obs_rms_mean", np.float32, observation_space.shape),
                ("obs_rms_var", np.float32, observation_space.shape),
                ("next_obs_rms_mean", np.float32, observation_space.shape),
                ("next_obs_rms_var", np.float32, observation_space.shape),
            ]
        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
            *normalizer_struct
        ])
        buffer = Buffer(struct, args.buffer_size, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, agent, data_logger, collector_callbacks,
                                     store_normalizer_stats=args.use_vec_normalization)

        return cls(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            callbacks=algorithm_callbacks
        )


@dataclass(frozen=True)
class ValueGradientPPOArgs(ModelBasedPPOArgs):
    check_reparam_consistency: bool
    use_log_likelihood: bool
    mini_rollout_size: int
    use_reparameterization: bool
    policy_loss_beta: Coefficient


class ValueGradientPPO(ModelBasedPPO):

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
        self.agent.update_model_parameters(samples=random_samples,
                                           max_grad_norm=self.args.max_grad_norm,
                                           lr=next(self.args.model_lr))
        self.agent.update_policy_parameters(
            sample=recent_sample,
            reward_fn=self.collector.env.reward_fn,
            value_coef=self.args.value_coef,
            ent_coef=self.args.ent_coef,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            epochs=self.args.policy_epochs,
            lr=next(self.args.policy_lr),
            clip_value=next(self.args.clip_value),
            batch_size=self.args.policy_batch_size,
            max_grad_norm=self.args.max_grad_norm,
            normalize_advantage=self.args.normalize_advantage,
            mini_rollout_size=self.args.mini_rollout_size,
            check_reparam_consistency=self.args.check_reparam_consistency,
            use_log_likelihood=self.args.use_log_likelihood,
            is_vec_normalize=self.args.use_vec_normalization,
            use_reparameterization=self.args.use_reparameterization,
            policy_loss_beta=next(self.args.policy_loss_beta),
        )
