from typing import List, Optional, Union, Dict
from abc import abstractmethod
import os
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.policies.policy import BasePolicy
from modular_baselines.algorithms.a2c.a2c import A2C
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing


class PPOPolicy(BasePolicy):
    """ Base PPO class for different frameworks """

    @abstractmethod
    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          epochs: int,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
        pass


class PPO(A2C):

    def __init__(self,
                 policy: PPOPolicy,
                 collector: RolloutCollector,
                 rollout_len: int,
                 ent_coef: float,
                 value_coef: float,
                 gamma: float,
                 gae_lambda: float,
                 epochs: int,
                 clip_value: Coefficient,
                 batch_size: int,
                 max_grad_norm: float,
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        super().__init__(policy,
                         collector,
                         rollout_len,
                         ent_coef,
                         value_coef,
                         gamma,
                         gae_lambda,
                         max_grad_norm,
                         callbacks)
        self.epochs = epochs
        self.clip_value = clip_value
        self.batch_size = batch_size

    def train(self) -> Dict[str, float]:
        """ One step traning. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.policy.train()
        sample = self.buffer.sample(batch_size=self.num_envs,
                                    rollout_len=self.rollout_len,
                                    sampling_length=self.rollout_len)
        return self.policy.update_parameters(
            sample,
            value_coef=self.value_coef,
            ent_coef=self.ent_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            epochs=self.epochs,
            clip_value=next(self.clip_value),
            batch_size=self.batch_size,
            max_grad_norm=self.max_grad_norm)

    @staticmethod
    def setup(env: VecEnv,
              policy: PPOPolicy,
              rollout_len: int,
              ent_coef: float,
              value_coef: float,
              gamma: float,
              gae_lambda: float,
              epochs: int,
              clip_value: Coefficient,
              batch_size: int,
              max_grad_norm: float,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "PPO":
        # TODO: Add different observation spaces
        observation_space = env.observation_space
        # TODO: Add different action spaces
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise NotImplementedError("Only Box observations are available")
        if not isinstance(action_space, (spaces.Box, spaces.Discrete)):
            raise NotImplementedError("Only Discrete and Box actions are available")
        policy_states_dtype = []
        # Check for reccurent policy
        policy_state = policy.init_state()
        if policy_state is not None:
            policy_states_dtype = [
                ("policy_state", np.float32, policy_state.shape),
                ("next_policy_state", np.float32, policy_state.shape)
            ]
        action_dim = action_space.shape[-1] if isinstance(action_space, spaces.Box) else 1

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
            *policy_states_dtype
        ])
        buffer = Buffer(struct, rollout_len, env.num_envs, buffer_callbacks)
        collector = RolloutCollector(env, buffer, policy, collector_callbacks)
        return PPO(
            policy=policy,
            collector=collector,
            rollout_len=rollout_len,
            ent_coef=ent_coef,
            value_coef=value_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            epochs=epochs,
            clip_value=clip_value,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            callbacks=algorithm_callbacks
        )
