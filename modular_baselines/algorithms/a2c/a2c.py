from typing import List, Optional, Union, Dict
from abc import abstractmethod
import os
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.policy import BasePolicy
from modular_baselines.loggers.data_logger import DataLogger, ListLog, DataLog


class A2CPolicy(BasePolicy):
    """ Base A2C class for different frameworks """

    @abstractmethod
    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          max_grad_norm: float,
                          ) -> Dict[str, float]:
        """ Update policy parameters using the given sample and a2c update mechanism.

            Args:
                sample (np.ndarray): Sample that contains at least observation, next_observation,
                    reward, termination, action, and policy_state if the policy is a reccurent policy.

            Returns:
                Dict[str, float]: Dictionary of losses to log
            """


class A2C(OnPolicyAlgorithm):
    """ A2C agent

    Args:
        policy (A2CPolicy): A2C policy of the selected framework
        collector (RolloutCollector): Rollout collector instance
        rollout_len (int): Length of the rollout per update
        ent_coef (float): Entropy loss multiplier
        value_coef (float): Value loss multiplier
        gamma (float): Discount rate
        gae_lambda (float): Eligibility trace lambda parameter
        max_grad_norm (float): Gradient clipping magnitude
        callbacks (Optional[Union[List[BaseAlgorithmCallback], BaseAlgorithmCallback]],
            optional): Algorithm callback(s). Defaults to None.
    """

    def __init__(self,
                 policy: A2CPolicy,
                 collector: RolloutCollector,
                 rollout_len: int,
                 ent_coef: float,
                 value_coef: float,
                 gamma: float,
                 gae_lambda: float,
                 max_grad_norm: float,
                 logger: DataLogger,
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        super().__init__(policy=policy,
                         collector=collector,
                         rollout_len=rollout_len,
                         logger=logger,
                         callbacks=callbacks)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

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
            max_grad_norm=self.max_grad_norm)

    def save(self, path: str) -> None:
        self.policy.save(os.path.join(path, "policy_params.b"))

    @staticmethod
    def setup(env: VecEnv,
              policy: A2CPolicy,
              rollout_len: int,
              ent_coef: float,
              value_coef: float,
              gamma: float,
              gae_lambda: float,
              max_grad_norm: float,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "A2C":
        """ Setup and A2C agent for the given environment with the given policy.

        Args:
            env (VecEnv): Vectorized gym environment
            policy (A2CPolicy): A2C policy of the selected framework
            rollout_len (int): Length of the rollout per update
            ent_coef (float): Entropy loss multiplier
            value_coef (float): Value loss multiplier
            gamma (float): Discount rate
            gae_lambda (float): Eligibility trace lambda parameter
            max_grad_norm (float): Gradient clipping magnitude
            buffer_callbacks (Optional[Union[List[BaseBufferCallback], BaseBufferCallback]],
                optional): Buffer callback(s). Defaults to None.
            collector_callbacks (Optional[Union[List[BaseCollectorCallback],
                BaseCollectorCallback]], optional): Collector callback(s). Defaults to None.
            algorithm_callbacks (Optional[Union[List[BaseAlgorithmCallback],
                BaseAlgorithmCallback]], optional): Algorithm callback(s). Defaults to None.

        Raises:
            NotImplementedError: If the observation space is not Box
            NotImplementedError: If the action space is not Discrete

        Returns:
            A2C: a2c agent
        """
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
        policy_state = policy.init_hidden_state()
        if policy_state is not None:
            policy_states_dtype = [("policy_state", np.float32, policy_state.shape)]
        action_dim = action_space.shape[-1] if isinstance(action_space, spaces.Box) else 1

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            *policy_states_dtype
        ])

        data_logger = DataLogger()
        policy.set_logger(data_logger)
        buffer = Buffer(struct, rollout_len, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, policy, data_logger, collector_callbacks)
        return A2C(policy, collector, rollout_len, ent_coef, value_coef,
                   gamma, gae_lambda, max_grad_norm, data_logger, algorithm_callbacks)
