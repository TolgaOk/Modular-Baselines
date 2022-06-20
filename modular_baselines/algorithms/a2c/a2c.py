from typing import List, Optional, Union, Dict, Tuple, Any
import os
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger


class A2C(OnPolicyAlgorithm):
    """ A2C agent

    Args:
        agent (BaseAgent): A2C agent of the selected framework
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
                 agent: BaseAgent,
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
        super().__init__(agent=agent,
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
        """ One step training. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.agent.train_mode()
        sample = self.buffer.sample(batch_size=self.num_envs,
                                    rollout_len=self.rollout_len,
                                    sampling_length=self.rollout_len)
        return self.agent.update_parameters(
            sample,
            value_coef=self.value_coef,
            ent_coef=self.ent_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            max_grad_norm=self.max_grad_norm)

    def save(self, path: str) -> None:
        raise NotImplementedError
        # self.agent.policy.save(os.path.join(path, "agent_params.b"))

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
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
        """ Setup and A2C agent for the given environment with the given agent.

        Args:
            env (VecEnv): Vectorized gym environment
            agent (BaseAgent): An agent of the selected framework
            data_logger (DataLogger): Logger for saving training log data
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
        # Check for recurrent policy
        policy_state = agent.init_hidden_state()
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

        buffer = Buffer(struct, rollout_len, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, agent, data_logger, collector_callbacks)
        return A2C(agent, collector, rollout_len, ent_coef, value_coef,
                   gamma, gae_lambda, max_grad_norm, data_logger, algorithm_callbacks)
