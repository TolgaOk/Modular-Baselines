from typing import List, Optional, Union, Dict, Tuple, Any
from gym import spaces
import numpy as np
from dataclasses import dataclass

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.collectors.recurrent import RecurrentRolloutCollector
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger
from modular_baselines.utils.annealings import Coefficient


@dataclass(frozen=True)
class A2CArgs():
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    lr: Coefficient
    max_grad_norm: float
    normalize_advantage: bool


class A2C(OnPolicyAlgorithm):
    """ A2C agent

    Args:
        agent (BaseAgent): A2C agent of the selected framework
        collector (RolloutCollector): Rollout collector instance
        args (A2CArgs): Hyperparameters of the algorithm
        callbacks (Optional[Union[List[BaseAlgorithmCallback], BaseAlgorithmCallback]],
            optional): Algorithm callback(s). Defaults to None.
    """

    def __init__(self,
                 agent: BaseAgent,
                 collector: RolloutCollector,
                 args: A2CArgs,
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
            lr=next(self.args.lr),
            max_grad_norm=self.args.max_grad_norm,
            normalize_advantage=self.args.normalize_advantage)

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: A2CArgs,
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
            args (A2CArgs): Hyperparameters of the algorithm
            buffer_callbacks (Optional[Union[List[BaseBufferCallback], BaseBufferCallback]], optional): Buffer callback(s). Defaults to None.
            collector_callbacks (Optional[Union[List[BaseCollectorCallback], BaseCollectorCallback]], optional): Collector callback(s). Defaults to None.
            algorithm_callbacks (Optional[Union[List[BaseAlgorithmCallback], BaseAlgorithmCallback]], optional): Algorithm callback(s). Defaults to None.

        Raises:
            NotImplementedError: If the observation space is not Box
            NotImplementedError: If the action space is not Discrete

        Returns:
            A2C: a2c agent
        """
        observation_space, action_space, action_dim = A2C._setup(env)

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
        ])
        buffer = Buffer(struct, args.rollout_len, env.num_envs, data_logger, buffer_callbacks)
        collector = RolloutCollector(env, buffer, agent, data_logger, collector_callbacks)
        return A2C(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            callbacks=algorithm_callbacks,
        )


class LstmA2C(A2C):
    """ LSTM based A2C agent """

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: A2CArgs,
              buffer_callbacks: Optional[Union[List[BaseBufferCallback],
                                               BaseBufferCallback]] = None,
              collector_callbacks: Optional[Union[List[BaseCollectorCallback],
                                                  BaseCollectorCallback]] = None,
              algorithm_callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                                  BaseAlgorithmCallback]] = None,
              ) -> "LstmA2C":
        """ A2C with LSTM agents

        Args:
            env (VecEnv): Vectorized gym environment
            agent (BaseAgent): LSTM based agent of selected framework
            data_logger (DataLogger): Logger for saving training log data
            args (A2CArgs): Hyperparameters of the algorithm
            buffer_callbacks (Optional[Union[List[BaseBufferCallback], BaseBufferCallback]], optional): Buffer callback(s). Defaults to None.
            collector_callbacks (Optional[Union[List[BaseCollectorCallback], BaseCollectorCallback]], optional): Collector callback(s). Defaults to None.
            algorithm_callbacks (Optional[Union[List[BaseAlgorithmCallback], BaseAlgorithmCallback]], optional): Algorithm callback(s). Defaults to None.

        Returns:
            LstmA2C: A2C with LSTM agent
        """
        observation_space, action_space, action_dim = A2C._setup(env)

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            *[(name, np.float32, array.shape[1:])
              for name, array in agent.init_hidden_state(1).items()],
            *[(f"next_{name}", np.float32, array.shape[1:])
              for name, array in agent.init_hidden_state(1).items()]
        ])

        buffer = Buffer(struct, args.rollout_len, env.num_envs, data_logger, buffer_callbacks)
        collector = RecurrentRolloutCollector(env, buffer, agent, data_logger, collector_callbacks)
        return A2C(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            callbacks=algorithm_callbacks,
        )
