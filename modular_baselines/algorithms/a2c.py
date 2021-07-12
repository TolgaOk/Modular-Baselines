from typing import List, Optional, Tuple, Union, Dict, Generator
from abc import abstractmethod
from gym import spaces
import numpy as np
import torch
from torch.types import Device
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.policies.policy import BasePolicy


class A2CPolicy(BasePolicy):
    """ Base A2C class for different frameworks """

    @abstractmethod
    def update_parameters(self, sample: np.ndarray) -> Dict[str, float]:
        """ Update policy parameters using the given sample and a2c update mechanism.

        Args:
            sample (np.ndarray): Sample that contains at least observation, next_observation,
                reward, termination, action, and policy_state if the policy is a reccurent policy.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        pass


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
                 callbacks: Optional[Union[List[BaseAlgorithmCallback],
                                           BaseAlgorithmCallback]] = None):
        super().__init__(policy=policy,
                         collector=collector,
                         rollout_len=rollout_len,
                         callbacks=callbacks)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def train(self) -> Dict[str: float]:
        """ One step traning. This will be called once per rollout.

        Returns:
            Dict[str: float]: Dictionary of losses to log
        """
        sample = self.buffer.sample(batch_size=self.num_env,
                                    rollout_len=self.rollout_len)
        return self.policy.update_parameters(
            sample,
            value_coef=self.value_coef,
            ent_coef=self.ent_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            max_grad_norm=self.max_grad_norm)

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
              collector_callback: Optional[Union[List[BaseCollectorCallback],
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
        # TODO: Add different observatoin spaces
        observation_space = env.observation_space
        # TODO: Add different action spaces
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise NotImplementedError("Only Box observations are available")
        if not isinstance(action_space, spaces.Discrete):
            raise NotImplementedError("Only Discrete actions are available")
        policy_states_dtype = []
        # Check for reccurent policy
        policy_state = policy.init_state()
        if policy_state is not None:
            policy_states_dtype = [("policy_state", np.float32, policy_state.shape)]

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", np.int32, (1,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            *policy_states_dtype
        ])
        buffer = Buffer(struct, rollout_len, env.num_envs, buffer_callbacks)
        collector = RolloutCollector(env, buffer, policy, collector_callback)
        return A2C(policy, collector, rollout_len, ent_coef, value_coef,
                   gamma, gae_lambda, max_grad_norm, algorithm_callbacks)


class TorchA2CPolicy(A2CPolicy):
    """ Pytorch A2C Policy base class """

    @property
    @abstractmethod
    def device(self) -> Device:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @property
    @abstractmethod
    def parameters(self) -> Generator[None, torch.nn.parameter.Parameter, None]:
        pass

    @abstractmethod
    def evaluate_rollout(self,
                         observation: torch.Tensor,
                         policy_state: Union[None, torch.Tensor],
                         action: torch.Tensor,
                         last_next_obseration: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass the given rollout. This is useful for both the advantage calculation
        and bacward pass. Note: BS -> batch size, R: rollout length

        Args:
            observation (torch.Tensor): Observation tensor with the shape (BS, R, *)
            policy_state (Union[None, torch.Tensor]): Policy state for reccurent models. None
                will be given if the buffer does not contain "policy_state" field.
            action (torch.Tensor): Action tensor with the shape (BS, R, *)
            last_next_obseration (torch.Tensor): [description]: Last observation tensor to
                calculate last value with the shape: (BS, *D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: values,
                log_probs, entropies, last_value
        """
        pass

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          max_grad_norm: float,
                          ) -> Dict[str: float]:
        """ Pytorch A2C parameter update method """
        batch_size, rollout_size = sample.shape
        policy_state = sample["policy_state"] if "policy_state" in sample.dtype.names else None
        values, log_probs, entropies, last_value = self.evaluate_rollout(
            *map(lambda array: torch.from_numpy(array).to(self.device),
                 (sample["observation"],
                  policy_state,
                  sample["action"],
                  sample["next_observation"][:, -1]))
        )

        advantages, returns = calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=values.detach().cpu().numpy(),
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda)

        advantages = torch.from_numpy(advantages).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)

        values, advantages, returns, log_probs = map(
            lambda tensor: tensor.reshape(batch_size * rollout_size, 1),
            (values, advantages, returns, log_probs))

        value_loss = torch.nn.functional.mse_loss(values, returns)
        policy_loss = (-log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        loss = value_loss * value_coef + policy_loss + entropy_loss * ent_coef

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

        return dict(value_loss=value_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy_loss=entropy_loss.item())
