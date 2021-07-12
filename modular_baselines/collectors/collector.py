import torch
import numpy as np

from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env import VecEnv

from modular_baselines.buffers.buffer import BaseBuffer
from modular_baselines.policies.policy import BasePolicy


class BaseCollectorCallback(ABC):
    """ Base class for buffer callbacks that only supports:
    on_rollout_start, on_rollout_step, and on_rollout_end calls.
    """

    @abstractmethod
    def on_rollout_start(self, *args) -> None:
        pass

    @abstractmethod
    def on_rollout_step(self, *args) -> None:
        pass

    @abstractmethod
    def on_rollout_end(self, *args) -> None:
        pass


class BaseCollector(ABC):
    """ Base abstract class for collectors """

    @abstractmethod
    def collect(self, n_rollout_steps: int) -> int:
        pass


class RolloutCollector(BaseCollector):
    """ Collects a rollout with every collect
    call and add the experience into the buffer.

    Args:
        env (VecEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to load rollout experiences
        policy (torch.nn.Module): Policy module for action sampling
        callbacks (List[BaseCollectorCallback], optional): Collector callbacks.
            Defaults to [].
        device (str, optional): torch device that is used to convert
            experiences before feeding them into the policy module. Defaults
            to "cpu".
    """

    _fields = ("observation", "next_observation", "reward",
               "termination", "action", "policy_state")

    def __init__(self,
                 env: VecEnv,
                 buffer: BaseBuffer,
                 policy: BasePolicy,
                 callbacks: Optional[Union[List[BaseCollectorCallback],
                                           BaseCollectorCallback]] = None):
        super().__init__()

        self.env = env
        self.buffer = buffer
        self.policy = policy

        for field in self._fields:
            assert field in buffer.struct.names, (
                "Buffer does not contain the field name {}".format(field))

        self.num_timesteps = 0
        self._last_policy_state = policy.init_state(batch_size=self.env.num_envs)
        self._last_obs = self.env.reset()

        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks] if callbacks is not None else []
        self.callbacks = callbacks

    def collect(self, n_rollout_steps: int) -> int:
        """ Collect a rollout of experience using the policy and load them to
        buffer

        Args:
            n_rollout_steps (int): Length of the rollout

        Returns:
            int: Total number of timesteps passed.
        """

        n_steps = 0
        for callback in self.callbacks:
            callback.on_rollout_start(locals())

        while n_steps < n_rollout_steps:

            actions, policy_state, policy_context = self.policy.action_sample(
                self._last_obs, self._last_policy_state)
            new_obs, rewards, dones, infos = self.env.step(actions)
            next_obs = new_obs
            # Terminated new_obs is different than next_obs
            terminated_indexes = np.argwhere(dones == 1).flatten()
            if len(terminated_indexes) > 0:
                next_obs = new_obs.copy()
                for index in terminated_indexes:
                    next_obs[index] = infos[index]["terminal_observation"]

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if self._last_policy_state is not None:
                policy_context["policy_state"] = self._last_policy_state
            self.buffer.push({
                "observation": self._last_obs,
                "next_observation": next_obs,
                "reward": rewards,
                "termination": dones,
                "action": actions,
                **policy_context
            })

            self._last_obs = new_obs
            self._last_policy_state = policy_state

            for callback in self.callbacks:
                callback.on_rollout_step(locals())

        for callback in self.callbacks:
            callback.on_rollout_end(locals())

        return self.num_timesteps
