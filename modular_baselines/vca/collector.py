import torch
import numpy as np
import gym
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from modular_baselines.collectors.collector import BaseOnPolicyCollector


class NStepCollector(BaseOnPolicyCollector):
    """ On Policy collector that is mainly used with value based PG algorithms
    """

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

            actions, *values_to_save = self.process_step()
            new_obs, rewards, dones, infos = self.env.step(actions)

            self.num_timesteps += self.env.num_envs
            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            next_obs = new_obs
            if np.any(dones):
                next_obs = np.zeros_like(new_obs)
                next_obs[0] = infos[0]["terminal_observation"]
                
            self.buffer.add(self._last_obs,
                            next_obs,
                            actions,
                            rewards,
                            dones,
                            *values_to_save)

            self._last_obs = new_obs
            self._last_dones = dones

            for callback in self.callbacks:
                callback.on_rollout_step(locals())

        for callback in self.callbacks:
            callback.on_rollout_end(locals())

        return self.num_timesteps

    def process_step(self) -> torch.Tensor:
        """ Sample actions, values and log probability of the actions using 
        the policy

        Returns:
            Tuple[torch.Tensor]: actions, values, and log probabilities tensors
        """

        with torch.no_grad():
            obs_tensor = torch.from_numpy(self._last_obs).to(self.device)
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                obs_tensor = make_onehot(
                    obs_tensor, self.env.observation_space.n)
            actions = self.policy.forward(obs_tensor)
        actions = actions.cpu().numpy().argmax(1)
        actions = np.expand_dims(actions, 1)
        return actions


def make_onehot(tensor: torch.Tensor, maxsize: int) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor)
    if len(tensor.shape) == 2:
        assert tensor.shape[1] == 1, ""
    else:
        assert len(tensor.shape) == 1, ""
    return (tensor.reshape(-1, 1) == torch.arange(maxsize).reshape(1, -1)).float()
