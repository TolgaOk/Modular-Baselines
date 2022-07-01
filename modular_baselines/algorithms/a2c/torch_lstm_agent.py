from typing import Tuple, Union, Dict, Generator, Any, List
from abc import abstractmethod
import numpy as np
import torch
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.a2c.torch_agent import TorchA2CAgent
from modular_baselines.loggers.data_logger import ListLog


class TorchLSTMA2CAgent(TorchA2CAgent):

    def init_hidden_state(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {name: np.zeros((batch_size, size), dtype=np.float32)
                for name, size in self.policy.hidden_state_info.items()}

    def sample_action(self,
                      observation: np.ndarray,
                      hidden_state: Dict[str, np.ndarray]
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:

        with torch.no_grad():
            th_observation = self.to_torch(observation).float()
            th_hidden_state = {name: self.to_torch(array) for name, array in hidden_state.items()}

            policy_dist, _, th_hidden_state = self.policy(th_observation, th_hidden_state)
            th_action = policy_dist.sample()
            if isinstance(self.action_space, Discrete):
                th_action = th_action.unsqueeze(-1)
        hidden_state = {name: tensor.cpu().numpy() for name, tensor in th_hidden_state.items()}
        return th_action.cpu().numpy(), hidden_state, {}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_next_obs, th_hidden_states, th_next_hidden_states = self.to_torch(
            [sample["observation"],
             sample["action"],
             sample["next_observation"][:, -1],
             {name: sample[name] for name in self.policy.hidden_state_info.keys()},
             {name: sample[f"next_{name}"][:, -1] for name in self.policy.hidden_state_info.keys()}
             ])
        return th_obs, th_action, th_next_obs, th_hidden_states, th_next_hidden_states

    def prepare_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_next_obs, th_hidden_state, th_next_hidden_state = self.rollout_to_torch(
            sample)

        policy_dist, th_flatten_values, _ = self.policy(*self.flatten_time([th_obs, th_hidden_state]))

        tf_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, th_next_value, _ = self.policy(th_next_obs, th_next_hidden_state)
        advantages, returns = self.to_torch(calculate_gae(
                rewards=sample["reward"],
                terminations=sample["termination"],
                values=tf_values.detach().cpu().numpy(),
                last_value=th_next_value.detach().cpu().numpy(),
                gamma=gamma,
                gae_lambda=gae_lambda)
        )

        return (*self.flatten_time([advantages, returns, th_action]), policy_dist, th_flatten_values)
