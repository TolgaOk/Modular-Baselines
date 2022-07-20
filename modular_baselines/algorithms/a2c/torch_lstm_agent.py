from typing import Tuple, Union, Dict, Generator, Any, List
from abc import abstractmethod
from functools import partial
import numpy as np
import torch
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.a2c.torch_agent import TorchA2CAgent
from modular_baselines.algorithms.agent import BaseRecurrentAgent
from modular_baselines.loggers.data_logger import SequenceNormDataLog


class TorchLstmA2CAgent(TorchA2CAgent, BaseRecurrentAgent):

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

            policy_params, _, th_hidden_state = self.policy(th_observation, th_hidden_state)
            policy_dist = self.policy.dist(policy_params)
            th_action = policy_dist.sample()
            if isinstance(self.action_space, Discrete):
                th_action = th_action.unsqueeze(-1)
        hidden_state = {name: tensor.cpu().numpy() for name, tensor in th_hidden_state.items()}
        return th_action.cpu().numpy(), hidden_state, {}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_dones, th_next_obs, th_hidden_states, th_next_hidden_states = self.to_torch(
            [sample["observation"],
             sample["action"],
             sample["termination"],
             sample["next_observation"][:, -1],
             {name: sample[name] for name in self.policy.hidden_state_info.keys()},
             {name: sample[f"next_{name}"] for name in self.policy.hidden_state_info.keys()}
             ])
        return th_obs, th_action, th_dones, th_next_obs, th_hidden_states, th_next_hidden_states

    def replay_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_dones, th_next_obs, th_hidden_states, th_next_hidden_state = self.rollout_to_torch(
            sample)

        policy_params = []
        values = []
        th_reset_state = self.to_torch(self.init_hidden_state(env_size))
        th_hidden_state = {name: tensor[:, 0] for name, tensor in th_hidden_states.items()}

        for step in range(rollout_size):

            # Log gradient of hidden states at each step
            for name in self.policy.hidden_state_info.keys():
                hidden = th_hidden_state[name]
                hidden.requires_grad = True
                # For gradients
                hidden.register_hook(
                    partial(lambda time, _name, grad:
                            getattr(self.logger, f"dict/time_sequence/grad_{_name}").add(
                                time, grad.norm(dim=-1, p="fro").mean(0).item()), step, name))
                # For forward values
                getattr(self.logger, f"dict/time_sequence/forward_{name}").add(
                                step, hidden.norm(dim=-1, p="fro").mean(0).item())

            # Checking for consistency
            for name, hidden_tensor in th_hidden_states.items():
                if not torch.allclose(hidden_tensor[:, step], th_hidden_state[name], atol=1e-8):
                    raise RuntimeError("Sampled and calculated hidden states are not matched!")

            policy_param, value, th_hidden_state = self.policy(th_obs[:, step], th_hidden_state)
            policy_params.append(policy_param)
            values.append(value)

            # Checking for consistency
            for name, hidden_tensor in th_next_hidden_state.items():
                if not torch.allclose(hidden_tensor[:, step], th_hidden_state[name], atol=1e-8):
                    raise RuntimeError(
                        "Sampled and calculated next hidden states are not matched!")

            th_done = th_dones[:, step]
            for name, tensor in th_hidden_state.items():
                th_hidden_state[name] = (tensor * (1 - th_done) +
                                         th_reset_state[name] * th_done)

        th_values = torch.stack(values, dim=1)
        th_flatten_values = self.flatten_time(th_values)
        policy_params = self.flatten_time(torch.stack(policy_params, dim=1))
        policy_dist = self.policy.dist(policy_params)

        # Checking for value consistency
        test_policy_parameters, test_values, _ = self.policy(
            *self.flatten_time([th_obs, th_hidden_states]))
        if not torch.allclose(th_flatten_values, test_values):
            raise RuntimeError("Value mismatch")
        if not torch.allclose(policy_params, test_policy_parameters):
            raise RuntimeError("Policy dist parameters mismatch")

        _, th_next_value, _ = self.policy(
            th_next_obs, {name: tensor[:, -1] for name, tensor in th_next_hidden_state.items()})
        advantages, returns = self.to_torch(calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=th_values.detach().cpu().numpy(),
            last_value=th_next_value.detach().cpu().numpy(),
            gamma=gamma,
            gae_lambda=gae_lambda)
        )

        return (*self.flatten_time([advantages, returns, th_action]), policy_dist, th_flatten_values)

    def _init_default_loggers(self) -> None:
        super()._init_default_loggers()
        loggers = {
            f"dict/time_sequence/{group}_{name}": SequenceNormDataLog()
            for name in self.policy.hidden_state_info.keys()
            for group in ("grad", "forward")
        }
        self.logger.add_if_not_exists(loggers)
