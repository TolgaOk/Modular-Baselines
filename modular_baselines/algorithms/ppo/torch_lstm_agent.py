from typing import Tuple, Union, Dict, Generator, Optional, Any, List
from abc import abstractmethod
import numpy as np
import torch
from torch.types import Device
from gym.spaces import Discrete

from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.algorithms.ppo.torch_agent import TorchPPOAgent
from modular_baselines.algorithms.agent import nested


class TorchLstmPPOAgent(TorchPPOAgent):
    """ Pytorch PPO Agent """

    def init_hidden_state(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {name: np.zeros((batch_size, size), dtype=np.float32)
                for name, size in self.policy.hidden_state_info.items()}

    def sample_action(self,
                      observation: np.ndarray,
                      hidden_state: Dict[str, np.ndarray]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action, content = super().sample_action(observation)
        return action, hidden_state, content

        with torch.no_grad():
            th_observation, th_hidden_state = self.to_torch(
                [observation.astype(np.float32), hidden_state])

            policy_params, _, th_next_hidden_state = self.policy(th_observation, th_hidden_state)
            policy_dist = self.policy.dist(policy_params)
            th_action = policy_dist.sample()
            log_prob = policy_dist.log_prob(th_action).unsqueeze(-1)
            if isinstance(self.action_space, Discrete):
                # th_action = th_action.unsqueeze(-1)
                raise NotImplementedError(
                    f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
        hidden_state = {name: tensor.cpu().numpy() for name, tensor in th_next_hidden_state.items()}
        return th_action.cpu().numpy(), hidden_state, {"old_log_prob": log_prob.cpu().numpy()}

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_dones, th_next_obs, th_old_log_prob, th_hidden_states, th_next_hidden_state = self.to_torch([
            sample["observation"],
            sample["action"],
            sample["termination"],
            sample["next_observation"][:, -1],
            sample["old_log_prob"],
            {name: sample[name] for name in self.policy.hidden_state_info.keys()},
            {name: sample[f"next_{name}"][:, -1] for name in self.policy.hidden_state_info.keys()}])
        return th_obs, th_action, th_dones, th_next_obs, th_old_log_prob, th_hidden_states, th_next_hidden_state

    def prepare_rollout(self, sample: np.ndarray, gamma: float, gae_lambda: float) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_dones, th_next_obs, th_old_log_prob, th_hidden_states, th_next_hidden_state = self.rollout_to_torch(
            sample)

        with torch.no_grad():
            _, th_flatten_values = self.replay_rollout(
                obs=th_obs,
                hidden_states=th_hidden_states,
                dones=th_dones,
                check_hidden=True)
            th_values = th_flatten_values.reshape(env_size, rollout_size, 1)
            _, th_next_value, _ = self.policy(th_next_obs, th_next_hidden_state)

        advantages, returns = self.to_torch(calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=th_values.detach().cpu().numpy(),
            last_value=th_next_value.detach().cpu().numpy(),
            gamma=gamma,
            gae_lambda=gae_lambda)
        )

        return (advantages, returns, th_action, th_old_log_prob, th_obs, th_hidden_states, th_dones)

    @nested
    def _make_mini_rollout(self, tensor: torch.Tensor, mini_rollout_size: int) -> torch.Tensor:
        return torch.cat(tensor.split(mini_rollout_size, dim=1), dim=0)

    @nested
    def _take_slice(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return tensor[indices]

    def rollout_loader(self,
                       batch_size: int,
                       advantages: torch.Tensor,
                       returns: torch.Tensor,
                       th_action: torch.Tensor,
                       th_old_log_prob: torch.Tensor,
                       *replay_data: torch.Tensor,
                       #    mini_rollout_size: int
                       ) -> Generator[Tuple[torch.Tensor], None, None]:
        env_size, rollout_size = advantages.shape[:2]
        if rollout_size % self.mini_rollout_size != 0:
            raise ValueError(
                f"rollout size {rollout_size} must be divisible to given mini rollout size {self.mini_rollout_size}")

        rollout_data = [advantages, returns, th_action, th_old_log_prob]
        mini_rollout_data, mini_rollout_replay_data = self._make_mini_rollout(
            [rollout_data, replay_data], mini_rollout_size=self.mini_rollout_size)
        perm_indices = torch.randperm(mini_rollout_replay_data[0].shape[0])

        for indices in perm_indices.split(batch_size):
            sliced_rollout_data, sliced_rollout_replay_data = self._take_slice(
                [mini_rollout_data, mini_rollout_replay_data], indices=indices)
            yield (*self.flatten_time(sliced_rollout_data), *sliced_rollout_replay_data)

    def replay_rollout(self,
                       obs: torch.Tensor,
                       hidden_states: Dict[str, torch.tensor],
                       dones: torch.Tensor,
                       #    use_sampled_hidden: bool = False,
                       check_hidden: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, rollout_size = obs.shape[:2]
        policy_param_list = []
        value_list = []
        th_reset_state = self.to_torch(self.init_hidden_state(batch_size))

        hidden_state = {name: hidden[:, 0] for name, hidden in hidden_states.items()}
        for step in range(rollout_size):
            sampled_hidden = {name: hidden[:, step] for name, hidden in hidden_states.items()}
            
            if check_hidden:
                for name, hidden in hidden_state.items():
                    if not torch.allclose(hidden, sampled_hidden[name], atol=1e-6):
                        raise RuntimeError("Sampled and calculated hidden states are not matched!")
            if self.use_sampled_hidden:
                hidden_state = sampled_hidden

            policy_param, value, hidden_state = self.policy(obs[:, step], hidden_state)
            policy_param_list.append(policy_param)
            value_list.append(value)

            done = dones[:, step]
            if done.any():
                for name, tensor in hidden_state.items():
                    hidden_state[name] = (tensor * (1 - done) +
                                          th_reset_state[name] * done).detach()

        policy_params = torch.cat(policy_param_list, dim=0)
        values = torch.cat(value_list, dim=0)
        return policy_params, values

    def update_parameters(self,
                          sample: np.ndarray,
                          value_coef: float,
                          ent_coef: float,
                          gamma: float,
                          gae_lambda: float,
                          epochs: int,
                          lr: float,
                          clip_value: float,
                          batch_size: int,
                          max_grad_norm: float,
                          normalize_advantage: bool,
                          mini_rollout_size: int,
                          use_sampled_hidden: bool,
                          ) -> Dict[str, float]:
        """ Pytorch A2C parameter update method """
        # Bad practice :/
        self.mini_rollout_size = mini_rollout_size
        self.use_sampled_hidden = use_sampled_hidden

        super().update_parameters(
            sample=sample, value_coef=value_coef, ent_coef=ent_coef, gamma=gamma,
            gae_lambda=gae_lambda, epochs=epochs, lr=lr, clip_value=clip_value,
            batch_size=batch_size, max_grad_norm=max_grad_norm, normalize_advantage=normalize_advantage,
        )
