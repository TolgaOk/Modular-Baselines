from typing import Tuple, Union, Dict, Generator, Optional, Any, List
from abc import abstractmethod
import os
from time import time
import torch
from gym import spaces
import numpy as np
from dataclasses import dataclass
from gym.spaces import Discrete, Space

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.collectors.recurrent import RecurrentRolloutCollector
from modular_baselines.algorithms.algorithm import OnPolicyAlgorithm, BaseAlgorithmCallback
from modular_baselines.buffers.buffer import Buffer, BaseBufferCallback
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing
from modular_baselines.loggers.data_logger import DataLogger, LastDataLog
from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.loggers.data_logger import ListDataLog
from modular_baselines.loggers.writers import ScalarWriter, DictWriter, BaseWriter, SaveModelParametersWriter, LogConfigs
from modular_baselines.utils.utils import to_torch, flatten_time, get_spaces


@dataclass(frozen=True)
class PPOArgs():
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    epochs: int
    lr: Coefficient
    clip_value: Coefficient
    batch_size: int
    max_grad_norm: float
    normalize_advantage: bool
    use_vec_normalization: bool
    vec_norm_info: Dict[str, Union[float, bool, int, str]]


class PPOAgent():

    def __init__(self,
                 network: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 observation_space: Space,
                 action_space: Space,
                 logger: DataLogger) -> None:

        self.network = network
        self.optimizer = optimizer
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logger

    def sample_action(self,
                      observation: np.ndarray,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            th_observation = to_torch(
                self.device, observation).float()

            policy_params, _ = self.network(th_observation)
            policy_dist = self.network.dist(policy_params)
            th_action = policy_dist.sample()
            log_prob = policy_dist.log_prob(th_action).unsqueeze(-1)
            if isinstance(self.action_space, Discrete):
                # th_action = th_action.unsqueeze(-1)
                raise NotImplementedError(
                    f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
        return th_action.cpu().numpy(), {"old_log_prob": log_prob.cpu().numpy()}

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.network.state_dict(),
            "policy_optimizer_state_dict": self.optimizer.state_dict(),
        },
            path)
        
    @property
    def device(self) -> "Device":
        return next(iter(self.network.parameters())).device


class PPO():

    def __init__(self,
                 agent: PPOAgent,
                 collector: RolloutCollector,
                 args: PPOArgs,
                 logger: DataLogger,
                 writers: List[BaseWriter]
                 ):

        self.agent = agent
        self.collector = collector
        self.logger = logger
        self.writers = writers

        self._init_default_loggers()

        self.buffer = self.collector.buffer
        self.num_envs = self.collector.env.num_envs

        self.args = args

    def learn(self, total_timesteps: int) -> None:
        """ Main loop for running the on-policy algorithm

        Args:
            total_timesteps (int): Total environment time steps to run
        """

        train_start_time = time()
        num_timesteps = 0
        iteration = 0

        for writer in self.writers:
            writer.on_training_start(locals())

        while num_timesteps < total_timesteps:
            iteration_start_time = time()

            num_timesteps = self.collector.collect(self.args.rollout_len)
            self.update()
            iteration += 1

            getattr(self.logger, "scalar/algorithm/iteration").push(iteration)
            getattr(self.logger, "scalar/algorithm/timesteps").push(num_timesteps)
            getattr(
                self.logger, "scalar/algorithm/time_elapsed").push(time() - train_start_time)
            getattr(self.logger, "scalar/algorithm/fps").push((time() -
                                                               iteration_start_time) / (self.num_envs * self.args.rollout_len))
        
            for writer in self.writers:
                writer.on_step(locals())

        for writer in self.writers:
            writer.on_training_end(locals())

        return None

    def update(self) -> Dict[str, float]:
        """ One step parameter update. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.agent.network.train(True)
        sample = self.buffer.sample(batch_size=self.num_envs,
                                    rollout_len=self.args.rollout_len,
                                    sampling_length=self.args.rollout_len)
        lr = next(self.args.lr)
        clip_value = next(self.args.clip_value)

        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = lr
        rollout_data = self.prepare_rollout(sample)

        for epoch in range(self.args.epochs):
            for advantages, returns, action, old_log_prob, *replay_data in self.rollout_loader(*rollout_data):

                if self.args.normalize_advantage:
                    advantages = (advantages - advantages.mean()
                                  ) / (advantages.std() + 1e-8)

                policy_params, values = self.replay_rollout(*replay_data)
                policy_dist = self.agent.network.dist(policy_params)
                if isinstance(self.agent.action_space, Discrete):
                    # action = action.squeeze(-1)
                    raise NotImplementedError(
                        f"Unsupported action space distribution {self.agent.action_space.__class__.__name__}!")
                log_probs = policy_dist.log_prob(action).unsqueeze(-1)
                entropies = policy_dist.entropy().unsqueeze(-1)

                value_loss = torch.nn.functional.mse_loss(values, returns)

                ratio = torch.exp(log_probs - old_log_prob.detach())
                surrogate_loss_1 = advantages * ratio
                surrogate_loss_2 = advantages * \
                    torch.clamp(ratio, 1 - clip_value, 1 + clip_value)
                policy_loss = - \
                    torch.minimum(surrogate_loss_1, surrogate_loss_2).mean()
                entropy_loss = -entropies.mean()
                loss = value_loss * self.args.value_coef + \
                    policy_loss + entropy_loss * self.args.ent_coef

                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.network.parameters(), self.args.max_grad_norm)
                self.agent.optimizer.step()

                getattr(self.logger,
                        "scalar/agent/value_loss").push(value_loss.item())
                getattr(self.logger,
                        "scalar/agent/policy_loss").push(policy_loss.item())
                getattr(self.logger,
                        "scalar/agent/entropy_loss").push(entropy_loss.item())
                getattr(
                    self.logger, "scalar/agent/approxkl").push(((ratio - 1) - ratio.log()).mean().item())
        getattr(self.logger, "scalar/agent/clip_range").push(clip_value)
        getattr(self.logger, "scalar/agent/learning_rate").push(lr)

    def rollout_to_torch(self, sample: np.ndarray) -> Tuple[torch.Tensor]:
        th_obs, th_action, th_next_obs, th_old_log_prob = to_torch(self.agent.device, [
            sample["observation"],
            sample["action"],
            sample["next_observation"][:, -1],
            sample["old_log_prob"]])
        return th_obs, th_action, th_next_obs, th_old_log_prob

    def prepare_rollout(self, sample: np.ndarray) -> List[torch.Tensor]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_next_obs, th_old_log_prob = self.rollout_to_torch(
            sample)

        _, th_flatten_values = self.agent.network(flatten_time(th_obs))
        th_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, th_next_value = self.agent.network(th_next_obs)

        advantages, returns = to_torch(self.agent.device, calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=th_values.detach().cpu().numpy(),
            last_value=th_next_value.detach().cpu().numpy(),
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda)
        )

        return (advantages, returns, th_action, th_old_log_prob, th_obs)

    def rollout_loader(self, *tensors: Tuple[torch.Tensor]
                       ) -> Generator[Tuple[torch.Tensor], None, None]:
        if len(tensors) == 0:
            raise ValueError("Empty tensors")
        flatten_tensors = [flatten_time(tensor) for tensor in tensors]
        perm_indices = torch.randperm(flatten_tensors[0].shape[0])

        for indices in perm_indices.split(self.args.batch_size):
            yield tuple([tensor[indices] for tensor in flatten_tensors])

    def replay_rollout(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_params, values = self.agent.network(obs)
        return policy_params, values

    def _init_default_loggers(self) -> None:
        loggers = {
            "scalar/agent/value_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/policy_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/entropy_loss": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/agent/learning_rate": ListDataLog(reduce_fn=lambda values: np.max(values)),
            "scalar/agent/clip_range": ListDataLog(reduce_fn=lambda values: np.max(values)),
            "scalar/agent/approxkl": ListDataLog(reduce_fn=lambda values: np.mean(values)),
            "scalar/algorithm/iteration": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/timesteps": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/time_elapsed": LastDataLog(reduce_fn=lambda value: value),
            "scalar/algorithm/fps": ListDataLog(reduce_fn=lambda values: int(1 / np.mean(values))),
        }
        self.logger.add_if_not_exists(loggers)

    @staticmethod
    def setup(env: VecEnv,
              agent: BaseAgent,
              data_logger: DataLogger,
              args: PPOArgs,
              writers: List[BaseWriter],
              ) -> "PPO":
        observation_space, action_space, action_dim = get_spaces(env)

        normalizer_struct = []
        if args.use_vec_normalization:
            normalizer_struct = [
                ("reward_rms_var", np.float32, (1,)),
                ("obs_rms_mean", np.float32, observation_space.shape),
                ("obs_rms_var", np.float32, observation_space.shape),
                ("next_obs_rms_mean", np.float32, observation_space.shape),
                ("next_obs_rms_var", np.float32, observation_space.shape),
            ]
        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
            *normalizer_struct
        ])
        buffer = Buffer(struct, args.rollout_len, env.num_envs,
                        data_logger, None)
        collector = RolloutCollector(
            env=env,
            buffer=buffer,
            agent=agent,
            logger=data_logger,
            store_normalizer_stats=args.use_vec_normalization,
            callbacks=None
        )
        return PPO(
            agent=agent,
            collector=collector,
            args=args,
            logger=data_logger,
            writers=writers
        )
