from typing import Any, Dict, Union
from argparse import ArgumentParser
import gym
import os
import numpy as np
import torch
from gym.spaces import Box

from modular_baselines.algorithms.ppo.model_based_ppo import ValueGradientPPOArgs, ValueGradientPPO
from modular_baselines.algorithms.ppo.torch_model_based_agent import TorchValueGradientAgent
from modular_baselines.networks.network import SeparateFeatureNetwork
from modular_baselines.networks.model import ModelNetwork, StiefelNetwork
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing

from torch_setup import MujocoTorchConfig, pre_setup, parallel_run, add_arguments


class WalkerObsNormalizer(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low/10,
            high=env.observation_space.high/10)

    def observation(self, observation):
        return observation / 10


class MujocoMaker():
    @staticmethod
    def reward(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        forward_reward = (next_state[..., 0] - state[..., 0]) / 0.008 + 1
        ctrl_reward = 1e-3 * torch.sum(torch.square(action), dim=-1)
        return (forward_reward - ctrl_reward).unsqueeze(-1)

    @staticmethod
    def policy_class(*args, **kwargs):
        return SeparateFeatureNetwork(*args, **kwargs)


class InvertedPendulumMaker():

    @staticmethod
    def make():
        env = gym.make("InvertedPendulum-v4")
        return env

    @staticmethod
    def reward(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The original reward function contains only constant")

    @staticmethod
    def policy_class(*args, **kwargs):
        return SeparateFeatureNetwork(*args, **kwargs)


class HopperMaker(MujocoMaker):
    @staticmethod
    def make():
        env = gym.make("Hopper-v4", exclude_current_positions_from_observation=False)
        return env

    @staticmethod
    def policy_class(*args, **kwargs):
        return MujocoSeparateFeatureNetwork(*args, **kwargs)


class Walker2dMaker(MujocoMaker):
    # With div 10 constant normalizer
    @staticmethod
    def make():
        env = gym.make("Walker2d-v4", exclude_current_positions_from_observation=False)
        return WalkerObsNormalizer(env)

    @staticmethod
    def reward(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return MujocoMaker.reward(state * 10, action, next_state * 10)

    @staticmethod
    def policy_class(*args, **kwargs):
        return MujocoSeparateFeatureNetwork(*args, **kwargs)


known_envs = {
    "Hopper-v4": HopperMaker(),
    "Walker2d-v4": Walker2dMaker(),
    "InvertedPendulum-v4": InvertedPendulumMaker(),
}


class MujocoSeparateFeatureNetwork(SeparateFeatureNetwork):

    def __init__(self, observation_space: Box, *args, **kwargs):
        observation_space = Box(
            low=observation_space.low[1:],
            high=observation_space.high[1:]
        )
        super().__init__(observation_space, *args, **kwargs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs[..., 1:]
        return super().forward(obs)


def value_gradient_ppo_setup(experiment_name: str, env_name: Union[gym.Env, str], config: MujocoTorchConfig, device: str):
    if env_name not in known_envs.keys():
        raise ValueError(f"Unknown environment name: {env_name}")
    env_maker = known_envs[env_name]
    env_fn, reward_fn = env_maker.make, env_maker.reward
    data_logger, logger_callbacks, vecenv = pre_setup(experiment_name, env_fn, config)
    vecenv.reward_fn = reward_fn

    policy = env_maker.policy_class(
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space)
    policy.to(device)
    model = StiefelNetwork(state_size=vecenv.observation_space.shape[0],
                           action_size=vecenv.action_space.shape[0],
                           #    hidden_size=128,
                           #    n_layers=2
                           )
    model.to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), eps=1e-5)
    model_optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)

    agent = TorchValueGradientAgent(
        policy=policy,
        model=model,
        policy_optimizer=policy_optimizer,
        model_optimizer=model_optimizer,
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space,
        logger=data_logger
    )

    learner = ValueGradientPPO.setup(
        env=vecenv,
        agent=agent,
        data_logger=data_logger,
        args=config.args,
        algorithm_callbacks=logger_callbacks
    )

    learner.learn(total_timesteps=config.total_timesteps)
    # np.save(f"buffer_{config.total_timesteps}.npy", learner.buffer.buffer)
    return learner


rollout_len = 2048
n_envs = 16

value_gradient_ppo_mujoco_config = [MujocoTorchConfig(
    args=ValueGradientPPOArgs(
        rollout_len=rollout_len,
        mini_rollout_size=64,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        policy_epochs=10,
        model_epochs=100,
        max_grad_norm=1.0,
        buffer_size=2048 * 64,
        normalize_advantage=True,
        clip_value=LinearAnnealing(0.2, 0.2, 5_000_000 // (rollout_len * n_envs)),
        policy_batch_size=n_envs,
        model_batch_size=64,
        policy_lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (rollout_len * n_envs)),
        # model_lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (rollout_len * n_envs)),
        model_lr=Coefficient(1e-4),
        check_reparam_consistency=True,
        use_log_likelihood=False,
        use_reparameterization=True,
        policy_loss_beta=LinearAnnealing(1.0, 0.1, 5_000_000 // (rollout_len * n_envs)),
        use_vec_normalization=True,
        vec_norm_info={
            "norm_reward": True,
            "norm_obs": True,
            "clip_obs": 1e5,
            "clip_reward": 1e5,
        },
        pre_trained_model="../../model_data_1M.b",
    ),
    name="ppo",
    n_envs=n_envs,
    total_timesteps=5_000_000,
    log_interval=1,
    record_video=False,
    seed=np.random.randint(2**10, 2**30),  # 683083883
) for _ in range(1)]

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    parser = ArgumentParser("Value Gradient PPO Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(value_gradient_ppo_setup,
                 value_gradient_ppo_mujoco_config,
                 n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names,
                 experiment_name=cli_args.experiment_name,
                 cuda_devices=cli_args.cuda_devices)
