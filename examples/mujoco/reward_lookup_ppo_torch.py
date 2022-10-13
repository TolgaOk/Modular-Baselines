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
from unitgym.reward_lookup import DelayedRewardLookUp

post_seed = np.random.randint(2**30)
np.random.seed(500)
random_rewards = np.random.uniform(-1, 1, 50)
np.random.seed(post_seed)

known_envs = {
    "10-8": lambda: DelayedRewardLookUp(10, 8, np.zeros(1) + 0.5),
    "20-18": lambda: DelayedRewardLookUp(20, 18, np.zeros(1) + 0.5),
    "5-3": lambda: DelayedRewardLookUp(5, 3, np.zeros(1) + 0.5),
    "10-3": lambda: DelayedRewardLookUp(10, 3, random_rewards[:6]),
    "10-5": lambda: DelayedRewardLookUp(10, 5, random_rewards[:4]),
    "20-3": lambda: DelayedRewardLookUp(20, 3, random_rewards[:16]),
    "20-10": lambda: DelayedRewardLookUp(20, 10, random_rewards[:9]),
    "50-25": lambda: DelayedRewardLookUp(50, 25, random_rewards[:24]),

}


def value_gradient_ppo_setup(experiment_name: str, env_name: Union[gym.Env, str], config: MujocoTorchConfig, device: str):

    if env_name not in known_envs.keys():
        raise ValueError(f"Unknown environment name: {env_name}")
    env_fn = known_envs[env_name]
    print(f"Min reward: {env_fn().min_reward}")
    env_reward_fn = env_fn().reward_fn
    def reward_fn(obs, act, next_obs): return env_reward_fn(obs, act)

    data_logger, logger_callbacks, vecenv = pre_setup(experiment_name, env_fn, config)
    vecenv.reward_fn = reward_fn

    policy = SeparateFeatureNetwork(
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space)
    policy.to(device)
    # model = ModelNetwork(state_size=vecenv.observation_space.shape[0],
    #                      action_size=vecenv.action_space.shape[0])
    model = StiefelNetwork(state_size=vecenv.observation_space.shape[0],
                           action_size=vecenv.action_space.shape[0])
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
    return learner


def make_config(env_length: int) -> MujocoTorchConfig:

    rollout_len = env_length
    n_envs = 500 // env_length
    total_timestep = 3_000_000

    return MujocoTorchConfig(
        args=ValueGradientPPOArgs(
            rollout_len=rollout_len,
            mini_rollout_size=rollout_len,
            ent_coef=1e-4,
            value_coef=0.5,
            gamma=0.99,
            gae_lambda=1.0,
            policy_epochs=1,
            model_epochs=25,
            max_grad_norm=1.0,
            buffer_size=2048 * 256,
            normalize_advantage=True,
            clip_value=LinearAnnealing(0.2, 0.2, total_timestep // (rollout_len * n_envs)),
            policy_batch_size=n_envs * rollout_len,
            model_batch_size=64,
            policy_lr=LinearAnnealing(3e-4, 0.0, total_timestep // (rollout_len * n_envs)),
            model_lr=LinearAnnealing(3e-4, 0.0, total_timestep // (rollout_len * n_envs)),
            check_reparam_consistency=False,
            use_log_likelihood=False,
            use_reparameterization=True,
            policy_loss_beta=LinearAnnealing(1.0, 0.1, 200_000 // (rollout_len * n_envs)),
            use_vec_normalization=False,
        ),
        name="ppo",
        n_envs=n_envs,
        total_timesteps=total_timestep,
        log_interval=100,
        record_video=False,
        seed=np.random.randint(2**10, 2**30),
    )


if __name__ == "__main__":
    parser = ArgumentParser("Value Gradient PPO Delayed Reward")
    add_arguments(parser)
    cli_args = parser.parse_args()
    if len(cli_args.env_names) != 1:
        raise ValueError("Only one environment is supported!")
    env_name = cli_args.env_names[0]
    env_length = int(env_name.split("-")[0])

    value_gradient_ppo_delayed_reward_config = make_config(env_length)
    parallel_run(value_gradient_ppo_setup,
                 value_gradient_ppo_delayed_reward_config,
                 n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names,
                 experiment_name=cli_args.experiment_name,
                 cuda_devices=cli_args.cuda_devices)
