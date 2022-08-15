from typing import Any, Dict, Union
from argparse import ArgumentParser
import gym
from gym.envs.mujoco.hopper_v4 import HopperEnv
import torch

from modular_baselines.algorithms.ppo.model_based_ppo import ValueGradientPPOArgs, ValueGradientPPO
from modular_baselines.algorithms.ppo.torch_model_based_agent import TorchValueGradientAgent
from modular_baselines.networks.network import SeparateFeatureNetwork
from modular_baselines.networks.model import ModelNetwork
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing

from torch_setup import MujocoTorchConfig, pre_setup, parallel_run, add_arguments


class HopperMaker():

    @staticmethod
    def reward(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        forward_reward = (next_state[..., 0] - state[..., 0]) / 0.008 + 1
        ctrl_reward = 1e-3 * torch.sum(torch.square(action), dim=-1)
        return (forward_reward - ctrl_reward).unsqueeze(-1)

    @staticmethod
    def make():
        env = HopperEnv(exclude_current_positions_from_observation=False)
        return env


known_envs = {
    "Hopper-v4": HopperMaker()
}


def value_gradient_ppo_setup(experiment_name: str, env_name: Union[gym.Env, str], config: MujocoTorchConfig, seed: int, device: str):
    if env_name not in known_envs.keys():
        raise ValueError("Unknown environment name")
    env_maker = known_envs[env_name]
    env_fn, reward_fn = env_maker.make, env_maker.reward
    data_logger, logger_callbacks, vecenv = pre_setup(experiment_name, env_fn, config, seed, use_vec_normalizer=True)
    vecenv.reward_fn = reward_fn

    policy = SeparateFeatureNetwork(
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space)
    policy.to(device)
    model = ModelNetwork(state_size=vecenv.observation_space.shape[0],
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



value_gradient_ppo_mujoco_config = MujocoTorchConfig(
    args=ValueGradientPPOArgs(
        rollout_len=5,
        mini_rollout_size=5,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        policy_epochs=1,
        model_epochs=2,
        max_grad_norm=1.0,
        buffer_size=2048 * 256,
        normalize_advantage=True,
        clip_value=LinearAnnealing(0.2, 0.2, 5_000_000 // (5 * 16)),
        policy_batch_size=16,
        model_batch_size=64,
        policy_lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (5 * 16)),
        model_lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (5 * 16)),
        check_reward_consistency=False,
        use_log_likelihood=False,
        use_reparameterization=True,
        policy_loss_beta=LinearAnnealing(1.0, 0.0, 2_000_000 // (5 * 16)),
    ),
    name="a2c-like",
    n_envs=16,
    total_timesteps=5_000_000,
    log_interval=256,
)

if __name__ == "__main__":
    parser = ArgumentParser("Value Gradient PPO Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(value_gradient_ppo_setup,
                 value_gradient_ppo_mujoco_config,
                 n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names,
                 n_seeds=cli_args.n_seeds,
                 experiment_name=cli_args.experiment_name,
                 cuda_devices=cli_args.cuda_devices)

    # value_gradient_ppo_setup(
    #     experiment_name=cli_args.experiment_name,
    #     env_name=cli_args.env_names[0],
    #     config=value_gradient_ppo_mujoco_config,
    #     seed=1073664207,
    #     device="cpu"
    # )
