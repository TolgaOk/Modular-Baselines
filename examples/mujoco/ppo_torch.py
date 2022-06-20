from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
from torch.types import Device
import numpy as np
from gym.spaces import Space

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from modular_baselines.algorithms.ppo.ppo import PPO
from modular_baselines.algorithms.ppo.torch_agent import TorchPPOAgent
from modular_baselines.networks.network import SeparateFeatureNetwork
from modular_baselines.loggers.basic import STDOUTLoggerCallback
from modular_baselines.loggers.data_logger import DataLogger
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing


def setup(env_name: str, hyperparameters: Dict[str, Any], seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = f"logs/{env_name}-{seed}"

    data_logger = DataLogger()
    vecenv = make_vec_env(
        env_name,
        n_envs=hyperparameters["n_envs"],
        seed=seed,
        wrapper_class=None,
        vec_env_cls=SubprocVecEnv)
    policy = SeparateFeatureNetwork(observation_space=vecenv.observation_space,
                                    action_space=vecenv.action_space)
    policy.to(hyperparameters["device"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparameters["lr"])
    agent = TorchPPOAgent(policy, optimizer, vecenv.observation_space,
                          vecenv.action_space, data_logger)

    agent = PPO.setup(
        env=vecenv,
        agent=agent,
        data_logger=data_logger,
        rollout_len=hyperparameters["n_steps"],
        ent_coef=hyperparameters["ent_coef"],
        value_coef=hyperparameters["vf_coef"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        epochs=hyperparameters["epochs"],
        clip_value=hyperparameters["clip_value"],
        batch_size=hyperparameters["batch_size"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        buffer_callbacks=None,
        collector_callbacks=None,
        algorithm_callbacks=[STDOUTLoggerCallback(interval=hyperparameters["log_interval"])])

    agent.learn(total_timesteps=hyperparameters["total_timesteps"])
    return agent


ppo_mujoco_walker2d_hyperparameters = dict(
    lr=3e-4,
    n_envs=16,
    n_steps=2048,
    ent_coef=1e-4,
    vf_coef=0.5,
    gamma=0.99,
    gae_lambda=0.95,
    epochs=10,
    clip_value=LinearAnnealing(0.2, 0.2, int(3_000_000 / 2048 / 16)),
    batch_size=64,
    max_grad_norm=1.0,
    total_timesteps=5_000_000,
    log_interval=1,
    device="cpu",
)


if __name__ == "__main__":
    agent = setup("Swimmer-v4", ppo_mujoco_walker2d_hyperparameters, seed=1003)
