from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import sys
import os
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.logger import HumanOutputFormat, CSVOutputFormat, JSONOutputFormat

from modular_baselines.algorithms.a2c.torch_agent import TorchA2CAgent
from modular_baselines.algorithms.a2c.a2c import A2C
from modular_baselines.networks.network import SharedFeatureNetwork
from modular_baselines.loggers.data_logger import DataLogger
from modular_baselines.loggers.basic import LogOutCallback
from modular_baselines.utils.annealings import LinearAnnealing


def setup(env_name: str, hyperparameters: Dict[str, Any], seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = f"logs/a2c-{env_name.lower()}/{seed}"
    data_logger = DataLogger()
    os.makedirs(log_dir, exist_ok=True)
    writers = [HumanOutputFormat(sys.stdout),
               CSVOutputFormat(os.path.join(log_dir, "progress.csv")),
               JSONOutputFormat(os.path.join(log_dir, "progress.json"))]
    logger_callback = LogOutCallback(
        interval=hyperparameters["log_interval"], dir_path=log_dir, writers=writers)

    vecenv = make_vec_env(
        env_name,
        n_envs=hyperparameters["n_envs"],
        seed=seed,
        wrapper_class=None,
        vec_env_cls=SubprocVecEnv)
    vecenv = VecNormalize(vecenv, training=True, gamma=hyperparameters["gamma"])

    policy = SharedFeatureNetwork(observation_space=vecenv.observation_space,
                                  action_space=vecenv.action_space)
    optimizer = torch.optim.Adam(policy.parameters())
    policy.to(hyperparameters["device"])

    agent = TorchA2CAgent(policy, optimizer, vecenv.observation_space,
                          vecenv.action_space, data_logger)
    learner = A2C.setup(
        env=vecenv,
        agent=agent,
        data_logger=data_logger,
        rollout_len=hyperparameters["n_steps"],
        ent_coef=hyperparameters["ent_coef"],
        value_coef=hyperparameters["vf_coef"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        lr=hyperparameters["lr"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        normalize_advantage=hyperparameters["normalize_advantage"],
        buffer_callbacks=None,
        collector_callbacks=None,
        algorithm_callbacks=logger_callback)

    learner.learn(total_timesteps=hyperparameters["total_timesteps"])
    return learner


a2c_mujoco_walker2d_hyperparameters = dict(
    n_envs=16,
    n_steps=8,
    ent_coef=1e-4,
    vf_coef=0.5,
    lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (8 * 16)),
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    normalize_advantage=True,
    total_timesteps=5_000_000,
    log_interval=256,
    device="cpu",
)


if __name__ == "__main__":
    agent = setup("Swimmer-v4", a2c_mujoco_walker2d_hyperparameters, seed=1006)
