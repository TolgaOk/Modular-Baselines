from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
from torch.types import Device
import numpy as np
from gym.spaces import Space

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure

from modular_baselines.algorithms.a2c.torch_policy import TorchA2CPolicy
from modular_baselines.networks.network import SharedFeatureNetwork
from modular_baselines.algorithms.a2c.a2c import A2C
from modular_baselines.loggers.basic import InitLogCallback, LogRolloutCallback, LogLossCallback


class Policy(SharedFeatureNetwork, TorchA2CPolicy):

    @property
    def device(self) -> Device:
        return next(iter(self.parameters())).device

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def init_state(self, batch_size=None):
        return None


def setup(env_name: str, hyperparameters: Dict[str, Any], seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = f"logs/{env_name}-{seed}"
    logger = configure(log_dir, ["stdout", "json", "csv"])

    vecenv = make_vec_env(
        env_name,
        n_envs=hyperparameters["n_envs"],
        seed=seed,
        wrapper_class=None,
        vec_env_cls=SubprocVecEnv)
    policy = Policy(observation_space=vecenv.observation_space,
                    action_space=vecenv.action_space,
                    lr=hyperparameters["lr"])
    policy.to(hyperparameters["device"])

    agent = A2C.setup(
        env=vecenv,
        policy=policy,
        rollout_len=hyperparameters["n_steps"],
        ent_coef=hyperparameters["ent_coef"],
        value_coef=hyperparameters["vf_coef"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        buffer_callbacks=None,
        collector_callbacks=LogRolloutCallback(logger),
        algorithm_callbacks=[InitLogCallback(logger,
                                             hyperparameters["log_interval"]),
                             LogLossCallback(logger)])

    agent.learn(total_timesteps=hyperparameters["total_timesteps"])
    return agent


a2c_mujoco_walker2d_hyperparameters = dict(
    lr=0.00025,
    n_envs=16,
    n_steps=8,
    ent_coef=1e-4,
    vf_coef=1.0,
    gamma=0.999,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    total_timesteps=1_000_000,
    log_interval=250,
    device="cpu",
)


if __name__ == "__main__":
    agent = setup("Walker2d-v4", a2c_mujoco_walker2d_hyperparameters, seed=1004)
