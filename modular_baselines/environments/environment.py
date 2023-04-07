from typing import Any, Protocol, Optional
from dataclasses import dataclass
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from modular_baselines.loggers.logger import MBLogger


n_envs = 16
env_name = "Walker2d-v4"
gamma = 0.99


class EnvArgs(Protocol):
    name: str
    gamma: float
    n_envs: int
    norm_reward: bool = False
    norm_obs: bool = False
    clip_obs: float = 1e5
    clip_reward: float = 1e5


@dataclass
class RunningStat():

    observation_mean: Optional[np.ndarray] = None
    observation_std: Optional[np.ndarray] = None
    return_mean: Optional[np.ndarray] = None
    return_std: Optional[np.ndarray] = None

    def update():
        pass
    def normalize():
        pass
    def inv_normalize():
        pass

class Env():

    def __init__(self,
                 env_name: str,
                 args: EnvArgs,
                 logger: MBLogger) -> None:
        self.env_name = env_name
        self.args = args
        self.logger = logger

        self.vecenv = make_vec_env(
            env_name,
            n_envs=n_envs,
            # seed=seed,
            vec_env_cls=SubprocVecEnv)
        self.vecenv = VecNormalize(
            self.vecenv,
            training=True,
            gamma=self.args.gamma,
            norm_reward=self.norm_reward,
            norm_obs=self.norm_obs,
            clip_obs=self.clip_obs,
            clip_reward=self.clip_reward,
        )

    def reset(): pass
    def step(): pass
