from typing import List, Any, Dict, Union, Optional, Tuple, Callable, Type, Iterable
import torch
import os
import numpy as np
import sys
from multiprocessing import Process, Queue
from dataclasses import dataclass
import argparse
import gym
from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.logger import HumanOutputFormat, CSVOutputFormat, JSONOutputFormat
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from modular_baselines.algorithms.algorithm import BaseAlgorithm
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.writers import ScalarWriter, DictWriter, BaseWriter, SaveModelParametersWriter, LogConfigs
from modular_baselines.loggers.data_logger import DataLogger


@dataclass(frozen=True)
class MujocoTorchConfig():
    args: Any
    name: str
    n_envs: int
    total_timesteps: int
    log_interval: int
    record_video: bool
    use_vec_normalization: bool
    seed: int


def pre_setup(experiment_name: str,
              env: Union[gym.Env, str],
              config: MujocoTorchConfig,
              ) -> Tuple[DataLogger, List[BaseWriter], VecEnv]:
    """ Prepare loggers and vectorized environment

    Args:
        experiment_name (str): Name of the experiment
        env (Union[gym.Env, str]): Name of the environment or the environment
        config (MujocoTorchConfig): Torch Mujoco configuration

    Returns:
        Tuple[DataLogger, List[BaseWriter], VecEnv]: Data logger, writers list and vectorized 
            environment
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env_name = env if isinstance(env, str) else env.__class__.__name__
    date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    log_dir = f"logs/{experiment_name}-{env_name.lower()}/{config.name}/{date_time}"
    data_logger = DataLogger()
    os.makedirs(log_dir, exist_ok=True)
    sb3_writers = [HumanOutputFormat(sys.stdout),
                   CSVOutputFormat(os.path.join(log_dir, "progress.csv")),
                   JSONOutputFormat(os.path.join(log_dir, "progress.json"))]
    logger_callbacks = [
        ScalarWriter(interval=config.log_interval, dir_path=log_dir, writers=sb3_writers),
        DictWriter(interval=config.log_interval, dir_path=log_dir),
        SaveModelParametersWriter(interval=config.log_interval * 5, dir_path=log_dir)
    ]

    vecenv = make_vec_env(
        env,
        n_envs=config.n_envs,
        seed=config.seed,
        wrapper_class=None,
        vec_env_cls=SubprocVecEnv)
    if config.use_vec_normalization:
        vecenv = VecNormalize(vecenv, training=True, gamma=config.args.gamma)
    if config.record_video:
        vecenv = VecVideoRecorder(
            vecenv,
            f"{log_dir}/videos",
            record_video_trigger=lambda x: x % 25000 == 0, video_length=1000
        )
    LogConfigs(config=config, dir_path=log_dir)

    return data_logger, logger_callbacks, vecenv


def setup(algorithm_cls: Type[BaseAlgorithm],
          agent_cls: Type[BaseAgent],
          network: Type[torch.nn.Module],
          experiment_name: str,
          env_name: str,
          config: MujocoTorchConfig,
          device: str
          ) -> BaseAlgorithm:

    experiment_name = "-".join([experiment_name, algorithm_cls.__name__])
    data_logger, logger_callbacks, vecenv = pre_setup(experiment_name, env_name, config)

    policy = network(observation_space=vecenv.observation_space,
                     action_space=vecenv.action_space)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), eps=1e-5)
    agent = agent_cls(policy,
                      optimizer,
                      vecenv.observation_space,
                      vecenv.action_space,
                      data_logger)

    learner = algorithm_cls.setup(
        env=vecenv,
        agent=agent,
        data_logger=data_logger,
        args=config.args,
        buffer_callbacks=None,
        collector_callbacks=None,
        algorithm_callbacks=logger_callbacks)

    learner.learn(total_timesteps=config.total_timesteps)
    return learner


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-name", type=str, default="",
                        help="Prefix of the experiment name")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of parallelized processes for experiments")
    parser.add_argument("--env-names", nargs='+', type=str, required=True,
                        help="Gym environment names")
    parser.add_argument("--cuda-devices", nargs='+', type=int, required=False,
                        help="Available cuda devices")


def worker(setup_fn, argument_queue: Queue, rank: int, cuda_devices) -> None:
    device = "cpu" if cuda_devices is None else f"cuda:{cuda_devices[rank % len(cuda_devices)]}"
    print(f"Worker-{rank} use device: {device}")
    while not argument_queue.empty():
        kwargs = argument_queue.get()
        setup_fn(device=device, **kwargs)


def parallel_run(setup_fn: Callable[[str, MujocoTorchConfig, int], BaseAlgorithm],
                 configs: Union[MujocoTorchConfig, Iterable[MujocoTorchConfig]],
                 experiment_name: str,
                 n_procs: int,
                 env_names: Tuple[str],
                 cuda_devices: Tuple[int],
                 ) -> None:

    if not isinstance(configs, Iterable):
        configs = [configs]

    arguments = [dict(env_name=env_name, config=config, experiment_name=experiment_name)
                for env_name in env_names
                for config in configs]

    argument_queue = Queue()
    for arg in arguments:
        argument_queue.put(arg)

    processes = [Process(target=worker, args=(setup_fn, argument_queue, rank, cuda_devices))
                 for rank in range(n_procs)]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()
