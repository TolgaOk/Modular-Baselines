from typing import Any, Dict
from argparse import ArgumentParser
import multiprocessing as mp
import numpy as np

from modular_baselines.algorithms.ppo.ppo import LstmPPO, LstmPPOArgs
from modular_baselines.algorithms.ppo.torch_lstm_agent import TorchLstmPPOAgent
from modular_baselines.networks.network import LSTMSeparateNetwork
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing

from torch_setup import MujocoTorchConfig, setup, parallel_run, add_arguments


def lstm_ppo_setup(env_name: str, config: Dict[str, Any], experiment_name: str, device: str):
    return setup(LstmPPO, TorchLstmPPOAgent, LSTMSeparateNetwork, experiment_name, env_name, config, device)


lstm_ppo_mujoco_config = MujocoTorchConfig(
    args=LstmPPOArgs(
        rollout_len=2048,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=10,
        lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (2048 * 16)),
        clip_value=LinearAnnealing(0.2, 0.2, 5_000_000 // (2048 * 16)),
        batch_size=64 // 16,
        max_grad_norm=1.0,
        normalize_advantage=True,
        mini_rollout_size=16,
        use_sampled_hidden=False,
        use_vec_normalization=True,
        vec_norm_info={
            "norm_reward": True,
            "norm_obs": True,
            "clip_obs": 1e5,
            "clip_reward": 1e5,
        },
    ),
    name=f"default_{16}_step",
    n_envs=16,
    total_timesteps=5_000_000,
    log_interval=1,
    record_video=False,
    seed=np.random.randint(2**10, 2**30))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = ArgumentParser("PPO Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(lstm_ppo_setup,
                 lstm_ppo_mujoco_config,
                 n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names,
                 experiment_name=cli_args.experiment_name,
                 cuda_devices=cli_args.cuda_devices)
