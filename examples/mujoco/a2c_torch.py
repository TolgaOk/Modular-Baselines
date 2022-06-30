from typing import Any, Dict
from argparse import ArgumentParser

from modular_baselines.algorithms.a2c import A2C, A2CArgs
from modular_baselines.algorithms.a2c.torch_agent import TorchA2CAgent
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing

from torch_setup import MujocoTorchConfig, setup, parallel_run, add_arguments


def a2c_setup(env_name: str, config: Dict[str, Any], seed: int):
    return setup(A2C, TorchA2CAgent, env_name, config, seed)


a2c_mujoco_config = MujocoTorchConfig(
    args=A2CArgs(
        rollout_len=8,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (8 * 16)),
        max_grad_norm=1.0,
        normalize_advantage=True,
    ),
    n_envs=16,
    total_timesteps=5_000_000,
    log_interval=256,
    device="cpu",
)

if __name__ == "__main__":
    parser = ArgumentParser("A2C Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(a2c_setup, a2c_mujoco_config, n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names, n_seeds=cli_args.n_seeds)
