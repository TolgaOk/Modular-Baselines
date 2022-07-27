from typing import Any, Dict
from argparse import ArgumentParser

from modular_baselines.algorithms.ppo.ppo import LstmPPO, LstmPPOArgs
from modular_baselines.algorithms.ppo.torch_lstm_agent import TorchLstmPPOAgent
from modular_baselines.networks.network import LSTMSeparateNetwork
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing

from torch_setup import MujocoTorchConfig, setup, parallel_run, add_arguments


def ppo_setup(env_name: str, config: Dict[str, Any], seed: int):
    return setup(LstmPPO, TorchLstmPPOAgent, LSTMSeparateNetwork, env_name, config, seed)


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
        batch_size=8,
        max_grad_norm=1.0,
        normalize_advantage=True,
        mini_rollout_size=8,
        use_sampled_hidden=False,
    ),
    n_envs=16,
    total_timesteps=5_000_000,
    log_interval=1,
    device="cpu",
)

if __name__ == "__main__":
    parser = ArgumentParser("PPO Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(ppo_setup, lstm_ppo_mujoco_config, n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names, n_seeds=cli_args.n_seeds)
