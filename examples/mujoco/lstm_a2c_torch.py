from typing import Any, Dict
from argparse import ArgumentParser

from modular_baselines.algorithms.a2c import LstmA2C, A2CArgs
from modular_baselines.algorithms.a2c.torch_lstm_agent import TorchLstmA2CAgent
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing
from modular_baselines.networks.network import LSTMSeparateNetwork

from torch_setup import MujocoTorchConfig, setup, parallel_run, add_arguments


def a2c_setup(env_name: str, config: Dict[str, Any], seed: int):
    return setup(LstmA2C, TorchLstmA2CAgent, LSTMSeparateNetwork, env_name, config, seed)


lstm_a2c_mujoco_config = MujocoTorchConfig(
    args=A2CArgs(
        rollout_len=8,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (8 * 16)),
        max_grad_norm=1.0,
        normalize_advantage=True,
        use_vec_normalization=True,
        vec_norm_info={
            "norm_reward": True,
            "norm_obs": True,
            "clip_obs": 1e5,
            "clip_reward": 1e5,
        },
    ),
    n_envs=16,
    total_timesteps=5_000_000,
    log_interval=256,
    record_video=False,
    seed=np.random.randint(2**10, 2**30),
)

if __name__ == "__main__":
    parser = ArgumentParser("LSTM A2C Mujoco")
    add_arguments(parser)
    cli_args = parser.parse_args()
    parallel_run(a2c_setup,
                 lstm_a2c_mujoco_config,
                 n_procs=cli_args.n_procs,
                 env_names=cli_args.env_names,
                 experiment_name=cli_args.experiment_name,
                 cuda_devices=cli_args.cuda_devices)
