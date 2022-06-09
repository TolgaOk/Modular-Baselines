from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
from torch.types import Device
import numpy as np
from gym.spaces import Space

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import Logger, configure

from modular_baselines.algorithms.ppo.torch_policy import TorchPPOPolicy
from modular_baselines.algorithms.ppo.ppo import PPO
from modular_baselines.loggers.basic import InitLogCallback, LogRolloutCallback, LogLossCallback
from modular_baselines.utils.annealings import Coefficient, LinearAnnealing


class Policy(torch.nn.Module, TorchPPOPolicy):

    def __init__(self, observation_space: Space, action_space: Space, hidden: int = 128, lr: float = 1e-4):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.in_size = observation_space.shape[0]
        self.out_size = action_space.shape[0]

        self.lr = lr
        self.hidden_size = hidden
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.RReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.RReLU(),
        )

        self.policy_head = torch.nn.Linear(hidden, self.out_size * 2)

        torch.nn.init.xavier_normal_(self.policy_head.weight, 0.01)
        self.value_head = torch.nn.Linear(hidden, 1)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor, *args
                ) -> Tuple[torch.distributions.Normal, Optional[torch.Tensor], torch.Tensor]:
        """ Return policy distribution and value
        Args:
            state (torch.Tensor): State tensor
        Returns:
            Tuple[torch.distributions.Normal, torch.Tensor]: Normal 
                policy distribution and value
        """
        feature = self.feature(state)
        logits = self.policy_head(feature)
        value = self.value_head(feature)

        mean, std_logit = logits.split(self.out_size, dim=1)
        std = torch.nn.functional.softplus(std_logit)
        dist = torch.distributions.Normal(loc=mean, scale=std + 0.1)
        dist = torch.distributions.independent.Independent(dist, 1)

        return dist, None, value

    @property
    def device(self) -> Device:
        return next(iter(self.parameters())).device

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer


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

    agent = PPO.setup(
        env=vecenv,
        policy=policy,
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
        collector_callbacks=LogRolloutCallback(logger),
        algorithm_callbacks=[InitLogCallback(logger,
                                             hyperparameters["log_interval"]),
                             LogLossCallback(logger)])

    agent.learn(total_timesteps=hyperparameters["total_timesteps"])
    return agent


ppo_mujoco_walker2d_hyperparameters = dict(
    lr=1e-4,
    n_envs=16,
    n_steps=2048,
    ent_coef=1e-4,
    vf_coef=0.5,
    gamma=0.995,
    gae_lambda=0.90,
    epochs=5,
    clip_value=LinearAnnealing(0.2, 0.01, int(3_000_000 / 2048 / 16)),
    batch_size=64,
    max_grad_norm=1.0,
    total_timesteps=3_000_000,
    log_interval=1,
    device="cpu",
)


if __name__ == "__main__":
    agent = setup("Walker2d-v4", ppo_mujoco_walker2d_hyperparameters, seed=1003)
