""" Example Pong-ram training using A2C. """
import gym
import torch
import numpy as np
import argparse

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from modular_baselines.buffers.buffer import RolloutBuffer
from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.algorithms.a2c import A2C
from modular_baselines.loggers.basic import(InitLogCallback,
                                            LogRolloutCallback,
                                            LogWeightCallback,
                                            LogGradCallback)
from modular_baselines.utils.wrappers import (NormalizeObservation,
                                              SkipSteps,
                                              AggregateObservation,
                                              IndexObsevation,
                                              IndexAction,
                                              ResetWithNonZeroReward)


def wrap_env(envname="Pong-ramDeterministic-v4",
             state_ix=[51, 50, 49, 54],
             action_ix=[2, 3],
             aggr_ix=[2, 3],
             skip_initial_n_steps=16):

    env = gym.make(envname)
    env = IndexObsevation(env, state_ix)
    env = AggregateObservation(env, aggr_ix)
    env = SkipSteps(env, skip_initial_n_steps)
    env = NormalizeObservation(env)
    env = IndexAction(env, action_ix)
    env = ResetWithNonZeroReward(env)

    return env


def wrap_env():
    env = gym.make("LunarLander-v2")
    return env


class Policy(torch.nn.Module):

    def __init__(self, observation_space, action_space, hidden_size=16, lr=1e-3):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("Unsupported observation space {}".format(
                observation_space))
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Unsupported action space {}".format(
                observation_space))

        self.action_layers = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, action_space.n)
        )
        self.value_layers = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def _forward(self, x):
        act_logit = self.action_layers(x)
        values = self.value_layers(x)
        return act_logit, values

    def forward(self, x):
        act_logit, values = self._forward(x)

        dist = torch.distributions.categorical.Categorical(logits=act_logit)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, values, log_prob

    def evaluate_actions(self, observation, action):
        act_logit, values = self._forward(observation)
        dist = torch.distributions.categorical.Categorical(logits=act_logit)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return values, log_prob, entropy


def run_experiment(args):

    if args.seed is None:
        seed = np.random.randint(0, 2**16)
    else:
        seed = args.seed

    # Logger Callbacks
    rollout_callback = LogRolloutCallback()
    learn_callback = InitLogCallback(args.log_interval,
                                     args.log_dir)
    weight_callback = LogWeightCallback("weights.json")
    grad_callback = LogGradCallback("grads.json")

    # Environment
    vecenv = make_vec_env(wrap_env,
                          n_envs=args.n_envs,
                          seed=seed,
                          vec_env_cls=SubprocVecEnv)

    # Policy
    policy = Policy(vecenv.observation_space,
                    vecenv.action_space,
                    hidden_size=args.hiddensize,
                    lr=args.lr)

    # Modules
    buffer = RolloutBuffer(buffer_size=args.n_steps,
                           observation_space=vecenv.observation_space,
                           action_space=vecenv.action_space,
                           device=args.device,
                           gae_lambda=args.gae_lambda,
                           gamma=args.gamma,
                           n_envs=args.n_envs)
    collector = OnPolicyCollector(env=vecenv,
                                  buffer=buffer,
                                  policy=policy,
                                  callbacks=[rollout_callback],
                                  device=args.device)
    model = A2C(policy=policy,
                rollout_buffer=buffer,
                rollout_len=args.n_steps,
                collector=collector,
                env=vecenv,
                ent_coef=args.ent_coef,
                vf_coef=args.val_coef,
                max_grad_norm=args.max_grad_norm,
                normalize_advantage=False,
                callbacks=[learn_callback, weight_callback, grad_callback],
                device=args.device)

    # Start learning
    model.learn(args.total_timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong-Ram A2C")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--hiddensize", type=int, default=64,
                        help="Hidden size of the policy")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Rollout Length")
    parser.add_argument("--batchsize", type=int, default=32,
                        help="Batch size of the a2c training")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="GAE coefficient")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="Entropy coefficient")
    parser.add_argument("--val-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum allowed graident norm")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
                        help=("Training length interms of cumulative"
                              " environment timesteps"))
    parser.add_argument("--log-interval", type=int, default=500,
                        help=("Logging interval in terms of training"
                              " iterations"))
    parser.add_argument("--log-dir", type=str, default=None,
                        help=("Logging dir"))
    args = parser.parse_args()
    run_experiment(args)
