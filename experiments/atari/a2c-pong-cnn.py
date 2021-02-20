""" Example Pong-ram training using A2C. """
import gym
import torch
import numpy as np
import argparse
from collections import namedtuple

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from modular_baselines.buffers.buffer import RolloutBuffer
from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.algorithms.a2c import A2C
from modular_baselines.runners.multi_seed import MultiSeedRunner
from modular_baselines.loggers.basic import(InitLogCallback,
                                            LogRolloutCallback,
                                            LogWeightCallback,
                                            LogGradCallback,
                                            LogHyperparameters)


def make_env(n_envs: int,
             seed: int,
             n_stack: int,
             envname: str = "PongNoFrameskip-v4"):

    env = make_atari_env(env_id=envname,
                         n_envs=n_envs,
                         seed=seed,
                         vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)

    return env


class Policy(torch.nn.Module):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 hidden_size: int = 128,
                 lr=1e-3,
                 rms_prob_eps=1e-5):
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

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(observation_space.shape[0], 32,
                            kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        flatten_size = 7 * 7 * 64
        self.action_layers = torch.nn.Sequential(
            torch.nn.Linear(flatten_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space.n)
        )
        self.value_layers = torch.nn.Sequential(
            torch.nn.Linear(flatten_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(),
                                             lr=lr,
                                             eps=rms_prob_eps)

    def _forward(self, x):
        x = self._preprocess(x)
        features = self.cnn(x)

        act_logit = self.action_layers(features)
        values = self.value_layers(features)
        return act_logit, values

    def _preprocess(self, x):
        return x.float() / 255

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


def run(args):

    seed = args.seed
    if args.seed is None:
        seed = np.random.randint(0, 2**16)

    # Logger Callbacks
    hyper_callback = LogHyperparameters(args._asdict())
    rollout_callback = LogRolloutCallback()
    learn_callback = InitLogCallback(args.log_interval,
                                     args.log_dir)
    weight_callback = LogWeightCallback("weights.json")
    grad_callback = LogGradCallback("grads.json")

    # Environment
    vecenv = make_env(n_envs=args.n_envs,
                      seed=seed,
                      n_stack=args.n_stack)

    # Policy
    policy = Policy(vecenv.observation_space,
                    vecenv.action_space,
                    hidden_size=args.hiddensize,
                    lr=args.lr,
                    rms_prob_eps=args.rms_prop_eps)

    # Stable baseline_policy
    if args.sb3_policy:
        print("Using SB3 policy with {} initialization!".format(
            "ortho" if args.ortho_init else "default"))
        policy = ActorCriticCnnPolicy(
            vecenv.observation_space,
            vecenv.action_space,
            lambda x: args.lr,
            ortho_init=args.ortho_init)

    # Modules
    buffer = RolloutBuffer(buffer_size=args.n_steps,
                           observation_space=vecenv.observation_space,
                           action_space=vecenv.action_space,
                           device=args.device,
                           gae_lambda=args.gae_lambda,
                           gamma=args.gamma,
                           n_envs=args.n_envs)
    # Collector
    collector = OnPolicyCollector(env=vecenv,
                                  buffer=buffer,
                                  policy=policy,
                                  callbacks=[rollout_callback],
                                  device=args.device)
    # Model
    model = A2C(policy=policy,
                rollout_buffer=buffer,
                rollout_len=args.n_steps,
                collector=collector,
                env=vecenv,
                ent_coef=args.ent_coef,
                vf_coef=args.val_coef,
                max_grad_norm=args.max_grad_norm,
                normalize_advantage=False,
                callbacks=[learn_callback,
                           weight_callback,
                           grad_callback,
                           hyper_callback],
                device=args.device)

    # Start learning
    model.learn(args.total_timesteps)
    return model


class PongA2cMultiSeed(MultiSeedRunner):

    def single_run(self, args: namedtuple):
        return run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong-Ram A2C")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--hiddensize", type=int, default=128,
                        help="Hidden size of the policy")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Rollout Length")
    parser.add_argument("--n-stack", type=int, default=4,
                        help="Frame stack")
    parser.add_argument("--batchsize", type=int, default=32,
                        help="Batch size of the a2c training")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="GAE coefficient")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--val-coef", type=float, default=0.25,
                        help="Value loss coefficient")
    parser.add_argument("--rms-prop-eps", type=float, default=1e-5,
                        help="RmsProp epsion coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum allowed graident norm")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
                        help=("Training length interms of cumulative"
                              " environment timesteps"))
    parser.add_argument("--log-interval", type=int, default=1000,
                        help=("Logging interval in terms of training"
                              " iterations"))
    parser.add_argument("--log-dir", type=str, default=None,
                        help=("Logging dir"))
    parser.add_argument("--sb3-policy", action="store_true",
                        help="Use SB3 policy")
    parser.add_argument("--ortho_init", action="store_true",
                        help="Use orthogonal initialization in the policy")

    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallelized jobs for experiments")
    parser.add_argument("--runs-per-job", type=int, default=1,
                        help="Number of parallelized jobs for experiments")

    args = parser.parse_args()
    args = vars(args)

    runs_per_job = args.pop("runs_per_job")
    n_jobs = args.pop("n_jobs")

    PongA2cMultiSeed(args, runs_per_job=runs_per_job).run(n_jobs=n_jobs)
