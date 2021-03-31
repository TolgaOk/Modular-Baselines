""" Example Pong-ram training using A2C. """
import gym
import torch
import numpy as np
import argparse
from functools import partial
from collections import namedtuple

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from modular_baselines.buffers.buffer import GeneralBuffer
from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.runners.multi_seed import MultiSeedRunner
from modular_baselines.utils.score import log_score
from modular_baselines.contrib.model_based.latent_a2c import LatentTransitionPredA2C
from modular_baselines.contrib.model_based.vae import LatentTransitionPredictionVae
from modular_baselines.loggers.basic import(InitLogCallback,
                                            LogRolloutCallback,
                                            LogWeightCallback,
                                            LogGradCallback,
                                            LogHyperparameters)

from vae_a2c_pong import make_env
from vae_a2c_pong import Policy


class LatentTransitionPredPolicy(Policy):
    def _init_networks(self):
        self.vae = LatentTransitionPredictionVae(
            action_size=self.action_space.n,
            image_channels=self.observation_space.shape[0],
            z_dim=self.hidden_size)
        self.action_layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.action_space.n))
        self.value_layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 1))


def run(args):

    seed = args.seed
    if args.seed is None:
        seed = np.random.randint(0, 2**16)

    # Logger Callbacks
    hyper_callback = LogHyperparameters(args._asdict())
    rollout_callback = LogRolloutCallback()
    learn_callback = InitLogCallback(args.log_interval,
                                     args.log_dir)
    algorithm_callbacks = [learn_callback, hyper_callback]
    if args.log_histogram:
        weight_callback = LogWeightCallback("weights.json")
        grad_callback = LogGradCallback("grads.json")
        algorithm_callbacks += [weight_callback, grad_callback]

    # Environment
    vecenv = make_env(n_envs=args.n_envs,
                      seed=seed,
                      n_stack=args.n_stack)

    # Policy
    policy = LatentTransitionPredPolicy(vecenv.observation_space,
                                        vecenv.action_space,
                                        hidden_size=args.hiddensize,
                                        lr=args.lr,
                                        rms_prob_eps=args.rms_prop_eps,
                                        ortho_init=args.ortho_init)

    # Modules
    buffer = GeneralBuffer(buffer_size=max(args.n_steps + 1, args.buffer_size // args.n_envs),
                           observation_space=vecenv.observation_space,
                           action_space=vecenv.action_space,
                           device=args.device,
                           n_envs=args.n_envs)
    # Collector
    collector = OnPolicyCollector(env=vecenv,
                                  buffer=buffer,
                                  policy=policy,
                                  callbacks=[rollout_callback],
                                  device=args.device)
    # Model
    model = LatentTransitionPredA2C(policy=policy,
                                    rollout_buffer=buffer,
                                    rollout_len=args.n_steps,
                                    collector=collector,
                                    env=vecenv,
                                    ent_coef=args.ent_coef,
                                    vf_coef=args.val_coef,
                                    gae_lambda=args.gae_lambda,
                                    gamma=args.gamma,
                                    batch_size=args.batch_size,
                                    kl_beta_coef=args.kl_beta_coef,
                                    model_update_freq=args.model_update_freq,
                                    model_batch_size=args.model_batch_size,
                                    max_grad_norm=args.max_grad_norm,
                                    normalize_advantage=False,
                                    callbacks=algorithm_callbacks,
                                    device=args.device)

    # Start learning
    model.learn(args.total_timesteps)
    return log_score(args.log_dir)


class PongA2cMultiSeed(MultiSeedRunner):

    def single_run(self, args: namedtuple):
        return run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong A2C")
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
    parser.add_argument("--batch-size", type=int, default=32,
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
    parser.add_argument("--kl_beta_coef", type=float, default=0.5,
                        help="Vae KL coefficient")
    parser.add_argument("--model_update_freq", type=float, default=1.0,
                        help="Vae update frequency")
    parser.add_argument("--model_batch_size", type=int, default=32,
                        help="Vae update batch size")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Vae update batch size")
    parser.add_argument("--rms-prop-eps", type=float, default=1e-5,
                        help="RmsProp epsion coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum allowed graident norm")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
                        help=("Training length interms of cumulative"
                              " environment timesteps"))
    parser.add_argument("--log-interval", type=int, default=2500,
                        help=("Logging interval in terms of training"
                              " iterations"))
    parser.add_argument("--log-dir", type=str, default=None,
                        help=("Logging dir"))
    parser.add_argument("--sb3-policy", action="store_true",
                        help="Use SB3 policy")
    parser.add_argument("--ortho-init", action="store_true",
                        help="Use orthogonal initialization in the policy")
    parser.add_argument("--log-histogram", action="store_true",
                        help="Log the histogram of weights and gradients")

    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallelized jobs for experiments")
    parser.add_argument("--runs-per-job", type=int, default=1,
                        help="Number of parallelized jobs for experiments")

    args = parser.parse_args()
    args = vars(args)

    runs_per_job = args.pop("runs_per_job")
    n_jobs = args.pop("n_jobs")

    PongA2cMultiSeed(args, runs_per_job=runs_per_job).run(n_jobs=n_jobs)
