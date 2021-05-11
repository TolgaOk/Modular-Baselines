""" Example Pong-ram training using A2C. """
import gym
import torch
import numpy as np
import argparse
import itertools
from functools import partial
from collections import namedtuple

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from modular_baselines.collectors.collector import OnPolicyCollector
from modular_baselines.runners.multi_seed import MultiSeedRunner
from modular_baselines.runners.optimizer import OptimizerRunner
from modular_baselines.utils.score import log_score
from modular_baselines.loggers.basic import(InitLogCallback,
                                            LogRolloutCallback,
                                            LogWeightCallback,
                                            LogGradCallback,
                                            LogHyperparameters)

from modular_baselines.contrib.jacobian_trace.buffer import JTACBuffer
from modular_baselines.contrib.jacobian_trace.jtac import JTAC
from modular_baselines.contrib.jacobian_trace.models.discrete_latent import DiscreteJTACModel
from modular_baselines.contrib.jacobian_trace.models.dense import (DenseModel,
                                                                   CategoricalDistributionModel,
                                                                   NormalDistributionModel)


def create_fc_policy(input_size, action_size, args):
    encoder_model = DenseModel(
        input_size=input_size,
        output_size=np.product(args.latent_shape) * args.embedding_size,
        layers=2,
        hidden_size=200,
        output_shape=(*args.latent_shape, args.embedding_size),
        activation=torch.nn.ELU)
    decoder_model = NormalDistributionModel(
        input_size=np.product(args.latent_shape) * args.embedding_size,
        output_size=input_size,
        layers=2,
        hidden_size=200,
        output_shape=None,
        activation=torch.nn.ELU,
        min_std=0.5)
    transition_model = DenseModel(
        input_size=np.product(args.latent_shape) * args.n_embeddings,
        output_size=np.product(args.latent_shape) * args.n_embeddings * action_size,
        layers=2,
        hidden_size=200,
        output_shape=(*args.latent_shape, args.n_embeddings, action_size),
        activation=torch.nn.ELU)
    actor_model = DenseModel(
        input_size=np.product(args.latent_shape) * args.embedding_size,
        output_size=action_size,
        layers=2,
        hidden_size=200,
        output_shape=None,
        activation=torch.nn.ELU)
    critic_model = DenseModel(
        input_size=np.product(args.latent_shape) * args.embedding_size,
        output_size=1,
        layers=2,
        hidden_size=200,
        output_shape=None,
        activation=torch.nn.ELU)

    policy = DiscreteJTACModel(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        transition_model=transition_model,
        actor_model=actor_model,
        critic_model=critic_model,
        n_embeddings=args.n_embeddings,
        embedding_size=args.embedding_size,
        model_lr=args.model_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr)
    return policy


def run(args):

    seed = args.seed
    if args.seed is None:
        seed = np.random.randint(0, 2**16)

    # Logger Callbacks
    hyper_callback = LogHyperparameters(args._asdict())
    rollout_callback = LogRolloutCallback()
    learn_callback = InitLogCallback(args.log_interval // (args.rollout_len * args.n_envs),
                                     args.log_dir)
    algorithm_callbacks = [learn_callback, hyper_callback]
    if args.log_histogram:
        weight_callback = LogWeightCallback("weights.json")
        grad_callback = LogGradCallback("grads.json")
        algorithm_callbacks += [weight_callback, grad_callback]

    # Environment
    vecenv = make_vec_env(env_id=args.envname,
                          n_envs=args.n_envs,
                          seed=args.seed,
                          vec_env_cls=SubprocVecEnv)

    # Policy
    policy = create_fc_policy(vecenv.observation_space.shape[0],
                              vecenv.action_space.n,
                              args)

    # Modules
    buffer = JTACBuffer(buffer_size=max(args.rollout_len, args.buffer_size),
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
    model = JTAC(
        policy=policy,
        rollout_buffer=buffer,
        collector=collector,
        env=vecenv,
        rollout_len=args.rollout_len,
        ent_coef=args.ent_coef,
        vf_coef=args.val_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        model_loss_coef=args.model_loss_coef,
        trans_loss_coef=args.trans_loss_coef,
        model_iteration_per_update=args.model_iteration_per_update *
        (args.rollout_len // args.model_horizon),
        model_batch_size=args.model_batch_size,
        model_horizon=args.model_horizon,
        policy_iteration_per_update=args.policy_iteration_per_update *
        (args.rollout_len // args.policy_nstep),
        policy_batch_size=args.policy_batch_size,
        policy_nstep=args.policy_nstep,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        enable_jtac=args.enable_jtac,
        callbacks=algorithm_callbacks,
        device=args.device)

    # Start learning
    model.learn(args.total_timesteps)

    return log_score(args.log_dir)


class LunarA2Crunner(OptimizerRunner):

    def single_run(self, args: namedtuple):
        return run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong-Ram A2C")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed")
    parser.add_argument("--envname", type=str, default="LunarLander-v2",
                        help="Gym environment name")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")

    parser.add_argument("--rollout-len", type=int, default=5,
                        help="Rollout Length")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Size of the buffer per environment")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="GAE coefficient")

    parser.add_argument("--model-lr", type=float, default=4e-4,
                        help="Model Learning rate")
    parser.add_argument("--actor-lr", type=float, default=4e-4,
                        help="Actor Learning rate")
    parser.add_argument("--critic-lr", type=float, default=4e-4,
                        help="Critic Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient")
    parser.add_argument("--val-coef", type=float, default=0.25,
                        help="Value loss coefficient")
    parser.add_argument("--model-loss-coef", type=float, default=1.0,
                        help="Model loss coefficient")
    parser.add_argument("--trans-loss-coef", type=float, default=5.0,
                        help="Transition loss coefficient")

    parser.add_argument("--model-iteration-per-update", type=int, default=1,
                        help="Number of updates per train call")
    parser.add_argument("--model-batch-size", type=int, default=32,
                        help="Batch size for the model loss")
    parser.add_argument("--model-horizon", type=int, default=5,
                        help="sampling horizon of the model loss")
    parser.add_argument("--policy-iteration-per-update", type=int, default=1,
                        help="Number of updates per train call")
    parser.add_argument("--policy-batch-size", type=int, default=16,
                        help="Batch size for the policy loss")
    parser.add_argument("--policy-nstep", type=int, default=5,
                        help="Trancation value of GAE (n-step)")

    parser.add_argument("--n-embeddings", type=int, default=128,
                        help="Number of embedding vectors")
    parser.add_argument("--embedding-size", type=int, default=32,
                        help="Size of an embedding vector")
    parser.add_argument("--latent-shape", type=int, nargs="+", default=(3, 3),
                        help="Spatial shape of the discrete encodings")

    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum allowed graident norm")
    parser.add_argument("--reward-clip", type=float, default=None,
                        help="Clip value of intermediate rewards")
    parser.add_argument("--enable-jtac", action="store_true",
                        help="Enable JTAC loss instead of usual PG loss")

    parser.add_argument("--tune/rollout-len", type=int, nargs="+", default=[5],
                        help="TUNE: Rollout Length")
    parser.add_argument("--tune/actor-lr", type=float, nargs="+", default=[4e-4],
                        help="TUNE: Learning rate")
    parser.add_argument("--tune/critic-lr", type=float, nargs="+", default=[4e-4],
                        help="TUNE: Learning rate")
    parser.add_argument("--tune/model-lr", type=float, nargs="+", default=[4e-4],
                        help="TUNE: Learning rate")
    parser.add_argument("--tune/ent-coef", type=float, nargs="+", default=[0.1],
                        help="TUNE: Entropy coefficient")

    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
                        help=("Training length interms of cumulative"
                              " environment timesteps"))
    parser.add_argument("--log-interval", type=int, default=20000,
                        help=("Logging interval in terms of environment"
                              " time steps"))
    parser.add_argument("--log-dir", type=str, default=None,
                        help=("Logging dir"))
    parser.add_argument("--log-histogram", action="store_true",
                        help="Log the histogram of weights and gradients")

    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallelized jobs for experiments")
    parser.add_argument("--seed-repeat", type=int, default=1,
                        help="Number of experiment with different seeds per hyperparameter set")
    parser.add_argument("--runs-per-job", type=int, default=1,
                        help="Number of parallelized jobs for experiments")

    args = parser.parse_args()
    args = vars(args)

    runs_per_job = args.pop("runs_per_job")
    n_jobs = args.pop("n_jobs")
    seed_repeat = args.pop("seed_repeat")

    LunarA2Crunner(args, runs_per_job=runs_per_job, seed_repeat=seed_repeat).run(n_jobs=n_jobs)
