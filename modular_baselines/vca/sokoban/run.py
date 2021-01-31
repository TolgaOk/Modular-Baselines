import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import argparse
from copy import deepcopy
from collections import namedtuple
from abc import ABC, abstractmethod
from multiprocessing import Process
import optuna

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from modular_baselines.loggers.basic import(InitLogCallback,
                                            LogRolloutCallback,
                                            LogWeightCallback,
                                            LogGradCallback,
                                            LogHyperparameters)

from modular_baselines.vca.algorithm import DiscreteStateVCA, ChannelStateVCA 
from modular_baselines.vca.buffer import Buffer
from modular_baselines.vca.collector import NStepCollector
from modular_baselines.vca.modules import (ChannelPolicyModule,
                                           FullCategoricalTransitionModule,
                                           CategoricalTransitionModule,
                                           MultiheadBernoulliTransitionModule,
                                           MultiheadCatgoricalTransitionModule)
from modular_baselines.vca.runner import ExperimentRunner
from modular_baselines.vca.maze.environment import ChannelMaze, MazeEnv
from environment import Sokoban


class ChannelRunner(ExperimentRunner):

    def make_env(self, envname):
        if envname == "Sokoban":
            return Sokoban()
        return ChannelMaze(world_map=getattr(MazeEnv, envname))

    def algo_generator(self, args):
        args = namedtuple("Args", args.keys())(*args.values())
        env = self.make_env(args.mazemap)

        set_random_seed(args.seed)
        vecenv = make_vec_env(
            lambda: self.make_env(args.mazemap),
            seed=args.seed)

        hyper_callback = LogHyperparameters(args._asdict())
        rollout_callback = LogRolloutCallback()
        init_callback = InitLogCallback(args.log_interval,
                                        args.log_dir)
        weight_callback = LogWeightCallback("weights.json")
        grad_callback = LogGradCallback("grads.json")

        buffer = Buffer(
            args.buffer_size,
            vecenv.observation_space,
            vecenv.action_space)

        policy_m = ChannelPolicyModule(
            vecenv.observation_space.shape[0],
            vecenv.action_space.n,
            hidden_size=args.policy_hidden_size,
            kernel_size=args.policy_kernel_size)
        trans_m = MultiheadBernoulliTransitionModule(
            vecenv.observation_space.shape[0],
            vecenv.action_space.n,
            kernel_size=args.transition_kernel_size,
            hidden_size=args.transition_hidden_size)

        collector = NStepCollector(
            env=vecenv,
            buffer=buffer,
            policy=policy_m,
            callbacks=[rollout_callback],
            device=args.device)
        algorithm = ChannelStateVCA(
            policy_module=policy_m,
            transition_module=trans_m,
            buffer=buffer,
            collector=collector,
            env=vecenv,
            reward_vals=env.expected_reward(),
            rollout_len=args.rollout_len,
            trans_opt=torch.optim.RMSprop(
                trans_m.parameters(), lr=args.trans_lr),
            policy_opt=torch.optim.RMSprop(
                policy_m.parameters(), lr=args.policy_lr),
            batch_size=args.batchsize,
            entropy_coef=args.entropy_coef,
            device=args.device,
            grad_clip=args.grad_clip,
            callbacks=[init_callback, weight_callback,
                       grad_callback, hyper_callback]
        )

        return algorithm


def default_args(parser):
    parser.add_argument("--buffer_size", help="",  type=int, default=10000)
    parser.add_argument("--batchsize", help="", type=int, default=32)
    parser.add_argument("--rollout_len", help="", type=int, default=10)
    parser.add_argument("--total_timesteps", help="",
                        type=int, default=int(1e4))
    parser.add_argument("--entropy_coef", help="", type=float, default=0.01)

    parser.add_argument("--use_gumbel", help="", action="store_true")
    parser.add_argument("--grad_norm", help="", action="store_true")
    parser.add_argument("--grad_clip", help="", action="store_true")
    parser.add_argument("--ideal_logits", help="", action="store_true")

    parser.add_argument("--policy_hidden_size", help="", type=int, default=64)
    parser.add_argument("--policy_kernel_size", help="", type=int, default=3)
    parser.add_argument("--transition_hidden_size",
                        help="", type=int, default=60)
    parser.add_argument("--transition_kernel_size",
                        help="", type=int, default=3)

    parser.add_argument("--policy_lr", help="", type=float, default=3e-3)
    parser.add_argument("--trans_lr", help="", type=float, default=3e-2)
    parser.add_argument("--alpha", help="", default=1e-3)

    parser.add_argument("--mazemap", help="", type=str,
                        default="little_world_map")

    parser.add_argument("--device", help="", type=str, default="cpu")
    parser.add_argument("--log_interval", help="", type=int, default=95)
    parser.add_argument("--seed", help="", type=int, default=None)
    parser.add_argument("--log_dir", help="", type=str, default="logs/")


def tune_args(parser):
    parser.add_argument("--tune_batchsize", help="",
                        action="extend", nargs="+", type=int)
    parser.add_argument("--tune_rollout_len", help="",
                        action="extend", nargs="+", type=int)
    parser.add_argument("--tune_entropy_coef", help="",
                        action="extend", nargs="+", type=float)

    parser.add_argument("--tune_policy_hidden_size", help="",
                        action="extend", nargs="+", type=int)
    parser.add_argument("--tune_policy_lr", help="",
                        action="extend", nargs="+", type=float)

    parser.add_argument("--tune_transition_hidden_size",
                        help="", action="extend", nargs="+", type=int)
    parser.add_argument("--tune_trans_lr", help="",
                        action="extend", nargs="+", type=float)
    parser.add_argument("--tune_alpha", help="",
                        action="extend", nargs="+", type=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Channel VCA")
    default_args(parser)
    tune_args(parser)
    args = vars(parser.parse_args())

    ChannelRunner(args, n_repeat=1)()
