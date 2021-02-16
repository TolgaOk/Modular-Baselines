import numpy as np
import os
import json
import warnings
import datetime
import argparse
from copy import deepcopy
from collections import namedtuple
from abc import ABC, abstractmethod
from multiprocessing import Process
import optuna
from typing import Dict

from modular_baselines.runners.base import BaseExperimentRunner


class MultiSeedRunner(BaseExperimentRunner):

    def __init__(self, args: Dict, n_seeds: int):
        super().__init__(args, n_seeds)

    def run(self, n_jobs=1):
        sampler = optuna.samplers.RandomSampler()
        study = optuna.create_study(sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_seeds, n_jobs=n_jobs)

    def objective(self, trial: optuna.Trial):
        args = deepcopy(self.args)
        args["seed"] = trial.suggest_int("seed", 2**10, 2**30)
        args["log_dir"] = os.path.join(
            args["log_dir"],
            self.log_dir_prefix,
            "{:4d}".format(trial._trial_id).replace(" ", "0"))

        args = namedtuple("args", args.keys())(*args.values())
        return self.single_run(args)
