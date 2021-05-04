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
import time
from typing import Dict

from modular_baselines.runners.base import BaseExperimentRunner


class OptimizerRunner(BaseExperimentRunner):

    def __init__(self, args: Dict, runs_per_job: int, seed_repeat: int = 1, direction: str = "maximize"):
        self.seed_repeat = seed_repeat
        self.direction = direction
        super().__init__(args, runs_per_job)

    def run(self, n_jobs=1):
        abs_path = os.path.abspath(os.path.dirname(self.args["log_dir"]))
        log_dir_path = os.path.join(
            abs_path,
            self.args["log_dir"],
            self.log_dir_prefix,)
        os.makedirs(log_dir_path, exist_ok=True)
        storage_url = "".join(("sqlite:///", os.path.join(
            log_dir_path,
            "store.db")))
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(max(2, (n_jobs * self.runs_per_job) // 10), 10))
        optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            study_name=self.log_dir_prefix,
            direction=self.direction)

        if n_jobs == 1:
            return self._run(storage_url)

        processes = []
        for job_ix in range(n_jobs):
            proc = Process(target=self._run, args=(storage_url,))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

    def _run(self, storage_url):

        study = optuna.load_study(
            study_name=self.log_dir_prefix,
            storage=storage_url)
        study.optimize(
            self.objective,
            n_trials=self.runs_per_job)

    def objective(self, trial: optuna.Trial):
        results = []
        trial_args = deepcopy(self.args)

        for hyperparam in list(trial_args.keys()):
            if hyperparam.startswith("tune/"):
                name = hyperparam[5:]
                if name not in trial_args.keys():
                    warnings.warn("{} has an unmatched arg named {}".format(hyperparam, name))
                    continue
                values = trial_args.pop(hyperparam)
                trial_args[name] = trial.suggest_categorical(name, values)
        initial_seed = trial.suggest_int("seed", 2**10, 2**30)

        for index in range(self.seed_repeat):
            args = deepcopy(trial_args)
            args["seed"] = initial_seed + 2 ** index
            args["log_dir"] = os.path.join(
                args["log_dir"],
                self.log_dir_prefix,
                "Trial_{:4d}".format(trial._trial_id).replace(" ", "0"),
                "{:4d}".format(index).replace(" ", "0"))
            args = namedtuple("args", args.keys())(*args.values())
            results.append(self.single_run(args))

        return sum(results) / self.seed_repeat
