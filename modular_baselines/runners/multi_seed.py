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


class MultiSeedRunner(BaseExperimentRunner):

    def __init__(self, args: Dict, runs_per_job: int):
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
        optuna.create_study(
            storage=storage_url,
            study_name=self.log_dir_prefix)

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
        sampler = optuna.samplers.RandomSampler()

        # print(storage_url)
        # return
        study = optuna.load_study(
            study_name=self.log_dir_prefix,
            sampler=sampler,
            storage=storage_url)
        study.optimize(
            self.objective,
            n_trials=self.runs_per_job)

    def objective(self, trial: optuna.Trial):
        args = deepcopy(self.args)
        args["seed"] = trial.suggest_int("seed", 2**10, 2**30)
        args["log_dir"] = os.path.join(
            args["log_dir"],
            self.log_dir_prefix,
            "{:4d}".format(trial._trial_id).replace(" ", "0"))

        args = namedtuple("args", args.keys())(*args.values())
        return self.single_run(args)
