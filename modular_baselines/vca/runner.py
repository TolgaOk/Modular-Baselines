import numpy as np
import os
import json
import datetime
import argparse
from copy import deepcopy
from collections import namedtuple
from abc import ABC, abstractmethod
from multiprocessing import Process
import optuna


class ExperimentRunner(ABC):

    def __init__(self, args, n_repeat=1):
        self.log_dir_prefix = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.n_repeat = n_repeat
        self.args = args

    def _objective(self, trial):

        args = deepcopy(self.args)

        for key, value in args.items():
            if key.startswith("tune_") and value is not None:
                name = key.replace("tune_", "")
                args[name] = trial.suggest_categorical(name, value)

        self(kwargs=args)
        return 0

    def __call__(self, kwargs=None):
        if kwargs is None:
            kwargs = self.args

        if self.n_repeat == 1:
            log_dir_suffix = ""

        if kwargs["seed"] is None:
            kwargs["seed"] = np.random.randint(0, 2**20)

        processes = []
        for log_dir_suffix in range(1, self.n_repeat + 1):
            args = deepcopy(kwargs)
            args["seed"] += (2**10) * (log_dir_suffix - 1)
            args["log_dir"] = os.path.join(
                kwargs["log_dir"],
                self.log_dir_prefix,
                datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S-%f"),
                str(log_dir_suffix) if self.n_repeat != 1 else "")
            proc = Process(target=self.single_run, args=(args,))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

    def single_run(self, kwargs):
        algo = self.algo_generator(kwargs)
        algo.learn(kwargs["total_timesteps"])

    def tune(self, n_trials):
        # sampler = optuna.samplers.TPESampler(multivariate=True)
        sampler = optuna.samplers.RandomSampler()
        study = optuna.create_study(sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)

    @abstractmethod
    def algo_generator(self, args):
        pass
