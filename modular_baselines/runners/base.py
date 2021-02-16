import datetime
import warnings
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory


class BaseExperimentRunner(ABC):

    def __init__(self, args, n_seeds):
        if args["log_dir"] is None:
            args["log_dir"] = TemporaryDirectory().name
            warnings.warn(
                "Temporary directory: {} is created for logging!".format(
                    args["log_dir"]))

        self.log_dir_prefix = self.current_time()
        self.n_seeds = n_seeds
        self.args = args

    @staticmethod
    def current_time():
        return datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def single_run(self, args):
        pass
