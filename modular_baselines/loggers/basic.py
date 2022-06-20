from typing import Any, Callable, Dict, List, Optional, Union, Type
import time
from collections import deque
import numpy as np
import os
import json
from collections import defaultdict
from abc import ABC, abstractmethod

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import safe_mean

from modular_baselines.algorithms.algorithm import BaseAlgorithmCallback
from modular_baselines.collectors.collector import BaseCollectorCallback


class InitLogCallback(BaseAlgorithmCallback):
    """ Initialize a standard Stable-Baselines3 logger without tensorboard writer.
    At every training step, check if the the logging period matches with the
    iteration number. If so, record the basic information about the training and
    dump the logs.

    Args:
        log_interval (int): Log at every n steps.
        save_dir (str, optional): Path of the directory to save files if there
            any. Defaults to None.
    """

    def __init__(self, logger: Logger, log_interval: int):
        self.log_interval = log_interval
        self.logger = logger
        self.start_time = None

    def on_training_start(self, *args) -> None:
        self.start_time = time.time()

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        if locals_["iteration"] % self.log_interval == 0:
            fps = int(locals_["num_timesteps"] /
                      (time.time() - self.start_time))
            self.logger.record("time/iterations",
                               locals_["iteration"],
                               exclude="tensorboard")
            self.logger.record("time/fps",
                               fps)
            self.logger.record("time/time_elapsed",
                               int(time.time() - self.start_time),
                               exclude="tensorboard")
            self.logger.record("time/total_timesteps",
                               locals_["num_timesteps"],
                               exclude="tensorboard")
            self.logger.dump(step=locals_["num_timesteps"])

    def on_training_end(self, *args) -> None:
        pass


class STDOUTLoggerCallback(BaseAlgorithmCallback):

    def __init__(self, interval: int) -> None:
        super().__init__()
        self.interval = interval

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        if locals_["iteration"] % self.interval == 0:
            logger = locals_["self"].logger
            print("-" * 20)
            for data_log in logger.__dict__.values():
                print(f"| {data_log.formatting(data_log.dump())}")
            print("-" * 20)

    def on_training_start(self, *args) -> None:
        pass

    def on_training_end(self, *args) -> None:
        pass


class LogRolloutCallback(BaseCollectorCallback):
    """ Accumulate the rewards and episode lengths of the experiences gathered
    by the collector. At the end of a rollout, record the average values. 
    """

    def __init__(self, logger: Logger):
        super().__init__()
        self.reset_info()
        self.logger = logger

    def reset_info(self):
        self.ep_info_buffer = deque(maxlen=100)
        self.ep_success_buffer = deque(maxlen=100)

    def on_rollout_start(self, *args) -> None:
        pass

    def on_rollout_step(self, locals_: Dict[str, Any]) -> None:
        dones = locals_["dones"]
        infos = locals_["infos"]

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def on_rollout_end(self, locals_: Dict[str, Any]) -> None:
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            for ep_info in self.ep_info_buffer:
                self.logger.record_mean(
                    "rollout/ep_rew_mean",
                    ep_info["r"])
                self.logger.record_mean(
                    "rollout/ep_len_mean",
                    ep_info["l"])
        self.reset_info()


class LogLossCallback(BaseAlgorithmCallback):

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        for name, loss in locals_["loss_dict"].items():
            self.logger.record_mean("train/{}".format(name), loss)

    def on_training_start(self, *args) -> None:
        pass

    def on_training_end(self, *args) -> None:
        pass


class LogParamCallback(BaseAlgorithmCallback, ABC):

    def __init__(self,
                 log_dir: str,
                 file_name: str,
                 log_interval: int = 100,
                 n_bins: int = 20):
        super().__init__()
        self.file_name = file_name
        self.log_interval = log_interval
        self.n_bins = n_bins
        self._reset_buffer()

        self.dir = os.path.join(log_dir, "hist")
        self.path = os.path.join(self.dir, self.file_name)
        if os.path.exists(self.path):
            raise FileExistsError(
                "File at {} already exists".format(self.path))
        os.makedirs(self.dir, exist_ok=True)

    def on_training_start(self, *args) -> None:
        pass

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        policy = locals_["self"].policy
        iteration = locals_["iteration"]

        for name, weight in policy.named_parameters():
            if self._param(weight) is not None:
                self.histogram_buffer[name].append(
                    self._param(weight).detach().cpu().clone().numpy())

        if iteration % self.log_interval == 0:
            for name, weight_list in self.histogram_buffer.items():
                freqs, bins = np.histogram(
                    np.stack(weight_list), bins=self.n_bins, density=True)
                self.histogram_buffer[name] = {
                    "bins": bins.tolist(),
                    "freqs": freqs.tolist()}
            with open(self.path, "a") as jsonfile:
                jsonstr = json.dumps(dict(self.histogram_buffer))
                jsonfile.write(jsonstr + "\n")
            self._reset_buffer()

    @abstractmethod
    def _param(self, weight):
        raise NotImplementedError

    def _reset_buffer(self):
        self.histogram_buffer = defaultdict(list)

    def on_training_end(self, *args) -> None:
        pass


class TorchLogGradCallback(LogParamCallback):

    def _param(self, weight):
        return weight.grad


class TorchLogWeightCallback(LogParamCallback):

    def _param(self, weight):
        return weight


class LogHyperparameters(BaseAlgorithmCallback):

    def __init__(self,
                 log_dir: str,
                 hyperparameters: Dict[str, Any],
                 file_name: str = "hyperparameters.json"):
        self.file_name = file_name
        self.log_dir = log_dir
        self.hyperparameters = hyperparameters

    def on_training_start(self, *args) -> None:
        path = os.path.join(self.log_dir, self.file_name)
        with open(path, "w") as fobj:
            json.dump(self.hyperparameters, fobj)

    def on_training_end(self, *args) -> None:
        pass

    def on_step(self, *args) -> None:
        pass
