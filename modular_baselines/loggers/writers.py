from typing import Any, Callable, Dict, List, Optional, Union, Type, Union
import time
from collections import deque
import numpy as np
import os
import json
from collections import defaultdict
from abc import ABC, abstractmethod

from stable_baselines3.common.logger import Logger, CSVOutputFormat, HumanOutputFormat, JSONOutputFormat

from modular_baselines.algorithms.algorithm import BaseAlgorithmCallback
from modular_baselines.collectors.collector import BaseCollectorCallback


class BaseWriter(BaseAlgorithmCallback):

    prefix: str

    def on_training_start(self, *args) -> None:
        pass

    def on_training_end(self, *args) -> None:
        pass


class ScalarWriter(BaseWriter):

    prefix: str = "scalar"

    def __init__(self,
                 interval: int,
                 dir_path: str,
                 writers: List[Union[CSVOutputFormat, HumanOutputFormat, JSONOutputFormat]]
                 ) -> None:
        super().__init__()
        self.interval = interval
        self.sb3_logger = Logger(dir_path, writers)

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        if locals_["iteration"] % self.interval == 0:
            logger = locals_["self"].logger
            for name, data_log in logger.__dict__.items():
                name_pieces = name.split("/")
                name = "/".join(name_pieces[1:])
                if name_pieces[0] == self.prefix:
                    self.sb3_logger.record(name, data_log.dump())
            self.sb3_logger.dump()

    def write_historgram(self, histgoram_data: Dict[str, Dict[str, List[Union[float, int]]]]) -> None:
        with open(self.path, "a") as jsonfile:
            jsonstr = json.dumps(histgoram_data)
            jsonfile.write(jsonstr + "\n")


class DictWriter(BaseWriter):

    prefix: str = "dict"

    def __init__(self,
                 interval: int,
                 dir_path: str):
        super().__init__()
        self.interval = interval
        self.dir = dir_path

    def on_training_start(self, locals_) -> None:
        logger = locals_["self"].logger
        for name, _ in logger.__dict__.items():
            name_pieces = name.split("/")
            if name_pieces[0] == self.prefix:
                dir_path = os.path.join(self.dir, "/".join(name_pieces[1:-1]))
                os.makedirs(dir_path, exist_ok=True)
                path = os.path.join(dir_path, name_pieces[-1] + ".json")
                if os.path.exists(path):
                    raise FileExistsError(
                        "File at {} already exists".format(path))

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        iteration = locals_["iteration"]
        if iteration % self.interval == 0:
            logger = locals_["self"].logger

            for name, data_log in logger.__dict__.items():
                name_pieces = name.split("/")
                name = "/".join(name_pieces[1:])
                if name_pieces[0] == self.prefix:
                    data = data_log.dump()
                    path = os.path.join(self.dir, name + ".json")
                    with open(path, "a") as json_file:
                        json_str = json.dumps(dict(data))
                        json_file.write(json_str + "\n")


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
