from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple, Protocol, Collection
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint
import numpy as np

from modular_baselines.loggers.datalog import BaseDataLog as DataLog


class KeyDefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Writer(Protocol):
    def write(self, name: str, logger: Dict[str, Any]) -> None: ...


Reducer: Type = Callable[[Any], Any]


@dataclass
class LogGroup():
    loggers: Dict[str, Union[
        "LogGroup",
        Tuple[DataLog, Reducer]]
    ]
    name: Optional[str] = None
    writers: Optional[List[Writer]] = None

    def write(self) -> Dict[str, Any]:
        if self.writers is not None and len(self.writers) > 0:
            log_dump = self.prepare()
            for writer in self.writers:
                writer.write(self.name, log_dump)

    def add_trigger(self, data_log: DataLog, condition_fn: Callable[[DataLog], bool]) -> None:
        data_log.add_callback(self.write, condition_fn)

    def prepare(self) -> Dict[str, Any]:
        log_values = KeyDefaultdict(lambda data_log: data_log.fetch())

        def _write_item(item: Union["LogGroup", Tuple[DataLog, Reducer]]) -> Any:
            if isinstance(item, LogGroup):
                return {
                    name: _write_item(value)
                    for name, value in item.loggers.items()
                }
            if isinstance(item, tuple):
                data_log, reducer = item
                log_value = log_values[data_log]
                return reducer(log_value)
            else:
                raise ValueError(f"Unexpected input type: {type(item)}")

        return _write_item(self)


class MBLogger():

    def __init__(self) -> None:
        self._log_groups = {}

    def __setattr__(self, name: str, data_log: DataLog) -> None:
        if name in self.__dict__.keys():
            raise ValueError(f"DataLog can not be reassigned! The name <{name}> already exist.")
        self.__dict__[name] = data_log

    def log(self, name: str) -> DataLog:
        if name not in self.__dict__.keys():
            raise ValueError(F"Data logger: {name} is not registered!")
        return self.__dict__[name]

    def add_group(self, log_group: LogGroup) -> None:
        if log_group.name is None:
            raise ValueError("Log group must have a name!")
        self._log_groups[log_group.name] = log_group

    def write(self, name: str) -> None:
        if name not in self._log_groups.keys():
            raise ValueError(f"The writer: {name} is not registered!")
        self._log_groups[name].write()
