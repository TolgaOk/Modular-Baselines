from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple, Protocol
from abc import abstractmethod
from collections import deque
import numpy as np


class BaseDataLog():

    ReduceCallableType = Callable[[Any], Any]
    LogItem = Any

    def __init__(self,
                 ) -> None:
        self.initialize()
        self.push_callbacks = []

    def push(self, value: LogItem) -> None:
        self._push(value)
        for callback, condition in self.push_callbacks:
            if condition(self):
                callback(self)

    def fetch(self) -> Any:
        return self._fetch()

    def add_callback(self, callback_fn: Callable[["BaseDataLog"], None], condition_fn: Callable[["BaseDataLog"], bool]) -> None:
        self.push_callbacks.append([callback_fn, condition_fn])

    @abstractmethod
    def initialize(self) -> None: ...
    @abstractmethod
    def _push(self, value: LogItem) -> None: ...
    @abstractmethod
    def _fetch(self) -> Any: ...


class ListDataLog(BaseDataLog):

    LogItem = Any
    ReduceCallableType = Callable[[List[LogItem]], Any]

    def initialize(self) -> None:
        self.data = []

    def _push(self, value: LogItem) -> None:
        self.data.append(value)

    def _fetch(self) -> Any:
        data = self.data
        self.initialize()
        return data


class SingularDataLog(BaseDataLog):

    LogItem = Any
    ReduceCallableType = Callable[[List[LogItem]], Any]

    def initialize(self) -> None:
        self.data = None

    def _push(self, value: LogItem) -> None:
        self.data = value

    def _fetch(self) -> Any:
        return self.data


class HistogramDataLog(BaseDataLog):

    LogItem = np.ndarray
    HistogramData = Dict[str, Tuple[np.ndarray, np.ndarray]]

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        super().__init__()

    def initialize(self) -> None:
        self.data = []

    def calculate_histogram(self, array: np.ndarray) -> HistogramData:
        freqs, bins = np.histogram(array, bins=self.n_bins, density=True)
        return {
            "bins": bins.tolist(),
            "freqs": freqs.tolist()
        }

    def _push(self, value: LogItem) -> None:
        self.data.append(value)

    def _fetch(self) -> Any:
        data = self.data
        self.initialize()
        return self.calculate_histogram(data)
