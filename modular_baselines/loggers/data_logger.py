from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
from abc import abstractmethod
from collections import defaultdict
import numpy as np


class BaseDataLog():

    ApplyCallableType = Callable[[Any], Any]
    ReduceCallableType = Callable[[Any], Any]
    LogItem = Any

    def __init__(self,
                 apply_fn: Optional[ApplyCallableType] = None,
                 reduce_fn: Optional[ReduceCallableType] = None
                 ) -> None:
        self.internal = self.default_internal()
        self.apply_fn = apply_fn if apply_fn is not None else lambda x: x
        self.reduce_fn = reduce_fn if reduce_fn is not None else lambda x: x

    def push(self, value: LogItem) -> None:
        self.internal = self._push(self.internal, self.apply_fn(value))

    def dump(self) -> Any:
        self.internal, output = self._dump(self.internal)
        return self.reduce_fn(output)
        
    @staticmethod
    @abstractmethod
    def default_internal() -> Any:
        pass

    @staticmethod
    @abstractmethod
    def _push(internal: Any, value: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def _dump(internal: Any) -> Tuple[Any, Any]:
        pass


class LastDataLog(BaseDataLog):

    @staticmethod
    def default_internal() -> Any:
        return None

    @staticmethod
    def _push(internal: Any, value: Any) -> Any:
        return value

    @staticmethod
    def _dump(internal: Any) -> Tuple[Any, Any]:
        return LastDataLog.default_internal(), internal


class ListDataLog(BaseDataLog):

    ReduceCallableType = Callable[[List[Any]], Any]

    @staticmethod
    def default_internal() -> Any:
        return []

    @staticmethod
    def _push(internal: Any, value: Any) -> Any:
        internal.append(value)
        return internal

    @staticmethod
    def _dump(internal: Any) -> Tuple[Any, Any]:
        return ListDataLog.default_internal(), internal


class BaseHistogram():

    HistogramData = Dict[str, Tuple[np.ndarray, np.ndarray]]

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins

    def calculate_histogram(self, arrays: Dict[str, np.ndarray]) -> HistogramData:
        histogram_data = {}
        for name, param in arrays.items():
            freqs, bins = np.histogram(param, bins=self.n_bins, density=True)
            histogram_data[name] = {
                "bins": bins.tolist(),
                "freqs": freqs.tolist()}
        return histogram_data


class HistListDataLog(ListDataLog, BaseHistogram):

    ReduceCallableType = Callable[[List[Any]], Dict[str, Any]]

    def __init__(self,
                 n_bins: int,
                 reduce_fn: ReduceCallableType,
                 apply_fn: Optional[BaseDataLog.ApplyCallableType] = None,
                 ) -> None:
        super().__init__(apply_fn, reduce_fn=lambda values: self.calculate_histogram(reduce_fn(values)))
        BaseHistogram.__init__(self, n_bins)


class ParamHistDataLog(LastDataLog, BaseHistogram):
    FetchParamCallable = Callable[[], Dict[str, np.ndarray]]
    LogItem = FetchParamCallable

    def __init__(self, n_bins: int) -> None:
        super().__init__(None, reduce_fn=self.reduce_param_hist)
        BaseHistogram.__init__(self, n_bins)

    def reduce_param_hist(self,
                          fetch_param_fn: FetchParamCallable
                          ) -> BaseHistogram.HistogramData:
        params = fetch_param_fn()
        return self.calculate_histogram(params)


class SequenceNormDataLog(BaseDataLog):

    Scalar = Union[np.float32, np.float64, float]
    InternalData = Dict[int, List[Scalar]]

    def __init__(self,
                 reduce_fn: Optional[BaseDataLog.ReduceCallableType] = None) -> None:
        super().__init__(None, reduce_fn)
        self.internal = self.default_internal()

    @staticmethod
    def default_internal() -> Any:
        return defaultdict(list)

    def add(self, time: int, value: Scalar) -> None:
        self.internal[time].append(value)

    @staticmethod
    def _push(self, *args) -> None:
        raise NotImplementedError

    @staticmethod
    def _dump(internal: defaultdict) -> Tuple[defaultdict, InternalData]:
        return SequenceNormDataLog.default_internal(), dict(internal)


class BaseNormDataLog(BaseDataLog):

    Scalar = Union[np.float32, np.float64, float]
    MetricCallableType = Callable[[np.ndarray], Scalar]

    def __init__(self, metric_fn: MetricCallableType) -> None:
        super().__init__(metric_fn, reduce_fn=None)


class LastNormDataLog(BaseNormDataLog, LastDataLog):
    pass


class ListNormDataLog(BaseNormDataLog, ListDataLog):
    pass


class DataLogger():

    def __init__(self, **data_logs: Optional[BaseDataLog]) -> None:
        for name, data_log in data_logs.items():
            self.__setattr__(name, data_log)

    def dump(self) -> None:
        return {name: value.dump() for name, value in self.__dict__.items() if isinstance(value, BaseDataLog)}

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, BaseDataLog):
            self.__dict__[__name] = __value
        if __name in self.__dict__.keys():
            raise ValueError(f"DataLog can not be reassigned or the name <{__name}> already exist.")
        else:
            raise ValueError(f"Only DataLogs are supported for assignments, given type: {type(__value)}")

    def check_attributes(self, names: List[str]) -> bool:
        for name in names:
            if name not in self.__dict__.keys():
                raise ValueError(f"Missing Data logger attribute: {name}")
        return True

    def add_if_not_exists(self, loggers: Dict[str, BaseDataLog]):
        for name, data_logger in loggers.items():
            if name not in self.__dict__.keys():
                setattr(self, name, data_logger)
