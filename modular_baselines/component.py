from typing import Tuple
from abc import ABC, abstractmethod

from modular_baselines.loggers.data_logger import DataLogger


class Component(ABC):

    def __init__(self, logger: DataLogger) -> None:
        super().__init__()
        self.logger = logger
        self._init_default_loggers()

    @abstractmethod
    def _init_default_loggers(self) -> None:
        pass
