from typing import Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod
from modular_baselines.loggers.data_logger import DataLogger

import numpy as np


class BasePolicy(ABC):

    def set_logger(self, logger: DataLogger) -> None:
        self.logger = logger
        self._init_default_loggers()

    @abstractmethod
    def _init_default_loggers(self) -> None:
        pass

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Optional[Any] = None
                      ) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def init_hidden_state(self, batch_size: Optional[int] = None) -> Any:
        pass
