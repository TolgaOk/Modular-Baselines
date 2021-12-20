from typing import Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Optional[Any] = None
                      ) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def init_state(self, batch_size: Optional[int] = None) -> Any:
        pass
