import torch
import numpy as np
from abc import ABC, abstractmethod


class BasePolicy(ABC):

    @abstractmethod
    def sample_action(self):
        pass

    @abstractmethod
    def init_state(self):
        pass
