from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback


class BaseAlgorithmCallback(BaseCallback):
    """ Base class for buffer callbacks that only supports:
    on_training_start, _on_training_start, and on_training_end calls.
    """

    def __init__(self, verbose: int = 0):
        super().__init__()

    def on_training_start(self, *args) -> None:
        self._on_training_start(*args)

    @abstractmethod
    def _on_training_start(self, *args) -> None:
        raise NotImplementedError

    def on_step(self, *args) -> bool:
        self._on_step(*args)

    @abstractmethod
    def _on_step(self, *args) -> bool:
        pass

    def on_training_end(self, *args) -> None:
        self._on_training_end(*args)

    @abstractmethod
    def _on_training_end(self, *args) -> None:
        pass


    def on_rollout_start(self) -> None:
        raise NotImplementedError

    def _on_rollout_start(self) -> None:
        raise NotImplementedError

    def on_rollout_end(self) -> None:
        raise NotImplementedError

    def _on_rollout_end(self) -> None:
        raise NotImplementedError
