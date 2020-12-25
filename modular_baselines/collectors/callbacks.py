from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback


class BaseCollectorCallback(BaseCallback):
    """ Base class for buffer callbacks that only supports:
    on_rollout_start, on_rollout_step, and on_rollout_end calls.
    """

    def __init__(self, verbose: int = 0):
        super().__init__()

    def on_rollout_start(self, *args) -> None:
        self._on_rollout_start(*args)

    @abstractmethod
    def _on_rollout_start(self, *args) -> None:
        pass

    def on_rollout_step(self, *args) -> None:
        self._on_rollout_step(*args)

    @abstractmethod
    def _on_rollout_step(self, *args) -> None:
        pass

    def on_rollout_end(self, *args) -> None:
        self._on_rollout_end(*args)

    @abstractmethod
    def _on_rollout_end(self, *args) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _on_training_start(self) -> None:
        raise NotImplementedError

    def on_step(self) -> bool:
        raise NotImplementedError

    def _on_step(self) -> bool:
        raise NotImplementedError

    def on_training_end(self) -> None:
        raise NotImplementedError

    def _on_training_end(self) -> None:
        raise NotImplementedError
