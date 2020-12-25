from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback


class BaseBufferCallback(BaseCallback):
    """ Base class for buffer callbacks that only supports:
        on_buffer_add, on_buffer_sample, and on_buffer_get calls.
    """

    def __init__(self, verbose: int = 0):
        super().__init__()

    def on_buffer_add(self) -> None:
        self._on_buffer_add()

    @abstractmethod
    def _on_buffer_add(self) -> None:
        pass

    def on_buffer_sample(self) -> None:
        self._on_buffer_sample()

    @abstractmethod
    def _on_buffer_sample(self) -> None:
        pass

    def on_buffer_get(self) -> None:
        self._on_buffer_get()

    @abstractmethod
    def _on_buffer_get(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        raise NotImplementedError

    def _on_rollout_start(self) -> None:
        raise NotImplementedError

    def on_rollout_end(self) -> None:
        raise NotImplementedError

    def _on_rollout_end(self) -> None:
        raise NotImplementedError

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
