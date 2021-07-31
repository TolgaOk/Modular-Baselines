import numpy as np
from typing import Dict, List, Optional, Union
import psutil
import warnings
from abc import ABC, abstractmethod


class BaseBufferCallback(ABC):
    """ Base class for buffer callbacks that only supports:
        on_buffer_push, on_buffer_sample, and on_buffer_init calls.
    """

    @abstractmethod
    def on_buffer_push(self) -> None:
        pass

    @abstractmethod
    def on_buffer_sample(self) -> None:
        pass

    @abstractmethod
    def on_buffer_init(self) -> None:
        pass


class BaseBuffer(ABC):

    @abstractmethod
    def push(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class Buffer():
    """ Standard buffer for both on-policy and off-policy agents.

    Implements a queue structure with a finite capacity. When the buffer is full it overwrites the
    values. Each item in the buffer is a Struct-Array of the given struct type.

        Args:
            struct (np.dtype): Array dtype of a single item
            capacity (int): Maximum size of the buffer
            num_envs (int): Number of parallel enviornments
            callbacks (Optional[Union[List[BaseBufferCallback], BaseBufferCallback]]): Callbacks
    """

    def __init__(self,
                 struct: np.dtype,
                 capacity: int,
                 num_envs: int,
                 callbacks: Optional[Union[List[BaseBufferCallback], BaseBufferCallback]] = None
                 ) -> None:
        self.struct = struct
        self.capacity = capacity
        self.num_envs = num_envs
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks] if callbacks is not None else []
        self.callbacks = callbacks

        available_memory = psutil.virtual_memory().available
        required_memory = struct.itemsize * capacity * num_envs
        if available_memory < required_memory:
            warnings.warn("Required memory {}GB is larger than the available memory {}GB".format(
                required_memory // 2**30, available_memory // 2**30
            ))

        self.buffer = np.zeros(shape=(capacity, num_envs), dtype=struct)
        self._write_index = 0
        self.full = False

        for callback in self.callbacks:
            callback.on_initialization(locals())

    @property
    def size(self) -> int:
        return self.capacity if self.full else self._write_index

    def push(self, item: Dict[str, np.ndarray]) -> None:
        """ Add an item to the buffer.

        Args:
            item (Dict[str, np.ndarray]): Item dictionary that is expected to have the same
                structure with the buffer dtype
        """
        for name in self.struct.names:
            assert name in item.keys(), "Field name {} is missing".format(name)
            expected_shape = (self.num_envs, *self.struct[name].shape)
            array = item[name]
            assert array.shape == expected_shape, (
                "Invalid shape for {}, expected shape: {}, given shape: {}").format(
                    name, expected_shape, array.shape)
            self.buffer[self._write_index][name] = array

        for callback in self.callbacks:
            callback.on_push(locals())

        self._write_index = (self._write_index + 1) % self.capacity
        self.full = self.full or self._write_index == 0

    def sample(self,
               batch_size: int,
               rollout_len: int,
               sampling_length: Optional[int] = None
               ) -> np.ndarray:
        """ Sample a random rollout from the buffer.

        Args:
            batch_size (int): Number of rollouts each having rollout_len size
            rollout_len (int): Size of the rollout (sequence)
            sampling_length (Optional[int]): Length of the sampling range. If left
                None, all the capacity of the buffer is used for sampling.

        Returns:
            np.ndarray: Sampled array of the buffer dtype
        """
        buffer_size = self.size
        sampling_length = sampling_length or buffer_size
        assert sampling_length > 0, "Non-positive sampling length"
        assert sampling_length >= rollout_len, "Sampling length is not big enough"
        assert sampling_length <= buffer_size, "Sampling length cannot exceed buffer size"
        time_indices = (np.random.randint(
            low=buffer_size - sampling_length,
            high=buffer_size - rollout_len + 1,
            size=batch_size) - self._write_index) % buffer_size
        time_indices = (time_indices.reshape(-1, 1) +
                        np.arange(rollout_len).reshape(1, -1)) % buffer_size
        env_indices = np.arange(batch_size).reshape(-1, 1) % self.num_envs

        sample = self.buffer[time_indices, env_indices]

        for callback in self.callbacks:
            callback.on_sample(locals())

        return self.postprocess_sample(sample)

    def postprocess_sample(self, sample: np.ndarray) -> np.ndarray:
        """ Transform the sample array.

        Args:
            sample (np.ndarray): Initial sample of the buffer dtype

        Returns:
            np.ndarray: Transformed sample of the buffer dtype
        """
        return sample
