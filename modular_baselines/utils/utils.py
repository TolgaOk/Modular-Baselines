from typing import Optional, Any, Dict, Tuple, Union, List, Callable
import torch
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import VectorEnv

def nested(function: Callable[[Union[torch.Tensor, np.ndarray, jnp.ndarray]], Union[torch.Tensor, np.ndarray, jnp.ndarray]]):
    def nested_apply(collection: Union[np.ndarray, Dict[str, Any], List[Any], Tuple[Any]], **kwargs):
        if isinstance(collection, dict):
            return {name: nested_apply(value, **kwargs) for name, value in collection.items()}
        if isinstance(collection, (list, tuple)):
            cls = type(collection)
            return cls([nested_apply(value, **kwargs) for value in collection])
        if isinstance(collection, (torch.Tensor, np.ndarray, jnp.ndarray)):
            return function(collection, **kwargs)
        raise ValueError(f"Type {type(collection)} is not supported!")
    return nested_apply


def to_torch(device: str, ndarray: np.ndarray):
    @nested
    def _to_torch(ndarray):
        return torch.from_numpy(ndarray).to(device)
    return _to_torch(ndarray)

def to_jax(ndarray: jnp.ndarray):
    @nested
    def _to_jax(ndarray):
        return jnp.array(ndarray, dtype=float)
    return _to_jax(ndarray)

@nested
def flatten_time(tensor: Union[torch.Tensor, jnp.ndarray]) -> Union[torch.Tensor, jnp.ndarray]:
    n_envs, n_rollout = tensor.shape[:2]
    return tensor.reshape(n_envs * n_rollout, *tensor.shape[2:])


def param_dict_as_numpy(named_parameters) -> Dict[str, np.ndarray]:
    return {name: param.detach().cpu().numpy()
            for name, param in named_parameters}


def grad_dict_as_numpy(named_parameters) -> Dict[str, np.ndarray]:
    return {name: param.grad.cpu().numpy() if param.grad is not None else param.grad
            for name, param in named_parameters}


def get_spaces(env: VectorEnv):
    # TODO: Add different observation spaces
    observation_space = env.single_observation_space
    # TODO: Add different action spaces
    action_space = env.single_action_space

    if not isinstance(observation_space, Box):
        raise NotImplementedError("Only Box observations are available")
    if not isinstance(action_space, (Box, Discrete)):
        raise NotImplementedError("Only Discrete and Box actions are available")

    action_dim = action_space.shape[-1] if isinstance(action_space, Box) else 1
    return observation_space, action_space, action_dim
