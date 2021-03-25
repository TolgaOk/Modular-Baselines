import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import os
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


class EnvWorker():

    def __init__(self, wrapped_env_fn):
        self.env = wrapped_env_fn.var()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.env.reset()
        return self._to_torch_obs(observation), reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self):
        observation = self.env.reset()
        return self._to_torch_obs(observation)

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def get_spaces(self):
        return (self.env.observation_space, self.env.action_space)

    def env_method(self, name, *args, **kwargs):
        method = getattr(self.env, name)
        return method(*args, **kwargs)

    def get_attr(self, name):
        return getattr(self.env, name)

    def set_attr(self, name, value):
        return setattr(self.env, name, value)

    @staticmethod
    def _to_torch_obs(obs):
        return torch.from_numpy(obs) 


def init_env_worker(rank, world_size):
    set_ip()
    rpc.init_rpc("worker_{}".format(rank), rank=rank, world_size=world_size)
    rpc.shutdown(graceful=True)


def call_method(self_reff, method, *args, **kwargs):
    return method(self_reff.local_value(), *args, **kwargs)


def set_ip():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "295010"


class RpcVecEnv(VecEnv):
    """ Creates a synchronized multiprocessing rpc vector environment.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self._step_reffs = None
        n_envs = len(env_fns)
        rpc_world_size = n_envs + 1

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.worker_reffs = []
        self.processes = []
        for rank in range(1, len(env_fns)+1):
            process = ctx.Process(target=init_env_worker, args=(rank, rpc_world_size), daemon=True)
            process.start()

            self.processes.append(process)
        set_ip()
        rpc.init_rpc("master", rank=0, world_size=rpc_world_size)

        for rank, env_fn in enumerate(env_fns):
            self.worker_reffs.append(
                rpc.remote("worker_{}".format(rank + 1),
                           EnvWorker,
                           (CloudpickleWrapper(env_fn),)))

        observation_space, action_space = rpc.rpc_sync(
            "worker_1", call_method, (self.worker_reffs[0], EnvWorker.get_spaces))
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        refs = [
            rpc.rpc_async(worker_reff.owner(), call_method, (worker_reff, EnvWorker.step, act))
            for worker_reff, act in zip(self.worker_reffs, actions)
        ]
        self._step_reffs = refs
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [ref.wait() for ref in self._step_reffs]
        self._step_reffs = None
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        refs = [
            rpc.rpc_async(worker_reff.owner(),
                          call_method,
                          (worker_reff, EnvWorker.step, seed + rank))
            for rank, worker_reff in enumerate(self.worker_reffs)
        ]
        return [ref.wait() for ref in refs]

    def reset(self) -> VecEnvObs:
        refs = [
            rpc.rpc_async(worker_reff.owner(), call_method, (worker_reff, EnvWorker.reset))
            for worker_reff in self.worker_reffs
        ]
        return _flatten_obs([ref.wait() for ref in refs], self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        for worker_reff in self.worker_reffs:
            rpc.rpc_async(worker_reff.owner(), call_method, (worker_reff, EnvWorker.close))
        for process in self.processes:
            process.join()
        rpc.shutdown(graceful=True)
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        refs = [
            rpc.rpc_async(worker_reff.owner(),
                          call_method,
                          (worker_reff, EnvWorker.render, "rgb_array"))
            for worker_reff in self.worker_reffs
        ]
        imgs = [ref.wait() for ref in refs]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_reffs = self._get_target_remotes(indices)
        refs = [
            rpc.rpc_async(worker_reff.owner(),
                          call_method,
                          (worker_reff, EnvWorker.get_attr, attr_name))
            for worker_reff in target_reffs
        ]
        return [ref.wait() for ref in refs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_reffs = self._get_target_remotes(indices)
        refs = [
            rpc.rpc_async(worker_reff.owner(),
                          call_method,
                          (worker_reff, EnvWorker.set_attr, attr_name, value))
            for worker_reff in target_reffs
        ]
        for ref in refs:
            ref.wait()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_reffs = self._get_target_remotes(indices)
        refs = [
            rpc.rpc_async(worker_reff.owner(),
                          call_method,
                          (worker_reff, EnvWorker.env_method,
                           method_name, method_args, method_kwargs))
            for worker_reff in target_reffs
        ]
        return [ref.wait() for ref in refs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        raise NotImplementedError

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.worker_reffs[i] for i in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, torch.stack([o[k] for o in obs])).numpy() for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(
            obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((torch.stack([o[i] for o in obs]).numpy() for i in range(obs_len)))
    else:
        return torch.stack(obs).numpy()


if __name__ == "__main__":
    import gym

    env = RpcVecEnv([lambda: gym.make("LunarLander-v2")]*2, "fork")
    print(env.reset().shape)
    print(env.step(np.ones((2,), dtype=np.long)))
