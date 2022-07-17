from typing import Any, List, Dict, Tuple
import numpy as np


class PrePoolEnvWrapper():
    """ Wrapper for envpool environments before applying SB3 wrappers to them.

    Args:
        pool_env (Any): _description_
    """

    def __init__(self, pool_env: Any):
        self.venv = pool_env
        self.num_envs = pool_env.config["num_envs"]
        self.all_rewards = np.zeros(self.num_envs)
        self.all_lengths = np.zeros(self.num_envs)
        self.observation_space = pool_env.observation_space
        self.action_space = pool_env.action_space

        self._actions = None

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_async(self, actions) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        state, reward, done, _ = self.venv.step(self._actions)
        self.all_rewards += reward
        self.all_lengths += 1
        info = [{"episode": {"r": self.all_rewards[index], "l": self.all_lengths[index]}} if _done else {}
                for index, _done in enumerate(done)]
        self.all_rewards *= (1 - done)
        self.all_lengths *= (1 - done)
        return state, reward, done, info
