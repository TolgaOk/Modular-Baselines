import torch
import numpy as np
import time
import gym
from typing import Any, Callable, Dict, List, Optional, Union

from modular_baselines.algorithms.callbacks import BaseAlgorithmCallback
from modular_baselines.vca.collector import make_onehot
from modular_baselines.vca.modules import CategoricalPolicyModule
from maze.environment import MazeEnv


def process_state(state, maxsize):
    state = torch.tensor([[state]])
    return make_onehot(state, maxsize)


def render(policy_param_path, fps=60):

    env = MazeEnv()
    policy_m = CategoricalPolicyModule(
        insize=env.observation_space.n,
        actsize=env.action_space.n,
        hidden_size=32)
    torch.load(policy_param_path)

    state = env.reset()
    done = False
    env.render()
    time.sleep(1)
    while not done:
        act = policy_m(process_state(state,
                                     env.observation_space.n)).argmax()
        state, reward, done, info = env.step(act)
        env.render()
        time.sleep(1/fps)


class LogPlaybackData(BaseAlgorithmCallback):
    def __init__(self,
                 file_name: str,
                 env: gym.Env,
                 log_interval: int = 100,):
        super().__init__()
        self.file_name = file_name
        self.env = env

    def _on_training_start(self, *args) -> None:
        pass

    def _on_step(self, locals_) -> bool:
        # Measure action_grad
        # Measure state_grad
        # Save renders
        pass

    def _on_training_end(self, *args) -> None:
        pass
