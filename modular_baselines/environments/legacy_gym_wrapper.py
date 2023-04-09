from typing import Any
import numpy as np
import gym



class LegacyWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env=env)

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, rewards, termination, truncation, infos = self.env.step(action)
        return obs, rewards, termination or truncation, infos

    def seed(self, seed: int) -> None:
        raise RuntimeError("You cannot seed the environment via this method!")

