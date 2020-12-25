import numpy as np
import torch
import unittest
import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from modular_baselines.buffers.buffer import RolloutBuffer
from modular_baselines.collectors.colletor import OnPolicyCollector


class Policy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones(x.shape[0]).long(), torch.ones(x.shape[0]), torch.ones(x.shape[0])


class TestOnPolicyCollector(unittest.TestCase):

    def setUp(self):
        env = gym.make("Pong-ramDeterministic-v4")
        self.buffer_size = 20
        self.n_envs = 4
        buffer = RolloutBuffer(buffer_size=self.buffer_size,
                               observation_space=env.observation_space,
                               action_space=env.action_space,
                               device="cpu",
                               gae_lambda=1,
                               gamma=0.99,
                               n_envs=self.n_envs)
        vecenv = make_vec_env(lambda: gym.make("Pong-ramDeterministic-v4"),
                              n_envs=self.n_envs,
                              vec_env_cls=SubprocVecEnv)

        self.collector = OnPolicyCollector(vecenv,
                                           buffer=buffer,
                                           policy=Policy())

    def test_collect(self):
        n_steps = 5
        self.collector.collect(n_steps)
        self.assertEqual(self.collector.buffer.pos, n_steps)



if __name__ == "__main__":
    unittest.main()
