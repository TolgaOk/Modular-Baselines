import numpy as np
import torch
import unittest
import gym

from modular_baselines.buffers.buffer import GeneralBuffer


class TestBuffers(unittest.TestCase):

    def setUp(self):
        env = gym.make("Pong-ramDeterministic-v4")
        self.env = env
        self.buffer_size = 10000
        self.n_envs = 4
        self.buffer = GeneralBuffer(buffer_size=self.buffer_size,
                                    observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    device="cpu",
                                    n_envs=self.n_envs,
                                    optimize_memory_usage=False)

    def test_shapes(self):
        self.assertTrue(
            self.buffer.observations.shape,
            (self.buffer_size, self.n_envs, self.env.observation_space.shape[0]))

        self.assertTrue(
            self.buffer.actions.shape,
            (self.buffer_size, self.n_envs, self.env.action_space.n))

        self.assertTrue(
            self.buffer.rewards.shape,
            (self.buffer_size, self.n_envs, 1))

        self.assertTrue(
            self.buffer.dones.shape,
            (self.buffer_size, self.n_envs, 1))

if __name__ == "__main__":
    unittest.main()
