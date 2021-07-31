import unittest
import numpy as np
from modular_baselines.algorithms.advantages import calculate_gae


class TestCalculateGae(unittest.TestCase):

    def test_assertions(self):
        rewards = np.ones((1, 5, 1))
        terminations = np.zeros((1, 5, 1))
        values = np.zeros((1, 5, 1))
        last_value = np.zeros((1,))
        self.assertRaises(AssertionError,
                          calculate_gae,
                          rewards,
                          terminations,
                          values,
                          last_value,
                          1.0,
                          1.0)

    def test_shape(self):
        rewards = np.ones((1, 5, 1))
        terminations = np.zeros((1, 5, 1))
        values = np.zeros((1, 5, 1))
        last_value = np.zeros((1, 1))

        advantages, returns = calculate_gae(
            rewards,
            terminations,
            values,
            last_value,
            0.5,
            1.0)

        for array in (advantages, returns):
            self.assertTrue(array.shape == values.shape)

    def test_td(self):
        rewards = np.arange(5).reshape(1, -1, 1)
        terminations = np.array([0, 0, 1, 0, 0]).reshape(1, -1, 1)
        values = np.ones((1, 5, 1))
        last_value = np.ones((1, 1)) * -2

        advantages, returns = calculate_gae(
            rewards,
            terminations,
            values,
            last_value,
            0.5,
            0.0)

        self.assertTrue(np.all(
            advantages[0, :, 0] == np.array([-0.5, 0.5, 1, 2.5, 2], dtype=np.float32)
        ))
        self.assertTrue(np.all(
            returns[0, :, 0] == np.array([0.5, 1.5, 2, 3.5, 3], dtype=np.float32)
        ))

    def test_lambda(self):
        rewards = np.arange(6).reshape(1, -1, 1) - 2
        terminations = np.array([0, 1, 0, 0, 0, 1]).reshape(1, -1, 1)
        values = 3 - np.arange(6).reshape(1, -1, 1)
        last_value = np.ones((1, 1))

        advantages, returns = calculate_gae(
            rewards,
            terminations,
            values,
            last_value,
            0.5,
            1.0)

        # print(advantages)
        self.assertTrue(np.all(
            advantages[0, :, 0] == np.array([-5.5, -3, 0.375, 2.75, 4.5, 5],
                                            dtype=np.float32)
        ))
        self.assertTrue(np.all(
            returns[0, :, 0] == np.array([-2.5, -1, 1.375, 2.75, 3.5, 3],
                                         dtype=np.float32)
        ))

        advantages, returns = calculate_gae(
            rewards,
            terminations,
            values,
            last_value,
            0.5,
            0.1)

        self.assertTrue(np.all(
            advantages[0, :, 0] == np.array([-4.15, -3, -0.969375, 0.6125, 2.25, 5],
                                            dtype=np.float32)
        ))
