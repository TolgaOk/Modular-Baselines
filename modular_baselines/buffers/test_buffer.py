import unittest
import numpy as np
from modular_baselines.buffers.buffer import Buffer


class TestBuffer(unittest.TestCase):

    def setUp(self) -> None:
        self.struct = np.dtype([
            ("state", np.float32, (20,)),
            ("next_state", np.float32, (20,)),
            ("action", np.int32, (4,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
        ])

    def test_init(self):
        buffer = Buffer(struct=self.struct, capacity=10, num_envs=2)
        self.assertEqual(buffer.size, 0)
        self.assertEqual(buffer.buffer.shape, (10, 2))
        self.assertFalse(buffer.full)

    def test_push_assertion(self):
        buffer = Buffer(struct=self.struct, capacity=10, num_envs=2)
        # Bad termination shape
        self.assertRaises(AssertionError,
                          buffer.push,
                          {"state": np.ones((5, 20)),
                           "next_state": np.ones((5, 20)),
                           "action": np.ones((5, 4)),
                           "reward": np.ones((5, 1)),
                           "termination": np.ones((5,)),
                           })
        # Bad field name
        self.assertRaises(AssertionError,
                          buffer.push,
                          {"state": np.ones((5, 20)),
                           "next_state": np.ones((5, 20)),
                           "action": np.ones((5, 4)),
                           "reward": np.ones((5, 1)),
                           "termion": np.ones((5,)),
                           })
        # Missing field name
        self.assertRaises(AssertionError,
                          buffer.push,
                          {"state": np.ones((5, 20)),
                           "next_state": np.ones((5, 20)),
                           "action": np.ones((5, 4)),
                           "reward": np.ones((5, 1)),
                           })
        # No env axis field name
        self.assertRaises(AssertionError,
                          buffer.push,
                          {"state": np.ones((20,)),
                           "next_state": np.ones((20,)),
                           "action": np.ones((4,)),
                           "reward": np.ones((1,)),
                           "termination": np.ones((1,)),
                           })

    def test_push_cycle(self):
        buffer = Buffer(struct=self.struct, capacity=10, num_envs=2)
        for index in range(12):
            self.assertEqual(buffer.full, index >= 10)
            buffer.push({"state": np.ones((2, 20)),
                         "next_state": np.ones((2, 20)),
                         "action": np.ones((2, 4)),
                         "reward": np.ones((2, 1)) * index,
                         "termination": np.ones((2, 1)),
                         })
        self.assertEqual(buffer.buffer["reward"][0].mean(), 10.0)

    def test_sample_assertion(self):
        buffer = Buffer(struct=self.struct, capacity=50, num_envs=2)
        for _ in range(25):
            buffer.push({"state": np.ones((2, 20)),
                         "next_state": np.ones((2, 20)),
                         "action": np.ones((2, 4)),
                         "reward": np.ones((2, 1)),
                         "termination": np.ones((2, 1)),
                         })
        self.assertRaises(AssertionError,
                          buffer.sample,
                          batch_size=1000,
                          rollout_len=26)
        self.assertRaises(AssertionError,
                          buffer.sample,
                          batch_size=1000,
                          rollout_len=5,
                          sampling_length=26)
        self.assertRaises(AssertionError,
                          buffer.sample,
                          batch_size=1000,
                          rollout_len=5,
                          sampling_length=1)
        self.assertRaises(AssertionError,
                          buffer.sample,
                          batch_size=1000,
                          rollout_len=5,
                          sampling_length=-1)

    def test_sample_rollout(self):
        buffer = Buffer(struct=self.struct, capacity=5, num_envs=2)
        actions = [np.arange(8).reshape(2, 4) + 20 * index for index in range(5)]
        for action in actions:
            buffer.push({"state": np.ones((2, 20)),
                         "next_state": np.ones((2, 20)),
                         "action": action,
                         "reward": np.ones((2, 1)),
                         "termination": np.ones((2, 1)),
                         })
        # batch_size = #env, rollout_len=buffer.capacity
        sample = buffer.sample(batch_size=2, rollout_len=5)
        self.assertTrue(np.all(sample["action"] == np.stack(actions, axis=1)))

    def test_sample_valid_items(self):
        buffer = Buffer(struct=self.struct, capacity=50, num_envs=2)
        for _ in range(25):
            buffer.push({"state": -np.ones((2, 20)),
                         "next_state": -np.ones((2, 20)),
                         "action": -np.ones((2, 4)),
                         "reward": -np.ones((2, 1)),
                         "termination": -np.ones((2, 1)),
                         })
        sample = buffer.sample(batch_size=1000, rollout_len=5)
        for field in self.struct.names:
            self.assertTrue(np.all(sample[field] == -1))

    def test_sample_integrity(self):
        buffer = Buffer(struct=self.struct, capacity=50, num_envs=2)
        for index in range(75):
            buffer.push({"state": np.ones((2, 20)),
                         "next_state": np.ones((2, 20)),
                         "action": np.ones((2, 4)),
                         "reward": np.ones((2, 1)) * index,
                         "termination": np.ones((2, 1)),
                         })

        sample = buffer.sample(batch_size=1000, rollout_len=20)
        self.assertTrue(np.all(
            sample["reward"] - sample["reward"][:, 0].reshape(-1, 1, 1) ==
            np.arange(20).reshape(1, -1, 1)))

    def test_sample_with_sampling_length(self):
        buffer = Buffer(struct=self.struct, capacity=50, num_envs=2)
        for index in range(75):
            buffer.push({"state": np.ones((2, 20)),
                         "next_state": np.ones((2, 20)),
                         "action": np.ones((2, 4)),
                         "reward": np.ones((2, 1)) * index,
                         "termination": np.ones((2, 1)),
                         })
        sample = buffer.sample(batch_size=1000, rollout_len=10, sampling_length=20)
        self.assertTrue(np.all(sample["reward"] >= 75 - 20))

        sample = buffer.sample(batch_size=1000, rollout_len=5, sampling_length=5)
        self.assertTrue(np.all(sample["reward"] - 70 == np.arange(5).reshape(1, -1, 1)))


if __name__ == '__main__':
    unittest.main()
