import numpy as np
import torch
import unittest
import gym

from stable_baselines3.common.env_util import make_vec_env

from modular_baselines.buffers.buffer import GeneralBuffer


class TestGeneralBuffer(unittest.TestCase):

    def setUp(self):
        self.n_env = 12
        self.env = make_vec_env("LunarLander-v2", self.n_env)
        self.buffer_name_space = [
            "observations", "next_observations", "actions",
            "rewards", "dones", "values", "log_probs", "advantages", "returns"]

    def _loop_fill_buffer(self, buffer, size):
        action_size = self.env.action_space.n
        state = self.env.reset()
        for i in range(size):
            act = np.random.randint(0, action_size, (self.n_env, 1))
            next_state, reward, done, _ = self.env.step(act.squeeze())
            buffer.add(
                state,
                next_state,
                act,
                reward,
                done,
                torch.ones(self.n_env, 1),
                torch.ones(self.n_env))

    def test_add(self):
        buffer = GeneralBuffer(20,
                               self.env.observation_space,
                               self.env.action_space,
                               n_envs=self.n_env)
        # Initial conditions
        self.assertEqual(buffer.pos, 0)
        self.assertEqual(buffer.size(), 0)

        self._loop_fill_buffer(buffer, 5)
        # Test for moving +5 steps
        self.assertEqual(buffer.pos, 5)
        self.assertEqual(buffer.size(), 5)

        for name in self.buffer_name_space:
            # Test for not filling the unseen indices
            self.assertTrue(np.all(getattr(buffer, name)[5:] == 0))
        # Test for newly added observations
        self.assertTrue(np.any(buffer.observations[:5] != 0))

        # Save newly added observations for testing
        obs = buffer.observations[:5].copy()

        self._loop_fill_buffer(buffer, 25)
        # Test for position and size
        self.assertEqual(buffer.pos, (25 + 5) % 20)
        self.assertEqual(buffer.size(), 20)

        # Test for overwriting
        self.assertFalse(np.all(obs == buffer.observations[:5]))

    def test_sample(self):
        pass

    def test_get_rollout(self):
        rollout_size = 5
        buffer = GeneralBuffer(20,
                               self.env.observation_space,
                               self.env.action_space,
                               n_envs=self.n_env)
        self._loop_fill_buffer(buffer, 11)
        buffer.actions[5:10] = np.ones_like(buffer.actions[5:10])

        rollout = buffer.get_rollout(rollout_size, batch_size=None)
        self.assertTrue(np.all(next(rollout).actions.numpy() == 1))

    def test_compute_returns_and_advantage(self):
        rollout_size = 5
        gamma = 0.5
        gae_lambda = 0.5
        buffer = GeneralBuffer(20,
                               self.env.observation_space,
                               self.env.action_space,
                               n_envs=self.n_env)
        self._loop_fill_buffer(buffer, 11)
        buffer.rewards[5:10] = np.ones_like(buffer.rewards[5:10])
        buffer.values[5:10] = np.ones_like(buffer.values[5:10])
        buffer.dones[5:10] = np.zeros_like(buffer.dones[5:10])
        buffer.dones[9] += 1

        buffer.compute_returns_and_advantage(
            rollout_size=rollout_size,
            gamma=gamma,
            gae_lambda=gae_lambda)

        adv = buffer.advantages[5:10]
        for index, value in enumerate([0, 0.5, 0.625, 0.65625, 0.6640625]):
            self.assertEqual(adv[rollout_size - index - 1][0], value)

if __name__ == "__main__":
    unittest.main()
