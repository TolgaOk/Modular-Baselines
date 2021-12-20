from logging import log
import unittest
import numpy as np
import torch
from torch.autograd import grad

from modular_baselines.algorithms.a2c import TorchA2CPolicy


class Policy(TorchA2CPolicy, torch.nn.Module):

    def __init__(self, insize, outsize, lr) -> None:
        torch.nn.Module.__init__(self)
        self.pi_params = torch.nn.Parameter(torch.ones(insize, outsize))
        self.value_params = torch.nn.Parameter(torch.ones(insize, 1))
        self._optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    @property
    def device(self):
        return "cpu"

    @property
    def optimizer(self):
        return self._optimizer

    def init_state(self, batch_size=None):
        return None

    def evaluate_rollout(self, observation, _, action, last_next_obseration):
        observations = torch.cat([observation, last_next_obseration.unsqueeze(1)], dim=1)
        logit = torch.einsum("brd,df->brf",
                             observations,
                             self.pi_params)[:, :-1]
        values = torch.einsum("brd,df->brf",
                              observations,
                              self.value_params)
        value, last_value = values[:, :-1], values[:, -1]
        dist = torch.distributions.Categorical(probs=(logit/2))
        dist.probs = (logit/2)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return value, log_prob, entropy, last_value

    def sample_action(self):
        return None


class TestTorchA2CPolicy(unittest.TestCase):

    def setUp(self) -> None:
        obs = np.eye(10, dtype=np.float32).reshape(10, 1, 10).repeat(5, axis=1)
        next_obs = np.zeros((10, 5, 10), dtype=np.float32)
        acts = np.random.randint(0, 2, size=(10, 5, 1))
        acts[-1] = 0
        acts[-2] = 1
        reward = np.arange(5, dtype=np.float32).reshape(1, -1, 1).repeat(10, axis=0)
        termination = np.array([0, 0, 1, 0, 0], dtype=np.float32).reshape(
            1, -1, 1).repeat(10, axis=0)
        struct = np.dtype([
            ("observation", np.float32, (10,)),
            ("next_observation", np.float32, (10,)),
            ("action", np.float32, (1,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
        ])
        sample = np.zeros((10, 5), dtype=struct)
        sample["observation"] = obs
        sample["next_observation"] = next_obs
        sample["action"] = acts
        sample["reward"] = reward
        sample["termination"] = termination

        self.sample = sample
        self.lr = 1.2
        self.advantages = np.array([-0.44, 0.6, 1, 2.8, 3], dtype=np.float32)

    def test_log_pi(self):
        policy = Policy(10, 2, lr=self.lr)
        policy.update_parameters(self.sample,
                                 value_coef=0.8,
                                 ent_coef=0,
                                 gamma=0.5,
                                 gae_lambda=0.2,
                                 max_grad_norm=100,)
        grad_pi = np.zeros_like(policy.pi_params.detach().numpy())

        acts = self.sample["action"]
        normalizer = self.lr / len(acts)  # mean at the batch dimension
        grad_pi[:, 1] = (acts.astype(np.float32) *
                         self.advantages.reshape(1, -1, 1)).mean(1).flatten()
        grad_pi[:, 0] = ((1 - acts.astype(np.float32)) *
                         self.advantages.reshape(1, -1, 1)).mean(1).flatten()
        grad_pi = grad_pi * normalizer

        self.assertTrue(np.allclose(grad_pi, (policy.pi_params.detach().numpy() - 1)))

    def test_entropy_pi(self):
        policy = Policy(10, 2, lr=self.lr)
        ent_coef = 1.1
        policy.update_parameters(self.sample,
                                 value_coef=0.8,
                                 ent_coef=ent_coef,
                                 gamma=0.5,
                                 gae_lambda=0.2,
                                 max_grad_norm=100,)
        grad_pi = np.zeros_like(policy.pi_params.detach().numpy())

        acts = self.sample["action"]
        normalizer = self.lr / len(acts)
        grad_pi[:, 1] = (acts.astype(np.float32) *
                         self.advantages.reshape(1, -1, 1)).mean(1).flatten()
        grad_pi[:, 0] = ((1 - acts.astype(np.float32)) *
                         self.advantages.reshape(1, -1, 1)).mean(1).flatten()
        grad_pi = grad_pi * normalizer

        self.assertTrue(np.allclose(
            -np.ones_like(grad_pi) * (np.log(0.5) + 1) / 2 * self.lr * ent_coef / len(acts),
            (policy.pi_params.detach().numpy() - grad_pi - 1)))

    def test_value(self):
        returns = self.advantages + 1 # values equal to 1
        policy = Policy(10, 2, lr=self.lr)
        policy.update_parameters(self.sample,
                                 value_coef=1.0,
                                 ent_coef=0.1,
                                 gamma=0.5,
                                 gae_lambda=0.2,
                                 max_grad_norm=100,)
        grad_value = 2 * (1 - returns).mean() * self.lr / len(self.sample["action"])
        self.assertTrue(np.allclose(
            -grad_value,
            policy.value_params.detach().numpy() - 1))
