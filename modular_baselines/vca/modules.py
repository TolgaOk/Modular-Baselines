import torch
from abc import ABC, abstractmethod


class CategoricalPolicyModule(torch.nn.Module):
    def __init__(self, insize, actsize, hidden_size, tau=1, use_gumbel=False):
        super().__init__()

        self.insize = insize
        self.actsize = actsize
        self.use_gumbel = use_gumbel

        self.net = torch.nn.Sequential(
            torch.nn.Linear(insize, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, actsize))
        self.tau = tau

    def forward(self, x):
        logits = self.net(x)

        hard_sample = torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self.tau, hard=True)
        return hard_sample

    def reparam_act(self, obs, acts):
        logits = self.net(obs)
        entropy = torch.distributions.categorical.Categorical(
            logits=logits).entropy()
        if self.use_gumbel:
            return self._gumbel(logits, acts), entropy
        return straight_through_reparam(logits, acts), entropy        

    def _gumbel(self, logits, acts):
        soft_sample = torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self.tau, hard=False)
        return acts - soft_sample.detach() + soft_sample


class BaseTransitionModule(torch.nn.Module, ABC):
    def __init__(self,
                 insize: int,
                 actsize: int,
                 hidden_size: int = 16):
        super().__init__()

        self.insize = insize
        self.actsize = actsize
        self.hidden_size = hidden_size

        self.init_network()

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def reparam(self):
        pass


class ContinuousTransitionModule(BaseTransitionModule):

    def init_network(self):
        self.act_fc = torch.nn.Linear(self.actsize, self.hidden_size)
        self.hidden_fc = torch.nn.Linear(self.insize, self.hidden_size)
        self.gru = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out_mean = torch.nn.Linear(self.hidden_size, self.insize)
        self.out_std = torch.nn.Linear(self.hidden_size, self.insize)

    def forward(self, obs, act):

        act = torch.relu(self.act_fc(act))
        obs = torch.relu(self.hidden_fc(obs))
        hidden = self.gru(act, obs)

        mean = self.out_mean(hidden)
        std = torch.nn.functional.softplus(self.out_std(hidden))
        return mean, std

    def dist(self, obs, act):
        mean, std = self(obs, act)
        dist = torch.distributions.normal.Normal(mean, std)

        return dist

    def reparam(self, sample, mean, std):
        diffable_sample = mean + std * \
            ((sample - mean) / (std + 1e-7)).detach()
        return diffable_sample


class CategoricalTransitionModule(BaseTransitionModule):

    def __init__(self,
                 insize: int,
                 actsize: int,
                 state_set: torch.LongTensor,
                 hidden_size: int = 16,
                 tau: float = 1.,
                 use_gumbel=False):
        super().__init__(insize=insize,
                         actsize=actsize,
                         hidden_size=hidden_size)
        self.tau = tau
        assert len(state_set.shape) == 1, ""
        self.state_set = state_set
        self.use_gumbel = use_gumbel

    def init_network(self):
        self.act_fc = torch.nn.Linear(self.actsize, self.hidden_size)
        self.hidden_fc = torch.nn.Linear(self.insize, self.hidden_size)
        self.gru = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out_logit = torch.nn.Linear(self.hidden_size, self.insize)

    def forward(self, obs, act):
        act = torch.relu(self.act_fc(act))
        obs = torch.relu(self.hidden_fc(obs))
        hidden = self.gru(act, obs)
        logits = self.out_logit(hidden)
        return logits

    def dist(self, obs, act):
        logits = self(obs, act)
        return logits

    def reparam(self, obs, logits):
        if self.use_gumbel:
            return self._gumbel(obs, logits)
        return straight_through_reparam(logits, obs)

    def _gumbel(self, obs, logits):
        soft_sample = torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self.tau, hard=False)
        return obs - soft_sample.detach() + soft_sample


class MultiheadCatgoricalTransitionModule(CategoricalTransitionModule):

    def init_network(self):
        self.fc = torch.nn.Linear(self.insize, self.hidden_size)
        self.head = torch.nn.Linear(self.hidden_size, self.insize * self.actsize)

    def _forward(self, obs):
        hidden = torch.relu(self.fc(obs))
        logits = self.head(hidden).reshape(-1, self.actsize, self.insize)
        return logits

    def forward(self, obs, act):
        logits = self._forward(obs)
        return torch.einsum("bao,ba->bo", logits, act)

    def dist(self, obs, act):
        logits = self._forward(obs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.einsum("bao,ba->bo", probs, act)
        return probs
        
    def reparam(self, obs, probs):
        if self.use_gumbel:
            raise NotImplementedError
        assert len(probs.shape) == 2, "Logits must be 2D"
        return obs + probs - probs.detach()


class CategoricalRewardModule(torch.nn.Module):
    def __init__(self,
                 insize,
                 reward_set,
                 hidden_size=16,
                 tau=1):
        super().__init__()
        self.tau = tau
        self.insize = insize
        reward_set = torch.as_tensor(reward_set)
        if len(reward_set.shape) == 1:
            reward_set = reward_set.reshape(1, -1)
        self.reward_set = reward_set

        self.net = torch.nn.Sequential(
            torch.nn.Linear(insize, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, len(reward_set.reshape(-1))))

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.categorical.Categorical(logits=logits)

    def expected(self, next_state):
        assert len(self.reward_set.shape) == 2, ""
        assert self.reward_set.shape[0] == 1, ""
        dist = self(next_state)
        return (self.reward_set * dist.probs).mean(-1, keepdims=True)


def straight_through_reparam(logits, onehot_sample):
    assert len(logits.shape) == 2, "Logit is not 2D but {}D".format(len(logits.shape))
    assert logits.shape == onehot_sample.shape, "Shape mismatch"
    probs = torch.nn.functional.softmax(logits, dim=1)
    r_prob = probs
    return onehot_sample + r_prob - r_prob.detach()