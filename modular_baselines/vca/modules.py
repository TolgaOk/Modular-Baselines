import torch
from abc import ABC, abstractmethod


class CategoricalPolicyModule(torch.nn.Module):
    def __init__(self, insize, actsize, hidden_size, tau=1):
        super().__init__()

        self.insize = insize
        self.actsize = actsize

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
        soft_sample = torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self.tau, hard=False)

        entropy = torch.distributions.categorical.Categorical(
            logits=logits).entropy()

        return acts - soft_sample.detach() + soft_sample, entropy


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
    def dist(self):
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
                 tau: float = 1.):
        super().__init__(insize=insize,
                         actsize=actsize,
                         hidden_size=hidden_size)
        self.tau = tau
        assert len(state_set.shape) == 1, ""
        self.state_set = state_set

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
        soft_sample = torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self.tau, hard=False)

        return soft_sample

    def reparam(self, obs, soft_sample):
        return obs - soft_sample.detach() + soft_sample


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
