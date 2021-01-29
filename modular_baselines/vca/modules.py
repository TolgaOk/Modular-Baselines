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
            # torch.nn.Tanh(),
            # torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, actsize))
        self.tau = tau

    def forward(self, x):
        hard_sample = self.dist(x).sample()
        return hard_sample

    def dist(self, x):
        logits = self.net(x)
        return torch.distributions.categorical.Categorical(
            logits=logits)

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
                 hidden_size: int = 32):
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


class MultiHeadContinuousTransitionModule(BaseTransitionModule):

    def init_network(self):
        self.hidden_in = torch.nn.Linear(
            self.insize, self.hidden_size, bias=True)
        self.hidden_out = torch.nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.forget_gate = torch.nn.Linear(self.hidden_size, self.insize)

        self.out_mean = torch.nn.Linear(
            self.hidden_size, self.insize * self.actsize, bias=True)
        self.out_std = torch.nn.Linear(
            self.hidden_size, self.insize * self.actsize, bias=True)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        assert len(act.shape) == 2
        assert act.shape[1] == self.actsize

        in_hidden = torch.relu(self.hidden_in(obs))
        out_hidden = torch.relu(self.hidden_out(in_hidden))

        forget = torch.sigmoid(self.forget_gate(in_hidden))
        # forget = forget.reshape(-1, self.actsize, self.insize)
        # forget = torch.einsum("bas,ba->bs", forget, act)

        # out_hidden = out_hidden * hidden_forget + (1 - hidden_forget) * in_hidden

        residual_mean = self.out_mean(
            out_hidden).reshape(-1, self.actsize, self.insize)
        # std = torch.nn.functional.softplus(self.out_std(out_hidden)).reshape(-1, self.actsize, self.insize)

        residual_mean = torch.einsum("bas,ba->bs", residual_mean, act)
        # std = torch.einsum("bas,ba->bs", std, act)

        mean = residual_mean * forget + (1 - forget) * obs
        # mean = residual_mean + obs
        std = torch.ones_like(mean) * 0.01

        return mean, std

    def gate_output(self, obs: torch.Tensor, act: torch.Tensor):
        in_hidden = torch.relu(self.hidden_in(obs))

        forget = torch.sigmoid(self.forget_gate(in_hidden))

        return forget

    def dist(self, obs, act):
        mean, std = self(obs, act)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

    def reparam(self, sample, dist):
        mean = dist.loc
        std = dist.scale
        diffable_sample = mean + std * \
            ((sample - mean) / (std + 1e-7)).detach()
        return diffable_sample


class SoftLinearMultiHeadTransitionModule(MultiHeadContinuousTransitionModule):

    def init_network(self):
        self.hidden = SoftedParameterLinearLayer(self.insize, self.hidden_size)
        self.out_mean = SoftedParameterLinearLayer(
            self.hidden_size, self.insize * self.actsize)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        mean = self.out_mean(self.hidden(obs))
        mean = mean.reshape(-1, self.actsize, self.insize)
        mean = torch.einsum("bas,ba->bs", mean, act)

        std = torch.ones_like(mean) * 0.01

        return mean, std

    def gate_output(self, obs: torch.Tensor, act: torch.Tensor):
        return obs


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
        self.head = torch.nn.Linear(
            self.hidden_size, self.insize * self.actsize)

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


class FullCategoricalTransitionModule(MultiheadCatgoricalTransitionModule):

    def init_network(self):
        self.logits = torch.nn.Parameter(
            torch.randn(self.insize, self.actsize, self.insize))

    @staticmethod
    def _forward(logits, obs, act):
        return torch.einsum("san,bs,ba->bn", logits, obs, act)

    def forward(self, obs, act):
        return self._forward(self.logits, obs, act)

    def dist(self, obs, act):
        probs = torch.nn.functional.softmax(self.logits, dim=-1)
        return self._forward(probs, obs, act)

    def reparam(self, obs, probs):
        if self.use_gumbel:
            raise NotImplementedError
        assert len(probs.shape) == 2, "Logits must be 2D"
        return obs + probs - probs.detach()


def straight_through_reparam(logits, onehot_sample):
    assert len(logits.shape) == 2, "Logit is not 2D but {}D".format(
        len(logits.shape))
    assert logits.shape == onehot_sample.shape, "Shape mismatch"
    probs = torch.nn.functional.softmax(logits, dim=1)
    r_prob = probs
    return onehot_sample + r_prob - r_prob.detach()


class SoftedParameterLinearLayer(torch.nn.Linear):

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        # y = xA^T + b
        # Shape of A -> (d', d) where d' denotes the output dimension
        soft_weight = torch.nn.functional.softmax(self.weight, dim=1)
        return torch.nn.functional.linear(input, soft_weight, self.bias)
