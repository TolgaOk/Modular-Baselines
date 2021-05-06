from itertools import chain
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

from modular_baselines.contrib.jacobian_trace.type_aliases import LatentTuple


class DenseModel(nn.Module):
    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ELU):
        super().__init__()
        self._output_size = output_size
        self._layers = layers
        self._hidden_size = hidden_size
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for _ in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, features):
        return self.model(features)


class NormalDistributionModel(DenseModel):

    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ReLU):
        super().__init__(feature_size,
                         output_size * 2,
                         layers,
                         hidden_size,
                         activation)

    def forward(self, features):
        mean, std = torch.chunk(super().forward(features), 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist

    @staticmethod
    def reparam(dist, sample):
        return dist.loc + dist.scale * ((sample - dist.loc)/(dist.scale + 1e-7)).detach()

    @staticmethod
    def detach_dist(dist):
        return torch.distributions.Normal(
            dist.loc.detach(),
            dist.scale.detach()
        )

class ObservationEncoder(DenseModel):
    pass


class ObservationDecoder(NormalDistributionModel):
    def forward(self, latent_tuple: LatentTuple):
        features = self.make_feature(latent_tuple)
        return super().forward(features)

    @staticmethod
    def make_feature(latent_tuple: LatentTuple):
        return torch.cat([latent_tuple.rsample, latent_tuple.embedding], dim=-1)


class StateDistribution(NormalDistributionModel):
    pass


class DiscreteActionTransitionDistribution(NormalDistributionModel):

    def __init__(self,
                 feature_size: int,
                 output_size: int,
                 action_size: int,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ReLU):
        self._action_size = action_size
        super().__init__(feature_size,
                         output_size * action_size,
                         layers,
                         hidden_size,
                         activation)

    def forward(self, latent_tuple: LatentTuple, actions: torch.Tensor):
        actions = DiscreteActor.maybe_onehot(actions, self._action_size)

        features = self.make_feature(latent_tuple)
        logits = DenseModel.forward(self, features).reshape(features.shape[0], self._action_size, -1)
        selected_logits = torch.einsum("baf,ba->bf", logits, actions)

        mean, std = torch.chunk(selected_logits, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist

    @staticmethod
    def make_feature(latent_tuple: LatentTuple):
        return torch.cat([latent_tuple.rsample, latent_tuple.embedding], dim=-1)


class DiscreteActor(DenseModel):

    def forward(self, latent_tuple: LatentTuple, onehot: bool = False):
        features = self.make_feature(latent_tuple)
        act_logits = super().forward(features)
        if onehot:
            return torch.distributions.one_hot_categorical.OneHotCategorical(logits=act_logits)
        return torch.distributions.categorical.Categorical(logits=act_logits)

    @staticmethod
    def make_feature(latent_tuple: LatentTuple):
        return torch.cat([latent_tuple.rsample, latent_tuple.embedding], dim=-1)

    @staticmethod
    def reparam(act_dist, actions: torch.Tensor):
        actions = DiscreteActor.maybe_onehot(actions, act_dist.probs.shape[-1])
        return actions + act_dist.probs - act_dist.probs.detach()

    @staticmethod
    def maybe_onehot(tensor: torch.Tensor, maxsize: int):
        if len(tensor.shape) == 2 and tensor.shape[1] != 1:
            return tensor
        return (torch.arange(maxsize, device=tensor.device).reshape(1, -1)
                == tensor.reshape(-1, 1)).float()

class Critic(DenseModel):

    def forward(self, latent_tuple: LatentTuple):
        features = self.make_feature(latent_tuple)
        return super().forward(features)

    @staticmethod
    def make_feature(latent_tuple: LatentTuple):
        return torch.cat([latent_tuple.rsample, latent_tuple.embedding], dim=-1)


class JTACModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 action_size,
                 model_lr=4e-4,
                 ac_lr: float = 7e-4,
                 state_size=32,
                 latent_size=200,
                 encoder_layers=2,
                 encoder_hidden_size=200,
                 decoder_layers=2,
                 decoder_hidden_size=200,
                 state_layers=2,
                 state_hidden_size=200,
                 trans_layers=2,
                 trans_hidden_size=200,
                 actor_layers=1,
                 actor_hidden_size=200,
                 critic_layers=1,
                 critic_hidden_size=200):
        super().__init__()
        self.encoder = ObservationEncoder(
            input_size,
            latent_size,
            encoder_layers,
            encoder_hidden_size)
        self.decoder = ObservationDecoder(
            latent_size + state_size,
            input_size,
            decoder_layers,
            decoder_hidden_size)
        self.state_dist = StateDistribution(
            latent_size,
            state_size,
            state_layers,
            state_hidden_size)
        self.transition_dist = DiscreteActionTransitionDistribution(
            latent_size + state_size,
            state_size,
            action_size,
            trans_layers,
            trans_hidden_size)

        self.actor = DiscreteActor(
            state_size + latent_size,
            action_size,
            actor_layers,
            actor_hidden_size)
        self.critic = Critic(
            state_size + latent_size,
            1,
            critic_layers,
            critic_hidden_size)

        self.model_optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters(),
                  self.state_dist.parameters(), self.transition_dist.parameters()),
            lr=model_lr)
        self.ac_optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=ac_lr)

    def get_latent(self, observation):
        observation = self._preprocess(observation)
        embedding = self.encoder(observation)
        state_dist = self.state_dist(embedding)
        rstate = state_dist.rsample()
        return LatentTuple(embedding, state_dist, rstate)

    def forward(self, observation):
        latent_tuple = self.get_latent(observation)

        values = self.critic(latent_tuple)
        actor_dist = self.actor(latent_tuple, onehot=False)

        action = actor_dist.sample()
        log_prob = actor_dist.log_prob(action)
        return action, values, log_prob

    def evaluate_actions(self, observation, action):
        latent_tuple = self.get_latent(observation)

        values = self.critic(latent_tuple)
        actor_dist = self.actor(latent_tuple, onehot=False)
        log_prob = actor_dist.log_prob(action)
        entropy = actor_dist.entropy()
        return values, log_prob, entropy

    def _preprocess(self, tensor):
        return tensor

    @staticmethod
    def batch_chunk_distribution(distribution):
        prev_loc, next_loc = torch.chunk(distribution.loc, 2, dim=0)
        prev_scale, next_scale = torch.chunk(distribution.scale, 2, dim=0)
        return (torch.distributions.Normal(prev_loc, prev_scale),
                torch.distributions.Normal(next_loc, next_scale))

    @staticmethod
    def make_normal_prior(distribution):
        return torch.distributions.Normal(
            torch.zeros_like(distribution.loc),
            torch.ones_like(distribution.scale))
