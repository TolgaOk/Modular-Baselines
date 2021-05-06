from itertools import chain
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

from modular_baselines.contrib.jacobian_trace.type_aliases import DiscreteLatentTuple
from modular_baselines.contrib.jacobian_trace.models.vector_representation import (DenseModel,
                                                                                   NormalDistributionModel)
from modular_baselines.contrib.jacobian_trace.models.vqvae import VectorQuantizer


class ObservationDecoder(NormalDistributionModel):
    def forward(self, latent_tuple: DiscreteLatentTuple):
        features = self.make_feature(latent_tuple)
        return super().forward(features)

    @staticmethod
    def make_feature(latent_tuple: DiscreteLatentTuple):
        return latent_tuple.quantized.reshape(latent_tuple.quantized.shape[0], -1)


class ObservationEncoder(DenseModel):
    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 n_embeddings: int,
                 n_latent: int,
                 activation=nn.ELU):
        super().__init__(feature_size, output_size * n_latent, layers, hidden_size, activation)
        self.n_embeddings = n_embeddings
        self.n_latent = n_latent
        self.embedding = VectorQuantizer(n_embeddings, output_size)

    def forward(self, observation: torch.Tensor):
        features = super().forward(observation)
        features = features.reshape(features.shape[0], self.n_latent, 1, -1)
        latent_tuple, q_latent_loss, e_latent_loss, perplexity = self.embedding(features)
        return latent_tuple, q_latent_loss, e_latent_loss, perplexity


class DiscreteTransitionDistribution(DenseModel):
    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 action_size: int,
                 layers: int,
                 hidden_size: int,
                 n_embeddings: int,
                 n_latent: int,
                 activation=nn.ReLU):
        self.action_size = action_size
        self.n_embeddings = n_embeddings
        self.n_latent = n_latent
        super().__init__(feature_size * n_latent,
                         output_size * action_size * n_latent,
                         layers,
                         hidden_size,
                         activation)
        self.embedding = VectorQuantizer(n_embeddings, feature_size)

    def forward(self, latent_tuple: DiscreteLatentTuple, actions: torch.Tensor):
        actions = DiscreteActor.maybe_onehot(actions, self.action_size)

        features = self.make_feature(latent_tuple)
        logits = super().forward(features).reshape(
            features.shape[0], self.action_size, -1)
        # BxHF
        selected_logits = torch.einsum("baf,ba->bf", logits, actions)

        selected_logits = selected_logits.reshape(selected_logits.shape[0] * self.n_latent, -1)
        dist = torch.distributions.OneHotCategorical(logits=selected_logits)
        return dist

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        # return shape: BxHWD
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)

    @staticmethod
    def reparam(dist: torch.distributions.Distribution, embedding: torch.Tensor):
        # dist: BHWxF
        # embeding: BxHxWxF
        probs = dist.probs.reshape(*embedding.shape)
        return embedding + probs - probs.detach()


class DiscreteActor(DenseModel):

    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 n_embeddings: int,
                 n_latent: int,
                 activation=nn.ReLU):
        self.n_embeddings = n_embeddings
        self.n_latent = n_latent
        super().__init__(feature_size * n_latent,
                         output_size,
                         layers,
                         hidden_size,
                         activation)
        self.embedding = VectorQuantizer(n_embeddings, feature_size)

    def forward(self, latent_tuple: DiscreteLatentTuple, onehot: bool = False):
        features = self.make_feature(latent_tuple)
        act_logits = super().forward(features)
        if onehot:
            return torch.distributions.one_hot_categorical.OneHotCategorical(logits=act_logits)
        return torch.distributions.categorical.Categorical(logits=act_logits)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        # return shape: BxHWD
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)

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

    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 n_embeddings: int,
                 n_latent: int,
                 activation=nn.ReLU):
        self.n_embeddings = n_embeddings
        self.n_latent = n_latent
        super().__init__(feature_size * n_latent,
                         output_size,
                         layers,
                         hidden_size,
                         activation)
        self.embedding = VectorQuantizer(n_embeddings, feature_size)

    def forward(self, latent_tuple: DiscreteLatentTuple):
        features = self.make_feature(latent_tuple)
        return super().forward(features)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        # return shape: BxHWD
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)


class DiscreteJTACModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 action_size,
                 model_lr=4e-4,
                 ac_lr: float = 7e-4,
                 n_embeddings=128,
                 n_latent=9,
                 quantize_size=32,
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
            quantize_size,
            encoder_layers,
            encoder_hidden_size,
            n_embeddings,
            n_latent)
        self.decoder = ObservationDecoder(
            quantize_size * n_latent,
            input_size,
            decoder_layers,
            decoder_hidden_size)
        self.transition_dist = DiscreteTransitionDistribution(
            quantize_size,
            n_embeddings,
            action_size,
            trans_layers,
            trans_hidden_size,
            n_embeddings,
            n_latent)

        self.actor = DiscreteActor(
            quantize_size,
            action_size,
            actor_layers,
            actor_hidden_size,
            n_embeddings,
            n_latent)
        self.critic = Critic(
            quantize_size,
            1,
            critic_layers,
            critic_hidden_size,
            n_embeddings,
            n_latent)

        self.model_optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters(),
                  self.transition_dist.parameters()),
            lr=model_lr)
        self.ac_optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=ac_lr)

    def get_latent(self, observation):
        observation = self._preprocess(observation)
        return self.encoder(observation)

    def _forward(self, observation):
        latent_tuple, q_latent_loss, e_latent_loss, perplexity = self.get_latent(observation)

        values = self.critic(latent_tuple)
        actor_dist = self.actor(latent_tuple, onehot=False)
        return actor_dist, values, latent_tuple

    def forward(self, observation):
        actor_dist, values, latent_tuple = self._forward(observation)

        action = actor_dist.sample()
        log_prob = actor_dist.log_prob(action)
        return action, values, log_prob

    def evaluate_actions(self, observation, action):
        actor_dist, values, latent_tuple = self._forward(observation)
        log_prob = actor_dist.log_prob(action)
        entropy = actor_dist.entropy()
        return values, log_prob, entropy

    def loss(self, obs, next_obs, actions):
        combined_obs = torch.cat([obs, next_obs], dim=0)
        combined_latent, q_latent_loss, e_latent_loss, perplexity = self.get_latent(combined_obs)
        pred_obs_dist = self.decoder(combined_latent)

        latent_tuple, next_latent_tuple = [DiscreteLatentTuple(*half_tensor) for half_tensor in zip(
            *[torch.chunk(tensor, 2, dim=0) for tensor in combined_latent])]
        trans_dist = self.transition_dist(latent_tuple, actions)
        transition_loss = -trans_dist.log_prob(next_latent_tuple.encoding.reshape(-1, next_latent_tuple.encoding.shape[-1]))
        transition_loss = transition_loss.reshape(obs.shape[0], -1).sum(1).mean(0)

        recon_loss = -pred_obs_dist.log_prob(combined_obs).sum(1).mean(0)

        return recon_loss, transition_loss, e_latent_loss, q_latent_loss, perplexity

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
