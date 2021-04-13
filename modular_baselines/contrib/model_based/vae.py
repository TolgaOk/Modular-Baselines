import torch
import numpy as np


class Vae(torch.nn.Module):
    def __init__(self, image_channels, z_dim=32):
        super().__init__()
        self.hidden_dims = (256, 3, 3)
        self.in_channel = image_channels
        self.hidden_channel = 128
        self.z_dim = z_dim

        self.build()

    def build(self):
        self.encoder = self.encoder_net(self.in_channel, 256, self.hidden_channel)
        self.fc_mean = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.fc_std = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.f_latent = torch.nn.Linear(self.z_dim, np.product(self.hidden_dims))
        self.decoder = self.decoder_net(256, self.in_channel, self.hidden_channel)

    @staticmethod
    def encoder_net(in_channel, out_channel=256, hidden_channel=128):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channel, out_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
        )

    @staticmethod
    def decoder_net(in_channel, out_channel, hidden_channel=128):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1),
            torch.nn.Sigmoid(),
        )

    def distribution(self, encoder_logit):
        encoder_logit = encoder_logit.reshape(encoder_logit.shape[0], -1)
        mean = self.fc_mean(encoder_logit)
        std = torch.nn.functional.softplus(self.fc_std(encoder_logit))
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

    def encode(self, tensor):
        encoder_logit = self.encoder(tensor)
        dist = self.distribution(encoder_logit)
        return dist

    def decode(self, latent):
        hidden = torch.relu(self.f_latent(latent))
        hidden = hidden.reshape(hidden.shape[0], *self.hidden_dims)
        return self.decoder(hidden)

    def forward(self, tensor):
        dist = self.encode(tensor)
        latent = dist.rsample()
        prediction = self.decode(latent)
        return latent, prediction, dist

    @staticmethod
    def loss(input_tensor, prediction, dist):
        prior_mean = torch.zeros_like(dist.loc)
        prior_std = torch.ones_like(dist.scale)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)

        kl_divergence = torch.distributions.kl_divergence(dist, prior_dist).mean()
        recon_loss = torch.nn.functional.mse_loss(prediction, input_tensor)
        return recon_loss, kl_divergence


class TransitionVae(Vae):

    def __init__(self, action_size, image_channels, z_dim=32):
        self.action_size = action_size
        super().__init__(image_channels, z_dim)

    def build(self):
        self.encoder = self.encoder_net(self.in_channel, 256, self.hidden_channel)
        self.decoder = self.decoder_net(256, self.in_channel, self.hidden_channel)

        self.fc_mean = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.fc_std = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.f_latent = torch.nn.Linear(self.z_dim, np.product(self.hidden_dims) * self.action_size)

    def decode(self, latent, actions):
        batch_size = latent.shape[0]
        action_logits = self.f_latent(latent).reshape(batch_size, self.action_size, -1)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        action_indexes = actions.unsqueeze(-1).repeat_interleave(action_logits.shape[-1], dim=-1)
        logit = action_logits.gather(dim=1, index=action_indexes)

        hidden = torch.relu(logit)
        hidden = hidden.reshape(hidden.shape[0], *self.hidden_dims)
        return self.decoder(hidden)

    def forward(self, tensor, actions):
        dist = self.encode(tensor)
        latent = dist.rsample()
        prediction = self.decode(latent, actions)
        return latent, prediction, dist


class LatentActionPredictionVae(Vae):

    def __init__(self, action_size, image_channels, z_dim=32):
        self.action_size = action_size
        super().__init__(image_channels, z_dim)

    def build(self):
        self.encoder = self.encoder_net(self.in_channel, 256, self.hidden_channel)
        self.fc_mean = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.fc_std = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)

        self.action_pred = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.action_size))

    def decode(self):
        raise NotImplementedError

    def forward(self, obs_tensor, next_obs_tensor):
        obs = torch.cat((next_obs_tensor, obs_tensor), dim=0)
        dists = self.encode(obs)
        next_obs_latent, obs_latent = dists.rsample().split(obs.shape[0] // 2, dim=0)
        prediction = self.action_pred(torch.cat((next_obs_latent, obs_latent), dim=-1))
        return prediction, dists

    @staticmethod
    def loss(actions, prediction, dist):
        if len(actions.shape) != 1:
            actions = actions.squeeze(1)
        prior_mean = torch.zeros_like(dist.loc)
        prior_std = torch.ones_like(dist.scale)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)

        kl_divergence = torch.distributions.kl_divergence(dist, prior_dist).mean()
        recon_loss = torch.nn.functional.cross_entropy(prediction, actions)
        return recon_loss, kl_divergence


class LatentTransitionPredictionVae(Vae):

    def __init__(self, action_size, image_channels, z_dim=32):
        self.action_size = action_size
        super().__init__(image_channels, z_dim)

    def build(self):
        self.encoder = self.encoder_net(self.in_channel, 256, self.hidden_channel)
        self.fc_mean = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)
        self.fc_std = torch.nn.Linear(np.product(self.hidden_dims), self.z_dim)

        self.latent_transition_pred = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.z_dim * self.action_size * 2))

    def decode(self):
        raise NotImplementedError

    def forward(self, obs_tensor, next_obs_tensor, actions):
        combined_obs = torch.cat((next_obs_tensor, obs_tensor), dim=0)
        dists = self.encode(combined_obs)
        next_obs_dist, obs_dist = [torch.distributions.normal.Normal(loc, std)
                                   for loc, std in zip(
            dists.loc.split(combined_obs.shape[0] // 2, dim=0),
            dists.scale.split(combined_obs.shape[0] // 2, dim=0))
        ]
        obs_latent = obs_dist.rsample()

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        action_indexes = actions.unsqueeze(-1).repeat_interleave(obs_latent.shape[-1] * 2, dim=-1)

        action_next_obs_latent = self.latent_transition_pred(
            obs_latent).reshape(obs_latent.shape[0], self.action_size, -1)
        pred_next_latent_obs_params = action_next_obs_latent.gather(
            dim=1, index=action_indexes).squeeze(1)

        pred_mean, pred_std_logit = pred_next_latent_obs_params.split(
            obs_latent.shape[-1], dim=-1)
        pred_std = torch.nn.functional.softplus(pred_std_logit)
        pred_next_obs_dist = torch.distributions.Normal(pred_mean, pred_std)

        return next_obs_dist, pred_next_obs_dist, dists

    @staticmethod
    def loss(next_obs_dist, pred_next_obs_dist, dist):
        prior_mean = torch.zeros_like(dist.loc)
        prior_std = torch.ones_like(dist.scale)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)

        kl_divergence = torch.distributions.kl_divergence(dist, prior_dist).mean()
        # We detach next observation latent vector so that the trivial gradients can bee avoided

        recon_loss = torch.distributions.kl_divergence(pred_next_obs_dist, next_obs_dist).mean()
        return recon_loss, kl_divergence
