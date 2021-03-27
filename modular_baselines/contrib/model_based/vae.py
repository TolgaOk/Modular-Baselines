import torch
import numpy as np

class Vae(torch.nn.Module):
    def __init__(self, image_channels=4, z_dim=32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            # Flatten()
        )
        self.h_dim = (256, 3, 3)
        self.fc_mean = torch.nn.Linear(np.product(self.h_dim), z_dim)
        self.fc_std = torch.nn.Linear(np.product(self.h_dim), z_dim)

        self.fc3 = torch.nn.Linear(z_dim, np.product(self.h_dim))
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.h_dim[0], 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, image_channels, kernel_size=1, stride=1),
            torch.nn.Sigmoid(),
        )

    def distribution(self, encoder_logit):
        encoder_logit = encoder_logit.reshape(encoder_logit.shape[0], -1)
        mean, std = self.fc_mean(encoder_logit), self.fc_std(encoder_logit)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

    def encode(self, tensor):
        encoder_logit = self.encoder(tensor)
        dist = self.distribution(encoder_logit)
        return dist

    def decode(self, latent):
        logit = self.fc3(latent)
        logit = logit.reshape(logit.shape[0], *self.h_dim)
        return self.decoder(logit)

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
