"""Taken from "VQ-VAE by Aäron van den Oord" notebook with a slight modification
https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=LXmpdTRf5ffa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from modular_baselines.contrib.jacobian_trace.type_aliases import DiscreteLatentTuple



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # input:  BHWD
        input_shape = inputs.shape
        encoding = self.discrete_encoding(inputs)

        # Quantize and unflatten
        quantized = self.quantize(encoding)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encoding.reshape(-1, encoding.shape[-1]), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return DiscreteLatentTuple(inputs, quantized, encoding), q_latent_loss, e_latent_loss, perplexity

    def discrete_encoding(self, inputs):
        # BxHxWxD -> BxHxWxE
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        return encodings.reshape(*inputs.shape[:3], encodings.shape[-1])

    def quantize(self, encoding):
        # BxHxWxE ExD -> BxHxWxD
        return torch.matmul(
            encoding.reshape(-1, encoding.shape[-1]),
            self._embedding.weight).reshape(*encoding.shape[:3], -1)
