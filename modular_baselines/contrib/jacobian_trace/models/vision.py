import numpy as np
import torch
from typing import Tuple


class ObservationEncoder(torch.nn.Module):
    def __init__(self,
                 depth: int = 32,
                 stride: int = 2,
                 shape: Tuple[int] = (3, 64, 64),
                 activation: torch.nn.Module = torch.nn.ReLU):
        super().__init__()
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            torch.nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            torch.nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            torch.nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            # activation(),
        )
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch_shape = observation.shape[:-3]
        img_shape = observation.shape[-3:]
        embed = self.convolutions(observation.reshape(-1, *img_shape))
        # embed = torch.reshape(embed, (*batch_shape, -1))
        return embed.permute(0, 2, 3, 1)

    @property
    def embed_size(self) -> int:
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size, conv4_shape


class ObservationDecoder(torch.nn.Module):
    def __init__(self,
                 depth: int = 32,
                 stride: int = 2,
                 activation: torch.nn.Module = torch.nn.ReLU,
                 embed_size: int = 1024,
                 shape: Tuple[int] = (3, 64, 64)):
        super().__init__()
        self.depth = depth
        self.shape = shape

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape,
                                         padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape,
                                         padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(conv3_shape, conv4_shape,
                                         padding, conv4_kernel_size, stride)
        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = torch.nn.Linear(embed_size, 32 * depth * np.prod(conv4_shape).item())
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size,
                                     stride, output_padding=conv4_pad),
            activation(),
            torch.nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size,
                                     stride, output_padding=conv3_pad),
            activation(),
            torch.nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size,
                                     stride, output_padding=conv2_pad),
            activation(),
            torch.nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size,
                                     stride, output_padding=conv1_pad),
        )

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        """
        :param features: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        features = features.permute(0, 3, 1, 2)
        features = features.reshape(features.shape[0], -1)
        batch_shape = features.shape[:-1]
        embed_size = features.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        features = features.reshape(squeezed_size, embed_size)
        features = self.linear(features)
        features = torch.reshape(features, (squeezed_size, *self.conv_shape))
        features = self.decoder(features)
        mean = torch.reshape(features, (*batch_shape, *self.shape))
        obs_dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, 1), len(self.shape))
        return obs_dist


def conv_out(h_in: int, padding: int, kernel_size: int, stride: int) -> int:
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in: int, conv_out: int, padding: int, kernel_size: int, stride: int) -> int:
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in: int, padding: int, kernel_size: int, stride: int) -> Tuple[int]:
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in: int,
                         conv_out: int,
                         padding: int,
                         kernel_size: int,
                         stride: int) -> Tuple[int]:
    return tuple(output_padding(h_in[i],
                                conv_out[i],
                                padding,
                                kernel_size,
                                stride) for i in range(len(h_in)))
