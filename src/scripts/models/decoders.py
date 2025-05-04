import torch
import torch.nn as nn

from torch.nn import functional as F

from .encoders import ResidualStack

from typing import List, Dict


class MLPBasicBlock(nn.Module):
    """
    Basic block for a Multi-Layer Perceptron (MLP) module.
    """

    def __init__(
        self, 
        in_channels, 
        n_hidden, 
        out_channels, 
    ):
        """
        Initialize the MLPBasicBlock module.

        Args:
            in_channels (int): Input size.
            n_hidden (int): Number of hidden units.
            out_channels (int): Output size.
        """

        super().__init__()

        layer1 = nn.Linear(in_channels, n_hidden)
        layer2 = nn.Linear(n_hidden, n_hidden)
        layer3 = nn.Linear(n_hidden, out_channels)
        self.layers = nn.ModuleList([
            layer1,
            layer2,
            layer3
        ])

    def forward(self, x):
        """
        Forward pass of the MLPBasicBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        out = x
        for layer in self.layers:
            out = F.leaky_relu(layer(out))
        return out

class DecoderVAE(nn.Module):

    def __init__(
        self,
        latent_size,
        hidden_size,
        out_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(latent_size, hidden_size),
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)],
                nn.Linear(hidden_size, out_size),
            ]
        )
        self.activation_fn = activation_fn

    def forward(self, lat):
        out = lat
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        return out

class DecoderCNNVAE(nn.Module):

    def __init__(
        self,
        out_channels,
        latent_size,
        blocks=[1, 1, 1], 
        activation_fn=nn.LeakyReLU(),
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(latent_size, 256 * 36),
                # nn.Linear(latent_size, 512 * 1),
                activation_fn,
        )
        self.layers = nn.ModuleList(
            [
                # nn.ConvTranspose2d(latent_size, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                # nn.ConvTranspose2d(latent_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                *[DeConvBasicBlock(256, 256) for _ in range(blocks[0])],
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                *[DeConvBasicBlock(128, 128) for _ in range(blocks[1])],
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                *[DeConvBasicBlock(64, 64) for _ in range(blocks[2])],
                # nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
                nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1),

                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.GELU(),
                # nn.Conv2d(64, 64, kernel_size=3, padding=1),
                # nn.GELU(),
                # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.GELU(),
                # nn.Conv2d(32, 32, kernel_size=3, padding=1),
                # nn.GELU(),
                # nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

                # nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                # *[DeConvBasicBlock(128, 128) for _ in range(blocks[0])],
                # nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                # *[DeConvBasicBlock(64, 64) for _ in range(blocks[1])],
                # nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                # *[DeConvBasicBlock(32, 32) for _ in range(blocks[2])],
                # nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                # *[DeConvBasicBlock(16, 16) for _ in range(blocks[2])],
                # nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=2),
                # *[DeConvBasicBlock(16, 16) for _ in range(blocks[2])],
                # nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1),

            ]
        )
        self.activation_fn = activation_fn
        self.sigmoid = nn.Tanh()

    def forward(self, lat):
        out = lat
        out = self.linear(out)
        # out = out.reshape(out.shape[0], -1, 4, 4)
        out = out.reshape(out.shape[0], -1, 6, 6)
        for layer in self.layers[:-1]:
            out = self.activation_fn(layer(out))
            # out = layer(out)

        out = self.sigmoid(self.layers[-1](out))

        return out

class DecoderVQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_hiddens,
        num_upsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon

class DeConvBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()                
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = F.leaky_relu(out + x)

        return out


class DecoderMLP(nn.Module):

    def __init__(
        self,
        hidden_size,
        out_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)],
                nn.Linear(hidden_size, out_size),
            ]
        )
        self.activation_fn = activation_fn

    def forward(self, lat):
        out = lat
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        return out