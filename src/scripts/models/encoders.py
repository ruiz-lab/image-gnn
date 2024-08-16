import torch
import torch.nn as nn

from torch.nn import functional as F

from typing import List, Dict, Any


class EncoderVAE(nn.Module):

    def __init__(
        self, 
        in_features, 
        hidden_size,
        latent_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, hidden_size),
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)]
            ]
        )
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.var_layer = nn.Linear(hidden_size, latent_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        mean = self.activation_fn(self.mean_layer(out))
        logvar = self.activation_fn(self.var_layer(out))

        return mean, logvar

class EncoderCNNVAE(nn.Module):

    def __init__(
        self,
        in_channels,
        latent_size,
        blocks=[1, 1, 1], 
        activation_fn=nn.LeakyReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                *[ConvBasicBlock(64, 64) for _ in range(blocks[0])],
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=1),
                *[ConvBasicBlock(128, 128) for _ in range(blocks[1])],
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(128, 256, kernel_size=1),
                *[ConvBasicBlock(256, 256) for _ in range(blocks[2])],
            ]
        )
        self.mean_layer = nn.Linear(256, latent_size)
        self.var_layer = nn.Linear(256, latent_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        out  = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        mean = self.activation_fn(
            self.mean_layer(
                torch.permute(out, (0, 2, 3, 1))
            )
        )
        logvar = self.activation_fn(
            self.var_layer(
                torch.permute(out, (0, 2, 3, 1))
            )
        )

        return (
            torch.permute(mean, (0, 3, 1, 2)), 
            torch.permute(logvar, (0, 3, 1, 2))
        )

class ConvBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(
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

class EncoderMLP(nn.Module):

    def __init__(
        self, 
        in_features, 
        hidden_size,
        latent_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, hidden_size),
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)]
            ]
        )
        self.activation_fn = activation_fn

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        return out