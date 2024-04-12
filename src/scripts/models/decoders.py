import torch
import torch.nn as nn

from torch.nn import functional as F

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

        # out = self.activation_fn(self.layers[-1](out))

        return out