import sys

import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv

from models.encoders import TemporalEncoder, EncoderVAE
from models.gnns import NeroStemGNN, GNNBasicBlock
from models.decoders import MLPBasicBlock, DecoderVAE

from typing import List, Dict


class GenericModel(nn.Module):
    """
    Generic model.

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        pass

    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass

    def step(self):
        pass

    def eval(self):
        pass

    def forward(self, batch):
        pass

class VAEModel(nn.Module):
    """
    Variational Autoencoder (VAE).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        if "latent_size" not in config.keys():
            config["latent_size"] = config["hidden_size"] // 2

        config["out_size"] = config["in_features"]

        return config

    def __init__(
        self,
        in_features=784,
        hidden_size=256,
        latent_size=128,
        out_size=784,
        layers=2,
        **kwargs
    ):
        super().__init__()

        self.encoder = EncoderVAE(in_features, hidden_size, latent_size, layers)
        self.decoder = DecoderVAE(latent_size, hidden_size, out_size, layers)

    def _step(self):
        pass

    def _eval(self):
        pass

    def _reparametrization(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)

        return mean + (epsilon * std)

    def forward(self, batch):
        mean, logvar = self.encoder(batch.x)

        z = self._reparametrization(mean, logvar)
        y_hat = self.decoder(z)

        return y_hat, mean, logvar, z

class GNNModel(nn.Module):
    """
    Graph Neural Network (GNN).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        return config

    def __init__(
        self,
        in_features,
        hidden_size,
        out_size,
        gnn_conv,
        gnn_conv_args,
        layers=1,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        gnn_conv = getattr(sys.modules[__name__], gnn_conv)
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                GNNBasicBlock(
                    in_features, 
                    hidden_size, 
                    gnn_conv, 
                    gnn_conv_args
                ),
                *[
                    GNNBasicBlock(
                        hidden_size, 
                        hidden_size, 
                        gnn_conv, 
                        gnn_conv_args
                    ) for l in range(layers-1)
                ],
                nn.Dropout(dropout),
                gnn_conv(
                    hidden_size, 
                    out_size, 
                    **gnn_conv_args
                )
            ]
        )

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch):
        out = batch.data
        for layer in self.layers:
            if isinstance(layer, GNNBasicBlock):
                out, _ = layer(out, batch.edge_index)
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index)
            else:
                out = layer(out)

        return out