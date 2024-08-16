import sys

import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, GatedGraphConv

from models.encoders import EncoderVAE, EncoderCNNVAE, EncoderMLP
from models.gnns import GNNBasicBlock
from models.decoders import DecoderVAE, DecoderCNNVAE, DecoderMLP

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

class CNNVAEModel(nn.Module):
    """
    CNN Variational Autoencoder (VAE).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):

        if "blocks" not in config.keys():
            config["blocks"] = [
                config["block_1"], 
                config["block_2"], 
                config["block_3"]
            ]

        return config

    def __init__(
        self,
        in_channels=3,
        latent_size=256,
        blocks=[3, 3, 1],
        **kwargs
    ):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = EncoderCNNVAE(in_channels, latent_size, blocks)
        self.decoder = DecoderCNNVAE(in_channels, latent_size, list(reversed(blocks)))

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
                    gnn_conv_args,
                    res_connect=False
                ),
                *[
                    GNNBasicBlock(
                        hidden_size,
                        hidden_size, 
                        gnn_conv, 
                        gnn_conv_args,
                        res_connect=True
                    ) for l in range(layers-1)
                ],
                # nn.Dropout(dropout),
                gnn_conv(
                    in_channels=hidden_size,
                    out_channels=out_size, 
                    **gnn_conv_args
                ),
            ]
        )

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch):
        out = batch.data

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                "edge_attr": batch.edge_weights 
            }
        elif self.gnn_conv in (GCNConv, GatedGraphConv):
            conv_fwd_args = {
                "edge_weight": batch.edge_weights
            }
        else:
            conv_fwd_args = {}

        conv_fwd_args = {}

        for layer in self.layers:
            if isinstance(layer, GNNBasicBlock):
                out, _ = layer(out, batch.edge_index, **conv_fwd_args)
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index, **conv_fwd_args)
            else:
                out = layer(out)

        return out

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron (MLP).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        return config

    def __init__(
        self,
        in_features=128,
        hidden_size=64,
        out_size=10,
        layers=2,
        **kwargs
    ):
        super().__init__()

        self.encoder = EncoderMLP(in_features, hidden_size, layers)
        self.decoder = DecoderMLP(hidden_size, out_size, layers)

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch):
        out = self.encoder(batch.data)

        y_hat = self.decoder(out)

        return y_hat