import sys

import torch
import torch.nn as nn

from models.encoders import TemporalEncoder, EncoderVAE
from models.gnns import NeroStemGNN
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
        pass

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

    def forward(self, batch):
        pass