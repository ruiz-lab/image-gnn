import sys

import jax

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, GatedGraphConv, ARMAConv

from annoy import AnnoyIndex

from models.encoders import EncoderVAE, EncoderCNNVAE, EncoderVQVAE, EncoderMLP
from models.gnns import GNNBasicBlock
from models.decoders import DecoderVAE, DecoderCNNVAE, DecoderVQVAE, DecoderMLP

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
        latent_size=384,
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
                # nn.Linear(in_features, hidden_size),
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
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_size)
            ]
        )

    def _step(self):
        pass

    def _eval(self):
        pass

    # def call(self, x, edge_index, conv_fwd_args):
    #     out = x

    #     for layer in self.layers:
    #         if isinstance(layer, GNNBasicBlock):
    #             out, _ = layer(out, edge_index, **conv_fwd_args)
    #         elif isinstance(layer, self.gnn_conv):
    #             out = layer(out, edge_index, **conv_fwd_args)
    #         else:
    #             out = layer(out)

    #     return out

    def forward(self, batch):
        out = batch.data

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                "edge_attr": batch.edge_weights 
            }
        elif self.gnn_conv in (GCNConv, GatedGraphConv, ARMAConv):
            conv_fwd_args = {
                "edge_weight": batch.edge_weights
            }
        else:
            conv_fwd_args = {}

        for layer in self.layers:
            if isinstance(layer, GNNBasicBlock):
                out, _ = layer(out, batch.edge_index, **conv_fwd_args)
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index, **conv_fwd_args)
            else:
                out = layer(out)

        # return out
        return out[torch.argwhere(batch.mask.reshape(-1)).reshape(-1)]
        # return jax.vmap(self.call, in_axes=(0, 0, 0))(out, batch.edge_index, conv_fwd_args)

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

        self.layers = nn.ModuleList([
            nn.Linear(in_features, hidden_size),
            *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-1)],
            nn.Linear(hidden_size, out_size)
        ])
        self.activation_fn = nn.ReLU()

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch):
        out = batch.data

        for layer in self.layers:
            out = self.activation_fn(layer(out))

        # y_hat = out

        # return y_hat
        return out[torch.argwhere(batch.mask.reshape(-1)).reshape(-1)]

class kNNModel():
    def __init__(self, ds, n_trees=500, k=-1, metric="euclidean"):
        self.indexes = list(range(ds.shape[0]))
        self.targets = ds[:, 1]
        self.ann : AnnoyIndex = AnnoyIndex(ds.shape[1], metric)
        _ = self.ann.build(n_trees)
        for i in range(ds.shape[0]):
            self.ann.add_item(i, ds[i, :])
 
        self.k = k

    def __call__(self, index, n):
        knn_idx, knn_dist = self.ann.get_nns_by_item(
            index, 
            n, 
            self.k, 
            include_distances=True
        )

        return (knn_idx, knn_dist)

    def predict(self, k):
        k_nn = [self.__call__(index, k)[0] for index in self.indexes]


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )

class VQVAEModel(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQVAE).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        return config

    def __init__(
        self,
        in_channels=3,
        num_hiddens=128,
        num_downsampling_layers=2,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embedding_dim=128,
        num_embeddings=512,
        use_ema=True,
        decay=0.99,
        epsilon=1e-5,
        **kwargs
    ):
        super().__init__()
        self.encoder = EncoderVQVAE(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = DecoderVQVAE(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, batch):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(batch.x)
        x_recon = self.decoder(z_quantized)
        return (
            x_recon,
            nn.Flatten()(z_quantized),
            dictionary_loss,
            commitment_loss,
            self.vq.use_ema,
            batch.data_var,
        )