import sys

import jax

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, GatedGraphConv, ARMAConv, SAGEConv, GraphConv

from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Batch, Data

from annoy import AnnoyIndex
from sklearn.decomposition import PCA

from models.encoders import EncoderVAE, EncoderCNNVAE, EncoderVQVAE, EncoderMLP, WeightedSAGEConv
from models.gnns import GNNBasicBlock, DGMLayer, DGMGNNBasicBlock, GNN1, GNN2
from models.decoders import DecoderVAE, DecoderCNNVAE, DecoderVQVAE, DecoderMLP

from typing import List, Dict, Literal, Optional


_CONV_MAP = {
    "gcn": GCNConv,
    "sage": SAGEConv,
    "gatv2": GATv2Conv,
    "graph": GraphConv,
}

_POOL_MAP = {
    "sum": global_add_pool,
    "max": global_max_pool,
    "mean": global_mean_pool,
}


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

class GNN(torch.nn.Module):
    @staticmethod
    def pre_init(config, **kwargs):
        config["in_channels"] = config["in_features"]
        config["hidden_channels"] = config["hidden_size"]
        config["out_channels"] = config["out_size"]

        return config

    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden_channels)
        self.conv2 = WeightedSAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        out = self.conv1(batch.x, batch.edge_index, batch.edge_weight)
        out = F.relu(out)
        #x = self.conv2(x, edge_index, edge_weight)
        #x = F.relu(x)
        out = self.lin(out)

        return out[batch.mask.bool()]


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
                nn.Dropout(dropout),
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

    def forward(self, batch):
        out = batch.x

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                "edge_attr": batch.edge_weight
            }
        elif self.gnn_conv in (GCNConv, GatedGraphConv, ARMAConv):
            conv_fwd_args = {
                "edge_weight": batch.edge_weight
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

        # return out[:batch.batch_size]
        return out[batch.mask.bool()]

class DGMGNNModel(nn.Module):
    """
    Differentiable Graph Module - Graph Neural Network (GNN).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        config["lgi_op"] = getattr(sys.modules[__name__], "DGMLayer")
        config["lgi_op_args"] = {
            "encoder": nn.Linear,
            "k": 5
        }
        return config

    def __init__(
        self,
        in_features,
        hidden_size,
        out_size,
        gnn_conv,
        gnn_conv_args,
        lgi_op,
        lgi_op_args,
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
                DGMGNNBasicBlock(
                    in_features, 
                    hidden_size, 
                    gnn_conv, 
                    gnn_conv_args,
                    lgi_op=lgi_op,
                    lgi_op_args={
                        "encoder": lgi_op_args["encoder"],
                        "in_channels": in_features,
                        "out_channels": hidden_size,
                        "k": lgi_op_args["k"]
                    },
                    res_connect=False
                ),
                *[
                    DGMGNNBasicBlock(
                        hidden_size,
                        hidden_size, 
                        gnn_conv, 
                        gnn_conv_args,
                        lgi_op=lgi_op,
                        lgi_op_args={
                            "encoder": lgi_op_args["encoder"],
                            "in_channels": (hidden_size + hidden_size),
                            "out_channels": hidden_size,
                            "k": lgi_op_args["k"]
                        },
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

    def forward(self, batch):
        out = batch.x
        out_lgi = None

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                # "edge_attr": batch.edge_weights 
            }
        # elif self.gnn_conv in (GCNConv, GatedGraphConv, ARMAConv):
        #     conv_fwd_args = {
        #         "edge_weight": batch.edge_weights
        #     }
        else:
            conv_fwd_args = {}

        for layer in self.layers:
            if isinstance(layer, DGMGNNBasicBlock):
                out, _, out_lgi = layer(
                    out, 
                    out_lgi, 
                    batch.edge_index, 
                    graph_batch=batch, 
                    **conv_fwd_args
                )
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index, **conv_fwd_args)
            else:
                out = layer(out)

        # return out
        return out[torch.argwhere(batch.mask.reshape(-1)).reshape(-1)]

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
        out = batch.x

        for layer in self.layers:
            out = self.activation_fn(layer(out))

        # y_hat = out

        # return y_hat
        return out[torch.argwhere(batch.mask.reshape(-1)).reshape(-1)]

class AEModel(nn.Module):
    """
    Deterministic Autoencoder
    """

    @staticmethod
    def pre_init(config, **kwargs):
        config["out_size"] = config["in_features"]

        return config

    def __init__(
            self, 
            in_features=784, 
            hidden_size=256,
            out_size=784, 
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
        dims = batch.x.shape

        enc = self.encoder(batch.x.flatten(1, -1))
        dec = self.decoder(enc)

        out = torch.reshape(dec, dims)

        # return out, enc, dec, dims
        return out

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

class PCAModel(PCA):
    def __init__(self, n_components):
        super().__init__(n_components)

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

#######################################################

class E2EWanGNN(nn.Module):
    """End‑to‑end model: GNN_1 on WANs → Y, then GNN_2 on meta‑graph → logits.

    Forward:
        logits_masked = model(wan_batch, meta_batch)
    """
    def __init__(
        self,
        # GNN_1
        wan_in_channels: int,
        g1_hidden: int = 64,
        g1_out: int = 128,
        g1_conv: str = "gcn",
        g1_aggregator: Literal["sum","max","mean"] = "sum",
        g1_dropout: float = 0.0,
        # GNN_2
        g2_hidden: int = 128,
        g2_layers: int = 2,
        g2_conv: str = "gcn",
        num_classes: int = 10,
        g2_dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn1 = GNN1(
            in_channels=wan_in_channels,
            hidden=g1_hidden,
            out_channels=g1_out,
            conv=g1_conv,
            aggregator=g1_aggregator,
            dropout=g1_dropout,
        )
        self.gnn2 = GNN2(
            in_channels=g1_out,
            hidden=g2_hidden,
            num_layers=g2_layers,
            num_classes=num_classes,
            conv=g2_conv,
            dropout=g2_dropout,
        )

    def forward(self, batch: Batch, **kwargs) -> torch.Tensor:
        # Stage 1: per‑WAN embeddings
        Y = self.gnn1(batch, device=kwargs['device'])  # [N_sub, g1_out]
        ei, ew = batch.edge_index, batch.edge_weight
        logits = self.gnn2(Y, ei, ew)  # [N_sub, num_classes]
        # return logits[batch.mask]
        return logits

