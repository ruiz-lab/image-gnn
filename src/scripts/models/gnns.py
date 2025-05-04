import sys

import torch
import torch.nn as nn

from torch.nn import functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GATConv, GatedGraphConv

from torch_geometric.nn.pool import global_mean_pool

from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

from typing import List, Dict


class GNNBasicBlock(nn.Module):
    """
    Basic block for graph neural network (GNN) with skip connection.
    """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            gnn_conv, 
            gnn_conv_args,
            res_connect,
            **kwargs
        ):
        """
        Initialize the GNNBasicBlock module.

        Args:
            in_channels (int): GNN nodes' input size.
            out_channels (int): GNN nodes' hidden size.
            gnn_conv (nn.Module): The GNN convolutional layer.
            gnn_conv_args (dict): Arguments for the GNN convolutional layer.
            **kwargs: Additional keyword arguments.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_connect = res_connect
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                gnn_conv(
                    in_channels=in_channels, out_channels=out_channels, **gnn_conv_args
                ),
                nn.BatchNorm1d(out_channels),
                # nn.LeakyReLU(),
                # gnn_conv(
                #     in_channels=out_channels, out_channels=out_channels, **gnn_conv_args
                # ),
                # nn.BatchNorm1d(out_channels),
            ]
        )

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass of the GNNBasicBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            torch.Tensor: Transformed graph representations.
            List[torch.Tensor]: List of attention weights if `return_attention_weights` is True, otherwise an empty list.
        """

        out = x

        attention_weights = []
        if "return_attention_weights" in kwargs:
            return_attention_weights = kwargs["return_attention_weights"]
        else:
            return_attention_weights = None

        for layer in self.layers:
            if isinstance(layer, self.gnn_conv):
                out = layer(
                    x=out,
                    edge_index=edge_index,
                    **kwargs
                )
                if return_attention_weights:
                    out, attention_weight = out
                    attention_weights.append(attention_weight)
            else:
                out = layer(out)

        if self.res_connect:
            out = out + x

        out = F.leaky_relu(out)

        # out = F.dropout(out, p=0.2)

        return out, attention_weights

class DGMLayer(nn.Module):

    def __init__(
        self, 
        encoder, 
        in_channels, 
        out_channels,
        k, 
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder(in_channels, out_channels)
        self.temperature = nn.Parameter(torch.tensor(4.0))
        self.k = k


    def sample_topk(self, k, logits):
        z = torch.distributions.Gumbel(0, 1).sample(logits.shape)
        z = z.to(logits.device)

        return torch.topk(logits + z, k, dim=-1)

    def forward(self, x, adj, **kwargs):
        # Graph Feature Learning
        if isinstance(self.encoder, MessagePassing):
            out = self.encoder(
                x=x, 
                edge_index=adj,
                **kwargs
            )
        else:
            out = self.encoder(x)

        cum_sum = 0
        edges_idx = torch.tensor([], device=out.device, dtype=torch.int32)
        graph_batch = kwargs['graph_batch']
        graph_map = graph_batch.batch
        for i in range(len(graph_batch.ptr) - 1):
        # for i, graph in enumerate(graph_batch.graph):
            # num_nodes = graph.size

            node_mask = graph_map == i
            num_nodes = node_mask.sum().item()
            node_mask = node_mask.long().argmax()


            graph_x = out[node_mask:(node_mask + num_nodes), :].detach()

            # Probabilistic Graph Generator
            X_i = graph_x[:, None, :]
            X_j = graph_x[None, :, :]
            d_e = ((X_i - X_j) ** 2).sum(-1)
            logits = torch.exp(- self.temperature * d_e).detach() # (N, N) prob. adj. matrix

            # Graph Sampling
            top_k = self.sample_topk(self.k, logits)
            edges_hat = top_k.indices
            # prob_adj_sampled = torch.gather(logits, 1, edges_hat)
            # graph.prob_adj_matrix.append(prob_adj_sampled)

            rows = torch.arange(
                num_nodes, 
                device=out.device
            ).view(num_nodes, 1).repeat(1, self.k)
            edges = torch.stack((rows.view(-1), edges_hat.view(-1)))
            edges_idx = torch.cat((edges_idx, edges + cum_sum), -1)

            cum_sum += num_nodes

        return out, edges_idx

class DGMGNNBasicBlock(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        gnn_conv, 
        gnn_conv_args, 
        lgi_op, 
        lgi_op_args,
        res_connect,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_connect = res_connect
        self.gnn_conv = gnn_conv
        self.lgi_op = lgi_op(**lgi_op_args) # Latent Graph Inference Op
        self.gnn = nn.ModuleList(
            [
                gnn_conv(
                    in_channels=in_channels, out_channels=out_channels, **gnn_conv_args
                ),
                nn.BatchNorm1d(out_channels),
                # nn.LeakyReLU(),
                # gnn_conv(
                #     in_channels=out_channels, out_channels=out_channels, **gnn_conv_args
                # ),
                # nn.BatchNorm1d(out_channels),
            ]
        )

    def forward(
            self, 
            x, 
            in_lgi, 
            edge_index, 
            graph_batch, 
            **kwargs
        ):
        """DGM-GNN block with skip connection."""
        out = x

        attention_weights = []
        if "return_attention_weights" in kwargs:
            return_attention_weights = kwargs["return_attention_weights"]
        else:
            return_attention_weights = None

        if in_lgi is not None:
            out_lgi = self.lgi_op(
                torch.cat((in_lgi[0], out), dim=-1), 
                edge_index, 
                graph_batch=graph_batch
            )
        else:
            out_lgi = self.lgi_op(
                x, 
                edge_index, 
                graph_batch=graph_batch
            )

        for layer in self.gnn:
            if isinstance(layer, self.gnn_conv):
                out = layer(
                    x=out,
                    edge_index=out_lgi[1],
                    **kwargs
                )
                if return_attention_weights:
                    out, attention_weight = out
                    attention_weights.append(attention_weight)
            else:
                out = layer(out)

        if self.res_connect:
            out = out + x

        out = F.leaky_relu(out)

        return out, attention_weights, out_lgi