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

        # if self.res_connect:
        #     out = out + x

        out = F.leaky_relu(out)

        return out, attention_weights