import sys

import torch
import torch.nn as nn

from torch.nn import functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GATConv

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
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                gnn_conv(
                    in_channels=in_channels, out_channels=out_channels, **gnn_conv_args
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),
                gnn_conv(
                    in_channels=out_channels, out_channels=out_channels, **gnn_conv_args
                ),
                nn.BatchNorm1d(out_channels),
            ]
        )

    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass of the GNNBasicBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph edge indices.
            return_attention_weights (bool, optional): Whether to return attention weights. (Only for GAT-like convolution)

        Returns:
            torch.Tensor: Transformed graph representations.
            List[torch.Tensor]: List of attention weights if `return_attention_weights` is True, otherwise an empty list.
        """

        out = x
        attention_weights = []
        for layer in self.layers:
            if isinstance(layer, self.gnn_conv):
                out = layer(
                    x=out,
                    edge_index=edge_index,
                    return_attention_weights=return_attention_weights
                    if return_attention_weights
                    else None,
                )
                if return_attention_weights:
                    out, attention_weight = out
                    attention_weights.append(attention_weight)
            else:
                out = layer(out)

        out = out + x
        out = F.leaky_relu(out)

        return out, attention_weights

class StemGNNBlock(nn.Module):
    """
    Spectral-Temporal Graph Neural Network (StemGNN) basic block.
    """

    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernel_size,
            gft_bool,
            dft_bool,
            idft_bool,
            igft_bool,
            gnn_conv, 
            gnn_conv_args
        ):
        """
        Initialize the StemGNNBlock module.

        Args:
            in_channels (int): GNN nodes' input size.
            out_channels (int): GNN nodes' hidden size.
            kernel_size (int): The kernel size for the 1D convolution.
            gft_bool (bool): Whether to perform graph Fourier transform.
            dft_bool (bool): Whether to perform discrete Fourier transform.
            idft_bool (bool): Whether to perform inverse discrete Fourier transform.
            igft_bool (bool): Whether to perform inverse graph Fourier transform.
            gnn_conv (nn.Module): The GNN convolutional layer.
            gnn_conv_args (dict): Arguments for the GNN convolutional layer.
        """

        super().__init__()

        self.gft_op = gft_bool
        self.igft_op = igft_bool
        self.dft_op = dft_bool
        self.idft_op = idft_bool
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=1,
            kernel_size=kernel_size, 
            padding=1,
            dtype=torch.complex64
        )
        self.glu = nn.GLU()
        self.gnn_conv = GNNBasicBlock(
            in_channels, 
            out_channels, 
            gnn_conv, 
            gnn_conv_args
        )

    def forward(self, X, edge_index):
        """
        Forward pass of the StemGNNBlock module.

        Args:
            X (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph (batched-)adjacency matrix.

        Returns:
            torch.Tensor: Output tensor.
        """

        if self.gft_op:
            X = self._graph_fourier_transform(X, edge_index)
        if self.dft_op:
            X = torch.fft.fft(X)

        B, N, D = X.shape
        X = X.reshape(B * N, D).unsqueeze(1)
        out : torch.Tensor = self.conv1d(X)
        out_re : torch.Tensor = F.relu(out.real.squeeze(1))
        out_img : torch.Tensor = F.relu(out.imag.squeeze(1))

        out_re = self.glu(out_re)
        out_img = self.glu(out_img)

        out_re, _ = self.gnn_conv(out_re, edge_index)
        out_img, _ = self.gnn_conv(out_img, edge_index)
        out = torch.cat(
            (out_re.unsqueeze(-1), out_img.unsqueeze(-1)),
            dim=-1
        )
        out = torch.view_as_complex(out)
        out = out.reshape(B, N, -1)

        if self.idft_op:
            out = torch.fft.ifft(out).real

        if self.igft_op:
            out = self._inv_graph_fourier_transform(out, edge_index).real

        return out

    def _graph_fourier_transform(self, X : torch.Tensor, edge_index):
        """
        Perform graph Fourier transform.

        Args:
            X (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph (batched-)adjacency matrix.

        Returns:
            torch.Tensor: Transformed tensor.
        """

        B, N, D = X.shape

        laplacian = self._get_laplacian(edge_index)
        _, eigen_basis = torch.linalg.eig(laplacian)

        X = X.reshape(B * N, D).to(torch.complex64)
        eigen_basis = eigen_basis.squeeze(0)

        X_tilde = torch.matmul(eigen_basis, X).reshape(B, N, D)

        return X_tilde

    def _inv_graph_fourier_transform(self, X_tilde : torch.Tensor, edge_index):
        """
        Perform inverse graph Fourier transform.

        Args:
            X_tilde (torch.Tensor): Transformed tensor.
            edge_index (torch.Tensor): Graph (batched-)adjacency matrix.

        Returns:
            torch.Tensor: Inverse transformed tensor.
        """

        B, N, D = X_tilde.shape

        laplacian = self._get_laplacian(edge_index)
        _, eigen_basis = torch.linalg.eig(laplacian)

        X_tilde = X_tilde.reshape(B * N, D).to(torch.complex64) 
        eigen_basis = eigen_basis.squeeze(0)

        X = torch.matmul(eigen_basis.T, X_tilde).reshape(B, N, D)

        return X

    def _get_laplacian(self, edge_index):
        """
        Compute the Laplacian matrix.

        Args:
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            torch.Tensor: Laplacian matrix.
        """

        row = edge_index[0]
        num_nodes = maybe_num_nodes(edge_index)

        dense_adj = to_dense_adj(edge_index)

        edge_weights = torch.ones(edge_index.size(1)).to(edge_index.device)
        deg = scatter(edge_weights, row, 0, dim_size=num_nodes, reduce='sum')
        id = torch.eye(num_nodes).to(edge_index.device)
        deg = deg * id

        return deg - dense_adj

class NeroStemGNN(nn.Module):
    """
    Nero Spectral-Temporal Graph Neural Network (NeroStemGNN) module.
    """

    def __init__(
        self,
        in_channels,
        n_hidden,
        out_channels,
        gnn_conv,
        gnn_conv_args,
        layers=1,
        dropout=0.1
    ):
        """
        Initialize the NeroStemGNN module.

        Args:
            in_channels (int): GNN nodes' input size.
            n_hidden (int): GNN nodes' hidden size.
            out_channels (int): GNN nodes' output size.
            gnn_conv (str): GNN convolutional layer class name.
            gnn_conv_args (dict): Arguments for the GNN convolutional layer.
            layers (int): Number of StemGNNBlock layers.
            dropout (float): Dropout rate.
        """

        super().__init__()

        gnn_conv = getattr(sys.modules[__name__], gnn_conv)
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                StemGNNBlock(
                    in_channels=in_channels,
                    out_channels=n_hidden,
                    kernel_size=4,
                    gft_bool=True,
                    dft_bool=True, 
                    idft_bool=True, 
                    igft_bool=True,
                    gnn_conv=gnn_conv, 
                    gnn_conv_args=gnn_conv_args
                ),
                *[
                    StemGNNBlock(
                        in_channels=n_hidden // int(2 ** (l+1)),
                        out_channels=n_hidden // int(2 ** (l+1)),
                        kernel_size=3,
                        gft_bool=True,
                        dft_bool=True,
                        idft_bool=True, 
                        igft_bool=True,
                        gnn_conv=gnn_conv,
                        gnn_conv_args=gnn_conv_args
                    ) for l in range(layers-1)
                ],
                nn.Dropout(dropout),
                gnn_conv(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    **gnn_conv_args
                )
            ]
        )

    def forward(self, batch, **kwargs) -> torch.Tensor:
        """
        Forward pass of the NeroStemGNN module.

        Args:
            batch (Batch): Input batch.

        Returns:
            torch.Tensor: Output tensor.
        """

        out = torch.cat((batch.x, batch.draft), dim=-1)
        for layer in self.layers:
            if isinstance(layer, StemGNNBlock):
                out = layer(out, batch.edge_index)
            elif isinstance(layer, self.gnn_conv):
                B, N, F = out.shape
                out = out.reshape(B * N, F)
                out = global_mean_pool(
                    layer(out, batch.edge_index), 
                    batch.batch
                )
            else:
                out = layer(out)

        return out

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
        for i, graph in enumerate(graph_batch.graph):
            num_nodes = graph.size

            node_mask = graph_map == i
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
            prob_adj_sampled = torch.gather(logits, 1, edges_hat)
            graph.prob_adj_matrix.append(prob_adj_sampled)

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
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_conv = gnn_conv
        self.lgi_op = lgi_op(**lgi_op_args)
        self.gnn_block = GNNBasicBlock(
            in_channels, 
            out_channels, 
            gnn_conv, 
            gnn_conv_args
        )

    def forward(
            self, 
            x,  
            edge_index,
            in_lgi, 
            return_attention_weights=False,
            **kwargs
        ):
        """DGM-GNN block with skip connection."""
        out = x
        attention_weights = []

        out_lgi = self._latent_graph_inferece(
            out, 
            edge_index, 
            in_lgi, 
            **kwargs
        )

        out, attention_weights = self.gnn_block(
            out, out_lgi[1], 
            return_attention_weights
        )

        return out, attention_weights, out_lgi

    def _latent_graph_inferece(self, x, edge_index, in_lgi, **kwargs):
        if in_lgi is not None:
            return self.lgi_op(
                torch.cat((in_lgi[0], x), dim=-1), 
                edge_index, 
                graph_batch=kwargs['graph_batch']
            )
        else:
            return self.lgi_op(
                x, 
                edge_index, 
                graph_batch=kwargs['graph_batch']
            )