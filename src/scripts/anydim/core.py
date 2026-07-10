"""Core model and training building blocks for the any-dimensional experiments.

Everything in this module is lifted verbatim (behaviourally) from
``src/scripts/neurips_exps.py`` so that the baseline experiment arms -- the MLP,
the fixed-size GNNs, and the linearly growing-subgraph GNN -- reproduce the
results previously reported in the paper. The only additions are a ``device``
argument threaded through ``train``/``evaluate`` (so the four arms can run on
separate GPUs) and a ``seed`` helper for reproducibility. Neither changes the
numerical behaviour when a single device / fixed seed is used.
"""

import random

import numpy as np

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import NeighborLoader

from tqdm import tqdm


# ==== 1. Weighted SAGE Conv ====
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = Linear(in_channels, out_channels)
        self.root_lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def update(self, aggr_out, x):
        return self.lin(aggr_out) + self.root_lin(x)


# ==== 2. GNN Model ====
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden_channels)
        self.conv2 = WeightedSAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        #x = self.conv2(x, edge_index, edge_weight)
        #x = F.relu(x)
        return self.lin(x)


# ==== 3. MLP Model (matched param count) ====
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)


# ==== 4. Training and Evaluation ====
def train(model, loader, use_edges, optimizer, criterion, device="cpu"):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
        loss = criterion(out[batch.mask.bool()], batch.y[batch.mask.bool()])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out[batch.mask.bool()].argmax(dim=1)
        correct += (pred == batch.y[batch.mask.bool()]).sum().item()
        total += batch.mask.sum().item()
    return total_loss / len(loader), correct / total


def evaluate(model, loader, use_edges, criterion, device="cpu"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
            loss = criterion(out[batch.mask.bool()], batch.y[batch.mask.bool()])
            total_loss += loss.item()
            pred = out[batch.mask.bool()].argmax(dim=1)
            correct += (pred == batch.y[batch.mask.bool()]).sum().item()
            total += batch.mask.sum().item()
    return total_loss / len(loader), correct / total


# ==== 5. Subsampling ====
def subsample(data, num_nodes, generator=None):
    idx = torch.randperm(data.num_nodes, generator=generator)[:num_nodes]
    mask = torch.zeros_like(data.mask, dtype=torch.bool)
    mask[idx] = True
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    return Data(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index[:, edge_mask],
        edge_weight=data.edge_weight[edge_mask],
        mask=mask
    )


def load_graph(path):
    """Load a graph pkl and normalize its edge-weight attribute.

    Datasets are inconsistent: most store ``edge_weight`` (singular), but some
    (e.g. CIFAR10) store ``edge_weights`` (plural), leaving ``data.edge_weight``
    as ``None``. ``WeightedSAGEConv`` and ``subsample`` both index
    ``edge_weight``, so we normalize here:
      1. use ``edge_weight`` if present and non-None,
      2. else fall back to ``edge_weights``,
      3. else default to all-ones (unweighted graph).
    """
    data = torch.load(path, weights_only=False)
    ew = getattr(data, "edge_weight", None)
    if ew is None:
        ew = getattr(data, "edge_weights", None)
    if ew is None:
        ew = torch.ones(data.edge_index.size(1), dtype=torch.float32)
    data.edge_weight = ew
    return data


def make_loader(data, num_neighbors, batch_size, shuffle=False):
    """NeighborLoader over the labelled (masked) nodes of ``data``."""
    return NeighborLoader(
        data,
        input_nodes=data.mask.bool(),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def set_seed(seed):
    """Seed python / numpy / torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
