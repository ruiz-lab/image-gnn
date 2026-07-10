import os
import sys

import torch

import torch.nn.functional as F

import matplotlib.pyplot as plt

from pathlib import Path

from torch.nn import Linear, Sequential, ReLU

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader

from tqdm import tqdm

train_data = torch.load('data/FMNISTGraph/fmnist_train_knn_graph.pkl')
test_data = torch.load('data/FMNISTGraph/fmnist_test_knn_graph.pkl')

num_classes = train_data.y.unique().numel()

print(f"Train nodes: {train_data.num_nodes}, edges: {train_data.num_edges}")
print(f"Test nodes: {test_data.num_nodes}, edges: {test_data.num_edges}")
print(f"Input features: {train_data.num_node_features}, classes: {train_data.y.unique().numel()}")


edge_index = train_data.edge_index
mask = edge_index[0] != edge_index[1]
non_self_weights = train_data.edge_weight[mask]
plt.hist(non_self_weights.numpy(), bins=100)
plt.title("Edge weights (excluding self-edges)")

print("Train label distribution:", torch.bincount(train_data.y[train_data.mask.bool()]))
print("Test label distribution:", torch.bincount(test_data.y[test_data.mask.bool()]))

weights = train_data.edge_weight.cpu().numpy()
plt.hist(weights, bins=50)
plt.title("Post-threshold and normalized edge weights")
plt.show()

print("Number of edges:", train_data.edge_index.size(1))


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
def train(model, loader, use_edges):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
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

def evaluate(model, loader, use_edges):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
        loss = criterion(out[batch.mask.bool()], batch.y[batch.mask.bool()])
        total_loss += loss.item()
        pred = out[batch.mask.bool()].argmax(dim=1)
        correct += (pred == batch.y[batch.mask.bool()]).sum().item()
        total += batch.mask.sum().item()
    return total_loss / len(loader), correct / total

# ==== 5. Subsampling ====
def subsample(data, num_nodes):
    idx = torch.randperm(data.num_nodes)[:num_nodes]
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

# ==== 6. Main experiment ====
sizes = [train_data.num_nodes]
mlp_hidden = 128
criterion = torch.nn.CrossEntropyLoss()

mlp = MLP(train_data.num_node_features, mlp_hidden, num_classes)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
train_loader_full = NeighborLoader(train_data, input_nodes=train_data.mask.bool(), num_neighbors=[-1], batch_size=256)
test_loader = NeighborLoader(test_data, input_nodes=test_data.mask.bool(), num_neighbors=[-1], batch_size=512)

# ==== 6. Train MLP Once ====
print("Training MLP")
for epoch in range(6):
    train_loss, train_acc = train(mlp, train_loader_full, use_edges=False)
    if epoch == 5:
      test_loss, test_acc = evaluate(mlp, test_loader, use_edges=False)
print(f"MLP Final Acc: Train {train_acc:.4f}, Test {test_acc:.4f}")
mlp_gap = train_acc - test_acc
mlp_train_acc = train_acc
mlp_test_acc = test_acc

# ==== 7. Run GNN at multiple sizes ====
gnn_gaps = []
gnn_train_accs = []
gnn_test_accs = []

for sz in sizes:
    print(f"Training GNN on {sz} nodes")
    sub_data = subsample(train_data, sz)
    loader = NeighborLoader(sub_data, input_nodes=sub_data.mask.bool(), num_neighbors=[10, 10], batch_size=256, shuffle=True)

    model = GNN(train_data.num_node_features, hidden_channels=128, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(6):
        train_loss, train_acc = train(model, loader, use_edges=True)
        if epoch == 5:
          test_loss, test_acc = evaluate(model, test_loader, use_edges=True)

    gnn_train_accs.append(train_acc)
    gnn_test_accs.append(test_acc)
    gnn_gaps.append(train_acc - test_acc)
    print(f"GNN Accs: Train {train_acc:.4f}, Test {test_acc:.4f}, Gap {train_acc - test_acc:.4f}")

# ==== 8. Plot ====
plt.figure(figsize=(8, 5))
plt.plot(sizes, gnn_gaps, marker='o', label='GNN Gen Gap')
plt.hlines(mlp_gap, sizes[0], sizes[-1], colors='r', linestyles='dashed', label='MLP Gen Gap')
plt.xlabel("Training Graph Size")
plt.ylabel("Generalization Gap (Train Acc - Test Acc)")
plt.title("GNN vs MLP Generalization Gap")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('plot_1.png')

# ==== 9. Print Final Accuracies ====
print("\nFinal Accuracies:")
print(f"MLP     | Train: {mlp_train_acc:.4f} | Test: {mlp_test_acc:.4f}")
for sz, tr_acc, te_acc in zip(sizes, gnn_train_accs, gnn_test_accs):
    print(f"GNN ({sz}) | Train: {tr_acc:.4f} | Test: {te_acc:.4f}")


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

# ==== 3. MLP Model ====
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
def train(model, loader, use_edges, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
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

def evaluate(model, loader, use_edges):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
        loss = criterion(out[batch.mask.bool()], batch.y[batch.mask.bool()])
        total_loss += loss.item()
        pred = out[batch.mask.bool()].argmax(dim=1)
        correct += (pred == batch.y[batch.mask.bool()]).sum().item()
        total += batch.mask.sum().item()
    return total_loss / len(loader), correct / total

# ==== 5. Subsampling ====
def subsample(data, num_nodes):
    idx = torch.randperm(data.num_nodes)[:num_nodes]
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

# ==== 6. Main experiment ====
# Assume train_data and test_data are already loaded
# sizes = [375, 750, 1500, 3000, 6000, 12000]
sizes = [6000, 12000, 24000, 48000, 96000]
mlp_hidden = 128
criterion = torch.nn.CrossEntropyLoss()

mlp = MLP(train_data.num_node_features, mlp_hidden, num_classes)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
train_loader_full = NeighborLoader(train_data, input_nodes=train_data.mask.bool(), num_neighbors=[-1], batch_size=256)
test_loader = NeighborLoader(test_data, input_nodes=test_data.mask.bool(), num_neighbors=[-1], batch_size=512)

# ==== Train MLP ====
print("Training MLP")
for epoch in range(6):
    train_loss, train_acc = train(mlp, train_loader_full, use_edges=False, optimizer=optimizer)
    if epoch == 5:
      test_loss, test_acc = evaluate(mlp, test_loader, use_edges=False)
print(f"MLP Final Acc: Train {train_acc:.4f}, Test {test_acc:.4f}")
mlp_gap = train_acc - test_acc
mlp_loss_gap = train_loss - test_loss

# ==== GNNs on Fixed Sizes ====
gnn_gaps = []
gnn_loss_gaps = []
gnn_train_accs = []
gnn_test_accs = []
gnn_train_losses = []
gnn_test_losses = []

for sz in sizes:
    print(f"Training GNN on {sz} nodes")
    sub_data = subsample(train_data, sz)
    loader = NeighborLoader(sub_data, input_nodes=sub_data.mask.bool(), num_neighbors=[10, 10], batch_size=256, shuffle=True)

    model = GNN(train_data.num_node_features, hidden_channels=128, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(6):
        train_loss, train_acc = train(model, loader, use_edges=True, optimizer=optimizer)
        if epoch == 5:
          test_loss, test_acc = evaluate(model, test_loader, use_edges=True)

    gnn_train_accs.append(train_acc)
    gnn_test_accs.append(test_acc)
    gnn_train_losses.append(train_loss)
    gnn_test_losses.append(test_loss)
    gnn_gaps.append(train_acc - test_acc)
    gnn_loss_gaps.append(train_loss - test_loss)

    print(f"GNN Accs: Train {train_acc:.4f}, Test {test_acc:.4f}, Gap {train_acc - test_acc:.4f}")

# ==== GNN on Growing Subgraphs ====
print("Training GNN on growing subgraphs")
size_schedule = [750, 1500, 3000, 6000, 12000, 24000]

cached_subgraphs = {}
for sz in set(size_schedule):
    cached_subgraphs[sz] = subsample(train_data, sz)

model_seq = GNN(train_data.num_node_features, hidden_channels=128, out_channels=num_classes)
optimizer_seq = torch.optim.Adam(model_seq.parameters(), lr=0.01)

seq_train_accs = []
seq_test_accs = []
seq_train_losses = []
seq_test_losses = []

for sz in size_schedule:
    sub_data = cached_subgraphs[sz]
    loader = NeighborLoader(sub_data, input_nodes=sub_data.mask.bool(), num_neighbors=[10, 10], batch_size=256, shuffle=True)

    train_loss, train_acc = train(model_seq, loader, use_edges=True, optimizer=optimizer_seq)
    test_loss, test_acc = evaluate(model_seq, test_loader, use_edges=True)

    seq_train_accs.append(train_acc)
    seq_test_accs.append(test_acc)
    seq_train_losses.append(train_loss)
    seq_test_losses.append(test_loss)

    print(f"[Size {sz}] Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

final_seq_train_acc = seq_train_accs[-1]
final_seq_test_acc = seq_test_accs[-1]
seq_gap = final_seq_train_acc - final_seq_test_acc
seq_loss_gap = seq_train_losses[-1] - seq_test_losses[-1]
print(f"Growing Graph GNN Final Acc: Train {final_seq_train_acc:.4f}, Test {final_seq_test_acc:.4f}, Gap {seq_gap:.4f}")

# ==== Plot Accuracy Generalization Gap ====
plt.figure(figsize=(8, 5))
plt.plot(sizes, gnn_gaps, marker='o', label='GNN Gen Gap')
plt.hlines(mlp_gap, sizes[0], sizes[-1], colors='r', linestyles='dashed', label='MLP Gen Gap')
plt.hlines(seq_gap, sizes[0], sizes[-1], colors='g', linestyles='dashdot', label='Growing Subgraph GNN Gen Gap')
plt.xlabel("Training Graph Size")
plt.ylabel("Generalization Gap (Train Acc - Test Acc)")
plt.title("GNN vs MLP Generalization Gap")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plot_2.png')

# ==== Plot Loss Curve for Growing GNN ====
plt.figure(figsize=(8, 5))
plt.plot(seq_train_losses, label='Train Loss')
plt.plot(seq_test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Growing Subgraph GNN: Train/Test Loss")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plot_3.png')

# ==== Plot Loss Generalization Gap ====
plt.figure(figsize=(8, 5))
plt.plot(sizes, gnn_loss_gaps, marker='s', label='GNN Loss Gap')
plt.hlines(mlp_loss_gap, sizes[0], sizes[-1], colors='r', linestyles='dashed', label='MLP Loss Gap')
plt.hlines(seq_loss_gap, sizes[0], sizes[-1], colors='g', linestyles='dashdot', label='Growing Subgraph GNN Loss Gap')
plt.xlabel("Training Graph Size")
plt.ylabel("Loss Generalization Gap (Train Loss - Test Loss)")
plt.title("GNN vs MLP Loss Generalization Gap")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plot_4.png')
