import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import sys
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import subgraph
from tqdm import tqdm


TRAIN_GRAPH_PATH = "data/ResultGraphs/cifar10_train_knn_graph-1000.pkl"
TEST_GRAPH_PATH = "data/ResultGraphs/cifar10_test_knn_graph-1000.pkl"

SIZES = [2000, 5000, 10000, 30000, 60000]  # train node counts
EPOCHS = 20
HIDDEN_DIM = 128
EMBED_DIM = 128
LR = 1e-3
WEIGHT_DECAY = 5e-4
SEED = 42

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
TRAIN_NUM_NEIGHBORS = [10, 10]
TEST_NUM_NEIGHBORS = [10, 10]  # [-1, -1] for full k-hop neighborhoods for eval


# Setting seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model definitions
class HeavyDropout8LayerCNN(nn.Module):
    """8-layer CNN encoder that maps image -> embedding."""
    def __init__(self, out_dim: int = 128, dropout_p: float = 0.5):
        super().__init__()
        dp = dropout_p
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(dp),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(dp),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(dp),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 8 -> 4

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.Dropout2d(dp),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dp),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-style message passing with edge weights."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin = Linear(in_channels, out_channels)
        self.root_lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def update(self, aggr_out, x):
        return self.lin(aggr_out) + self.root_lin(x)

class EmbeddingToImageAdapter(nn.Module):
    def __init__(self, in_dim, out_shape=(3, 32, 32), hidden=2048, p=0.2):
        super().__init__()
        c, h, w = out_shape
        self.c, self.h, self.w = c, h, w
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, c * h * w),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(-1, self.c, self.h, self.w)

class CNNGNNModel(nn.Module):
    """CNN encoder + 2-layer weighted GraphSAGE classifier."""
    def __init__(self, hidden_dim: int, num_classes: int, embed_dim: int = 128, x_mode: str = "image", in_dim: int = None):
        super().__init__()
        self.adapter = None
        if x_mode == "embedding":
            if in_dim is None:
                raise ValueError("in_dim is required when x_mode='embedding'")
            self.adapter = EmbeddingToImageAdapter(in_dim=in_dim, out_shape=(3, 32, 32))

        self.encoder = HeavyDropout8LayerCNN(out_dim=embed_dim)
        self.conv1 = WeightedSAGEConv(embed_dim, hidden_dim)
        self.conv2 = WeightedSAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight):
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.encoder(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


# Data helpers
def ensure_graph_fields(data: Data):
    if not hasattr(data, "mask"):
        raise ValueError("Graph is missing 'mask' field.")
    data.mask = data.mask.bool()

    if not hasattr(data, "edge_weight") or data.edge_weight is None:
        data.edge_weight = torch.ones(
            data.edge_index.size(1),
            dtype=torch.float32,
            device=data.edge_index.device
        )

    if data.x.ndim == 4:
        if data.x.size(1) != 3:
            raise ValueError(f"Expected 3-channel images for CNN, got C={data.x.size(1)}")
        return {"x_mode": "image", "in_dim": None}

    # embedding features
    if data.x.ndim == 2:
        return {"x_mode": "embedding", "in_dim": int(data.x.size(1))}

    raise ValueError(f"Unsupported data.x shape: {tuple(data.x.shape)}")


def build_train_subgraph(data: Data, requested_nodes: int) -> Data:
    """
    Sample from masked train nodes, then build induced subgraph.
    Returned graph has:
    - relabeled nodes [0..k-1]
    - x/y sliced to sampled nodes
    - mask all True (all nodes in this mini-train-graph are supervised)
    """
    train_nodes = data.mask.nonzero(as_tuple=True)[0]
    if train_nodes.numel() == 0:
        raise ValueError("No train nodes in data.mask.")

    k = min(requested_nodes, train_nodes.numel())
    perm = torch.randperm(train_nodes.numel(), device=train_nodes.device)[:k]
    keep_nodes = train_nodes[perm]

    sub_edge_index, sub_edge_weight = subgraph(
        subset=keep_nodes,
        edge_index=data.edge_index,
        edge_attr=data.edge_weight,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    out = Data(
        x=data.x[keep_nodes],
        y=data.y[keep_nodes],
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        mask=torch.ones(k, dtype=torch.bool, device=data.x.device),
    )
    return out


# Train / eval functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Train", leave=False, file=sys.__stdout__):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_weight)
        m = batch.mask.bool()
        loss = criterion(out[m], batch.y[m])
        loss.backward()
        optimizer.step()

        n = int(m.sum().item())
        total_loss += loss.item() * n
        total_count += n

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        m = batch.mask.bool()
        pred = out[m].argmax(dim=1)
        correct += (pred == batch.y[m]).sum().item()
        total += int(m.sum().item())

    return correct / max(total, 1)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# Main transferability experiment
def main():
    set_seed(SEED)

    log_path = "cifar_transfer-true.txt"
    log_file = open(log_path, "w", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        train_data = torch.load(TRAIN_GRAPH_PATH, weights_only=False)
        test_data = torch.load(TEST_GRAPH_PATH, weights_only=False)

        train_info = ensure_graph_fields(train_data)
        test_info = ensure_graph_fields(test_data)
        if train_info["x_mode"] != test_info["x_mode"]:
            raise ValueError(
                f"Train/test x mode mismatch: {train_info['x_mode']} vs {test_info['x_mode']}"
            )
        x_mode = train_info["x_mode"]
        in_dim = train_info["in_dim"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        train_data = train_data.to(device)
        test_data = test_data.to(device)

        num_classes = int(train_data.y[train_data.mask].unique().numel())
        available_train = int(train_data.mask.sum().item())

        print(f"Train graph: nodes={train_data.num_nodes}, edges={train_data.num_edges}, supervised={available_train}")
        print(f"Test graph:  nodes={test_data.num_nodes}, edges={test_data.num_edges}, supervised={int(test_data.mask.sum().item())}")
        print(f"Num classes: {num_classes}")

        test_loader = NeighborLoader(
            test_data,
            input_nodes=test_data.mask,
            num_neighbors=TEST_NUM_NEIGHBORS,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss()
        results = []

        for requested_size in SIZES:
            actual_size = min(requested_size, available_train)
            print(f"\n=== Transferability: train on {actual_size} nodes (requested {requested_size}) ===")

            sub_train = build_train_subgraph(train_data, actual_size)

            train_loader = NeighborLoader(
                sub_train,
                input_nodes=sub_train.mask,
                num_neighbors=TRAIN_NUM_NEIGHBORS,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
            )

            model = CNNGNNModel(
                hidden_dim=HIDDEN_DIM,
                num_classes=num_classes,
                embed_dim=EMBED_DIM,
                x_mode=x_mode,
                in_dim=in_dim,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f}")

            test_acc = evaluate(model, test_loader, device)
            print(f"Full-test accuracy: {test_acc:.4f}")

            results.append((actual_size, test_acc))

        print("\nSummary (train_size -> test_acc_on_full_test_graph):")
        for sz, acc in results:
            print(f"{sz:>7d} -> {acc:.4f}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"Saved full run log to: {log_path}")


if __name__ == "__main__":
    main()



import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import sys
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import subgraph
from tqdm import tqdm


TRAIN_GRAPH_PATH = "data/ResultGraphs/FMNIST_train_knn_graph.pkl"
TEST_GRAPH_PATH = "data/ResultGraphs/FMNIST_test_knn_graph.pkl"

SIZES = [2000, 7000, 15000, 35000, 70000]  # train node counts
EPOCHS = 20
HIDDEN_DIM = 128
EMBED_DIM = 128
LR = 1e-3
WEIGHT_DECAY = 5e-4
SEED = 42

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
TRAIN_NUM_NEIGHBORS = [10, 10]
TEST_NUM_NEIGHBORS = [10, 10]  # [-1, -1] for full k-hop neighborhoods for eval


# Setting seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model definitions
class HeavyDropout8LayerCNN(nn.Module):
    """8-layer CNN encoder that maps image -> embedding."""
    def __init__(self, out_dim: int = 128, dropout_p: float = 0.5):
        super().__init__()
        dp = dropout_p
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(dp),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(dp),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(dp),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 8 -> 4

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.Dropout2d(dp),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dp),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-style message passing with edge weights."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin = Linear(in_channels, out_channels)
        self.root_lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def update(self, aggr_out, x):
        return self.lin(aggr_out) + self.root_lin(x)


class EmbeddingToImageAdapter(nn.Module):
    def __init__(self, in_dim, out_shape=(3, 32, 32), hidden=2048, p=0.2):
        super().__init__()
        c, h, w = out_shape
        self.c, self.h, self.w = c, h, w
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, c * h * w),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(-1, self.c, self.h, self.w)


class CNNGNNModel(nn.Module):
    """CNN encoder + 2-layer weighted GraphSAGE classifier."""
    def __init__(self, hidden_dim: int, num_classes: int, embed_dim: int = 128, x_mode: str = "image", in_dim: int = None):
        super().__init__()
        self.adapter = None
        if x_mode == "embedding":
            if in_dim is None:
                raise ValueError("in_dim is required when x_mode='embedding'")
            self.adapter = EmbeddingToImageAdapter(in_dim=in_dim, out_shape=(3, 32, 32))

        self.encoder = HeavyDropout8LayerCNN(out_dim=embed_dim)
        self.conv1 = WeightedSAGEConv(embed_dim, hidden_dim)
        self.conv2 = WeightedSAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight):
        if self.adapter is not None:
            x = self.adapter(x)
        x = self.encoder(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


# Data helpers
def ensure_graph_fields(data: Data):
    if not hasattr(data, "mask"):
        raise ValueError("Graph is missing 'mask' field.")
    data.mask = data.mask.bool()

    if not hasattr(data, "edge_weight") or data.edge_weight is None:
        data.edge_weight = torch.ones(
            data.edge_index.size(1),
            dtype=torch.float32,
            device=data.edge_index.device
        )

    if data.x.ndim == 4:
        if data.x.size(1) != 3:
            raise ValueError(f"Expected 3-channel images for CNN, got C={data.x.size(1)}")
        return {"x_mode": "image", "in_dim": None}

    if data.x.ndim == 2:
        return {"x_mode": "embedding", "in_dim": int(data.x.size(1))}

    raise ValueError(f"Unsupported data.x shape: {tuple(data.x.shape)}")


def build_train_subgraph(data: Data, requested_nodes: int) -> Data:
    train_nodes = data.mask.nonzero(as_tuple=True)[0]
    if train_nodes.numel() == 0:
        raise ValueError("No train nodes in data.mask.")

    k = min(requested_nodes, train_nodes.numel())
    perm = torch.randperm(train_nodes.numel(), device=train_nodes.device)[:k]
    keep_nodes = train_nodes[perm]

    sub_edge_index, sub_edge_weight = subgraph(
        subset=keep_nodes,
        edge_index=data.edge_index,
        edge_attr=data.edge_weight,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    out = Data(
        x=data.x[keep_nodes],
        y=data.y[keep_nodes],
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        mask=torch.ones(k, dtype=torch.bool, device=data.x.device),
    )
    return out


# Train / eval
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Train", leave=False, file=sys.__stdout__):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_weight)
        m = batch.mask.bool()
        loss = criterion(out[m], batch.y[m])
        loss.backward()
        optimizer.step()

        n = int(m.sum().item())
        total_loss += loss.item() * n
        total_count += n

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        m = batch.mask.bool()
        pred = out[m].argmax(dim=1)
        correct += (pred == batch.y[m]).sum().item()
        total += int(m.sum().item())

    return correct / max(total, 1)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# Main transferability experiment
def main():
    set_seed(SEED)

    log_path = "fmnist_transfer-true.txt"
    log_file = open(log_path, "w", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        train_data = torch.load(TRAIN_GRAPH_PATH, weights_only=False)
        test_data = torch.load(TEST_GRAPH_PATH, weights_only=False)

        train_info = ensure_graph_fields(train_data)
        test_info = ensure_graph_fields(test_data)
        if train_info["x_mode"] != test_info["x_mode"]:
            raise ValueError(
                f"Train/test x mode mismatch: {train_info['x_mode']} vs {test_info['x_mode']}"
            )
        x_mode = train_info["x_mode"]
        in_dim = train_info["in_dim"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        train_data = train_data.to(device)
        test_data = test_data.to(device)

        num_classes = int(train_data.y[train_data.mask].unique().numel())
        available_train = int(train_data.mask.sum().item())

        print(f"Train graph: nodes={train_data.num_nodes}, edges={train_data.num_edges}, supervised={available_train}")
        print(f"Test graph:  nodes={test_data.num_nodes}, edges={test_data.num_edges}, supervised={int(test_data.mask.sum().item())}")
        print(f"Num classes: {num_classes}")

        test_loader = NeighborLoader(
            test_data,
            input_nodes=test_data.mask,
            num_neighbors=TEST_NUM_NEIGHBORS,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss()
        results = []

        for requested_size in SIZES:
            actual_size = min(requested_size, available_train)
            print(f"\n=== Transferability: train on {actual_size} nodes (requested {requested_size}) ===")

            sub_train = build_train_subgraph(train_data, actual_size)

            train_loader = NeighborLoader(
                sub_train,
                input_nodes=sub_train.mask,
                num_neighbors=TRAIN_NUM_NEIGHBORS,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
            )

            model = CNNGNNModel(
                hidden_dim=HIDDEN_DIM,
                num_classes=num_classes,
                embed_dim=EMBED_DIM,
                x_mode=x_mode,
                in_dim=in_dim,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f}")
                log_file.write(f"size={actual_size} epoch={epoch:02d} train_loss={train_loss:.6f}\n")

            test_acc = evaluate(model, test_loader, device)
            print(f"Full-test accuracy: {test_acc:.4f}")
            log_file.write(f"summary size={actual_size} test_acc={test_acc:.6f}\n")

            results.append((actual_size, test_acc))

        print("\nSummary (train_size -> test_acc_on_full_test_graph):")
        for sz, acc in results:
            print(f"{sz:>7d} -> {acc:.4f}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"Saved full run log to: {log_path}")


if __name__ == "__main__":
    main()
