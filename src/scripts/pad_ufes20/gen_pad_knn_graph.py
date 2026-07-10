"""
Build kNN data-manifold graphs for PAD-UFES-20 embeddings.

Path-parametrized variant of src/scripts/gen_knn_graphs.py. Consumes the
SmSL-tagged train/test embedding files (rows = [latent..., label, tag]), builds
an approximate-kNN graph with Gaussian-kernel edge weights, thresholds weak
edges, symmetrizes, and saves train/test PyG Data graphs.

Usage:
    python src/scripts/pad_ufes20/gen_pad_knn_graph.py \
        --train data/PadUfes20Embeddings/smsl_embeddings/pad_smsl_train_embeddings.npy \
        --test  data/PadUfes20Embeddings/smsl_embeddings/pad_smsl_test_embeddings.npy \
        --out_dir data/PadUfes20Graph \
        -k 100
"""
import sys
import argparse

from pathlib import Path

import torch
import numpy as np

# make src/scripts importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.models import kNNModel

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/PadUfes20Graph")
    parser.add_argument("-k", "--knn", type=int, required=True)
    parser.add_argument(
        "--ker_width",
        type=str,
        default="5.0",
        help="Gaussian-kernel width: a number for a hardcoded value (default 5.0), "
             "or 'auto' to use the median-distance heuristic "
             "(ker_width = sqrt(median neighbor distance)).",
    )
    parser.add_argument(
        "--auto_percentile",
        type=float,
        default=50.0,
        help="Percentile of neighbor distances used by --ker_width auto (default 50 = median).",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--n_trees", type=int, default=100000)
    return parser.parse_args(args)


def build_edge_index(node_idx, neighb_idxs):
    source = torch.full((len(neighb_idxs),), node_idx, dtype=torch.long)
    target = torch.tensor(neighb_idxs, dtype=torch.long)
    return torch.stack([source, target], dim=0)


def main(sys_args):
    args = parse_args(sys_args)
    k = args.knn
    auto_ker_width = str(args.ker_width).strip().lower() == "auto"

    train_ds = np.load(Path(args.train), allow_pickle=True)
    test_ds = np.load(Path(args.test), allow_pickle=True)
    full_ds = np.append(train_ds, test_ds, 0)

    model = kNNModel(ds=full_ds[:, :-2], n_trees=args.n_trees)

    x = torch.tensor(full_ds[:, :-2], dtype=torch.float)
    y = torch.tensor(full_ds[:, -2], dtype=torch.long)

    # First pass: collect all neighbor lists and their distances.
    all_neighbors = []
    all_distances = []
    for i in tqdm(range(len(model.indexes))):
        neighbors, distances = model(i, k)
        all_neighbors.append(neighbors)
        all_distances.append(torch.tensor(distances))

    flat_dist = torch.cat(all_distances, dim=0)
    # Exclude self-distance (0) when picking the kernel width so it reflects
    # the actual neighbor scale.
    nonzero_dist = flat_dist[flat_dist > 0]

    if auto_ker_width:
        # Median-distance (RBF) heuristic: ker_width = sqrt(percentile distance),
        # so the median edge gets weight exp(-1) ~= 0.37.
        pctl = float(torch.quantile(nonzero_dist, args.auto_percentile / 100.0))
        ker_width = float(np.sqrt(max(pctl, 1e-12)))
        print(f"[auto ker_width] {args.auto_percentile:.0f}th-pctile neighbor dist = {pctl:.4f} "
              f"-> ker_width = {ker_width:.4f}")
    else:
        ker_width = float(args.ker_width)
        print(f"[fixed ker_width] {ker_width:.4f}")

    edge_indices = []
    edge_weights = []
    for i, (neighbors, distances) in enumerate(zip(all_neighbors, all_distances)):
        edge_indices.append(build_edge_index(i, neighbors))
        edge_weights.append(torch.exp(-distances / (ker_width ** 2)))

    edge_index = torch.cat(edge_indices, dim=1)
    edge_weight = torch.cat(edge_weights, dim=0).view(-1)
    edge_weight[edge_weight < args.threshold] = 0.0

    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")

    train_mask = torch.zeros(full_ds.shape[0], dtype=torch.bool)
    train_mask[:train_ds.shape[0]] = True
    test_mask = ~train_mask

    train_graph_data = Data(x, edge_index, edge_weight=edge_weight, y=y, mask=train_mask)
    test_graph_data = Data(x, edge_index, edge_weight=edge_weight, y=y, mask=test_mask)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Encode the actual hyperparameters used into the filename so the graph is
    # self-documenting: k, kernel width (auto-resolved value), threshold.
    kw_str = f"auto{ker_width:.2f}" if auto_ker_width else f"{ker_width:.2f}"
    tag = f"k{k}_kw{kw_str}_thr{args.threshold:.2f}"
    train_path = out_dir / f"pad_train_knn_graph-{tag}.pkl"
    test_path = out_dir / f"pad_test_knn_graph-{tag}.pkl"

    torch.save(train_graph_data, train_path)
    torch.save(test_graph_data, test_path)

    n_nonzero = int((edge_weight > 0).sum())
    print(f"nodes: {full_ds.shape[0]}, edges: {edge_index.shape[1]}, "
          f"nonzero-weight edges (after threshold {args.threshold}): {n_nonzero}")
    print(f"train graph -> {train_path}")
    print(f"test  graph -> {test_path}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
