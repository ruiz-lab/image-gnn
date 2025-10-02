import sys
import yaml
import pickle
import argparse

import torch

import numpy as np

import matplotlib.pyplot as plt

from models.models import kNNModel
from utils.metrics import kl_div
from data_preproc.datasets import WANDataset

from sklearn.model_selection import StratifiedShuffleSplit

from annoy import AnnoyIndex

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from dataclasses import dataclass

from tqdm import tqdm
from time import time
from pathlib import Path


def smart_load(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


@dataclass
class Embedding:
    dataset: np.array
    train: bool


def build_datasets(ds_config):
    datasets = []
    for ds_type, configs in ds_config.items():
        datasets.append(
            Embedding(
                dataset=smart_load(configs["base_dir"]),
                train=True if ds_type == "train" else False
            )
        )
    return datasets


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bd",
        "--base_dir",
        type=str,
        help="Data file base dir",
        required=True
    )

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        help="Dataset configuration file",
        required=False
    )

    parser.add_argument(
        "-k",
        "--knn",
        type=str,
        help="Number of sampled neighbors",
        required=True
    )

    return parser.parse_args(args)


def build_edge_index(node_idx, neighb_idxs):
    source = torch.full((len(neighb_idxs),), node_idx, dtype=torch.long)
    target = torch.tensor(neighb_idxs, dtype=torch.long)
    return torch.stack([source, target], dim=0)

def kl_item_vec(Q, eps=1e-12):
    return np.log(np.clip(Q, eps, None)).astype(np.float64).ravel()

def kl_item_query(P, pi):
    return (pi * P).astype(np.float64).ravel()

def main(sys_args):
    args = parse_args(sys_args)

    k = int(args.knn)

    wan_ds = WANDataset(args.base_dir)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

    train_index = []
    test_index = []
    for train_idx, test_idx in sss.split(torch.zeros(len(wan_ds.targets)), wan_ds.targets):
        train_index.append(train_idx)
        test_index.append(test_idx)
    train_index = train_index[0]
    test_index = test_index[0]

    def build_ann_kl_model(matrices, n_trees=500, eps=1e-12):
        d = matrices[0].size

        index = AnnoyIndex(d, 'dot')
        for j, Q in enumerate(matrices):
            y = kl_item_vec(Q, eps)
            index.add_item(j, y.tolist())
        index.build(n_trees)

        return index

    model = build_ann_kl_model(wan_ds.data)

    edge_indices = []
    edge_weights = []
    for i in tqdm(range(len(wan_ds))):
        pi = wan_ds._get_limiting_distribution(wan_ds.data[i])[:, None]
        neighbors, distances = model.get_nns_by_vector(
            kl_item_query(
                wan_ds.data[i], 
                wan_ds._get_limiting_distribution(wan_ds.data[i])
            ), 
            k, 
            include_distances=True
        )
        neighbors = [j for j in neighbors if j != i]

        edge_indices.append(build_edge_index(i, neighbors))
        edge_weights.append(
            torch.tensor(
                [kl_div(wan_ds.data[i], wan_ds.data[neighbor], pi) for neighbor in neighbors]
            )
        )

    edge_index = torch.cat(edge_indices, dim=1)
    edge_weight = torch.cat(edge_weights, dim=0)

    edge_weight = edge_weight.view(-1)

    edge_index, edge_weight = to_undirected(
        edge_index,
        edge_weight,
        reduce='mean'
    )

    train_mask = torch.zeros(len(wan_ds), dtype=torch.bool)
    train_mask[train_index] = True
    test_mask = ~train_mask

    train_graph_data = Data(x=[wan_ds[i] for i in range(len(wan_ds))], edge_index=edge_index, edge_weight=edge_weight, y=wan_ds.targets, mask=train_mask)
    test_graph_data = Data(x=[wan_ds[i] for i in range(len(wan_ds))], edge_index=edge_index, edge_weight=edge_weight, y=wan_ds.targets, mask=test_mask)

    torch.save(train_graph_data, 'data/WANGraph/wan_train_knn_graph.pkl')
    torch.save(test_graph_data, 'data/WANGraph/wan_test_knn_graph.pkl')


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
