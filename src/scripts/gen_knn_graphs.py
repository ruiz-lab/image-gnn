import sys

import yaml

import argparse

import numpy as np

import torch

from tqdm import tqdm

from time import time

from models.models import kNNModel

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

from dataclasses import dataclass


@dataclass
class Embedding:
    dataset: np.array
    train: bool


def build_datasets(ds_config):
    datasets  = []
    for ds_type, configs in ds_config.items():
        datasets.append(
            Embedding(
                dataset=np.load(configs["base_dir"]),
                train=True if ds_type == "train" else False
            )
        )

    return datasets


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_config", 
        type=str, 
        help="Training configuration file", 
        required=True
    )

    parser.add_argument(
        "-k",
        "--knn", 
        type=str, 
        help="Number of sampled neighbors", 
        required=True
    )
 
    return parser.parse_args(args)


# def build_edge_index(node_idx, neighb_idxs):
#     s = np.array([[node_idx] * len(neighb_idxs)], dtype=int)
#     r = [neighb_idxs]

#     return torch.tensor(np.append(s, r, axis=0))

def build_edge_index(node_idx, neighb_idxs):
    source = torch.full((len(neighb_idxs),), node_idx, dtype=torch.long)
    target = torch.tensor(neighb_idxs, dtype=torch.long)
    return torch.stack([source, target], dim=0)


def main(sys_args):
    args = parse_args(sys_args)

    k = int(args.knn)
    ker_width = 5

    with open(args.dataset_config, 'r') as f:
        ds_config = yaml.safe_load(f)

    ds = build_datasets(ds_config)
    # ds[0].dataset = ds[0].dataset[:2500, :]
    # ds[1].dataset = ds[1].dataset[:2500, :]
    full_ds = np.append(ds[0].dataset, ds[1].dataset, 0)
    model = kNNModel(
        ds=full_ds[:, :-2],
        n_trees=100000
    )

    x = torch.empty(full_ds[:, :-2].shape)
    y = torch.tensor(full_ds[:, -2], dtype=torch.long)

    edge_indices = []
    edge_weights = []
    for i in tqdm(range(len(model.indexes))):
        neighbors, distances = model(i, k)
        edge_indices.append(build_edge_index(i, neighbors))
        edge_weights.append(torch.exp(-torch.tensor(distances) / (ker_width ** 2)))

    edge_index = torch.cat(edge_indices, dim=1)
    edge_weight = torch.cat(edge_weights, dim=0)

    edge_weight = edge_weight.view(-1)
    edge_weight[edge_weight < 0.75] = 0.0

    edge_index, edge_weight = to_undirected(
        edge_index, 
        edge_weight, 
        reduce='mean'
    )

    train_mask = torch.ones((full_ds.shape[0],), dtype=torch.long)
    train_mask[ds[0].dataset.shape[0]:] = torch.zeros(
        (ds[1].dataset.shape[0],), 
        dtype=torch.long
    )

    test_mask = torch.zeros((full_ds.shape[0]), dtype=torch.long)
    test_mask[ds[0].dataset.shape[0]:] = torch.ones(
        (ds[1].dataset.shape[0],), 
        dtype=torch.long
    )

    train_graph_data = Data(x, edge_index, edge_weights=edge_weight, y=y, mask=train_mask)
    test_graph_data = Data(x, edge_index, edge_weights=edge_weight, y=y, mask=test_mask)

    torch.save(train_graph_data, 'data/CIFAR10Graph/cifar10_train_knn_graph-2.pkl')
    torch.save(test_graph_data, 'data/CIFAR10Graph/cifar10_test_knn_graph-2.pkl')


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))