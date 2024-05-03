import sys

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.nn import functional as F

from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from itertools import product

from torch_geometric.data import Data

from pathlib import Path
from itertools import product
from dataclasses import dataclass, field

from typing import List, Dict


def build_datasets(ds_config):
    datasets  = []
    for ds_type, configs in ds_config.items():
        ds_class = getattr(sys.modules[__name__], configs["dataset_class"])
        datasets.append(ds_class(**configs))

    return datasets


class MNISTDataset(MNIST):
    def __init__(
        self, 
        root="data/", 
        train=True, 
        **kwargs
    ):
        super().__init__(root, train, download=True, transform=Compose([ToTensor()]))

    def __getitem__(self, index):
        X, y = super().__getitem__(index)

        data = Data()
        data.x = torch.flatten(X, start_dim=1)
        data.y = torch.flatten(X, start_dim=1)

        return data

    def __len__(self):
        return super().__len__()

class MNISTEmbeddedDataset(Dataset):

    def __init__(
        self, 
        base_dir,
        ker_width=10,
        graph_connectivity="fully_connected",
        **kwargs
    ):
        super().__init__()

        self.base_dir = base_dir
        self.kernel_width = ker_width
        self.graph_connectivity = graph_connectivity
        self.data = self._get_data()

    def _get_data(self):
        return np.load(Path(self.base_dir))

    def _build_graph(self, data, graph_connectivity, index, sample_size=100):
        graph_data = Data()

        pop_size = data.shape[0]
        idx_prob = [1 / (pop_size - 1)] * pop_size
        idx_prob[index] = 0
        idx_sample = np.random.choice(pop_size, sample_size-1, replace=False, p=idx_prob)
        idx_sample = np.append(idx_sample, index)

        graph_data.data = torch.tensor(data[idx_sample, :-1], dtype=torch.float32)
        graph_data.y = torch.tensor(data[idx_sample, -1], dtype=torch.int64)
        graph_data.num_nodes = graph_data.data.shape[0]

        X_i = graph_data.data[:, None, :]
        X_j = graph_data.data[None, :, :]
        d_e = ((X_i - X_j) ** 2).sum(-1)

        graph_data.edge_weights = torch.exp(- d_e / (self.kernel_width ** 2))

        if graph_connectivity == "fully_connected":
            graph_data.edge_index = torch.from_numpy(
                np.array(
                    list(
                        product(
                            range(graph_data.num_nodes), 
                            range(graph_data.num_nodes)
                        )
                    )
                ).T
            ).to(torch.int32)

        return graph_data

    def __getitem__(self, index):
        return self._build_graph(
            self.data, 
            self.graph_connectivity, 
            index
        )

    def __len__(self):
        return len(self.data)