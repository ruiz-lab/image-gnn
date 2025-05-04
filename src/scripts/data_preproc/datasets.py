import sys

import re

import csv

import pathlib

import numpy as np

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

from PIL import Image

from medmnist import PathMNIST

from torch.utils.data import Dataset

from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA, VisionDataset
from torchvision.transforms import ToTensor, Normalize, Compose

from torch_geometric.data import Data

from pathlib import Path
from itertools import product
from dataclasses import dataclass, field

from time import time

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
        super().__init__(
            root, 
            train, 
            download=True, 
            transform=Compose([ToTensor()])
        )

    def __getitem__(self, index):
        X, y = super().__getitem__(index)

        data = Data()
        # data.x = torch.flatten(X, start_dim=1)
        # data.y = torch.flatten(X, start_dim=1)
        data.x = X[None, ...]
        data.y = X[None, ...]

        return data

    def __len__(self):
        return super().__len__()


class CIFAR10Dataset(CIFAR10):
    def __init__(
        self, 
        root="data/", 
        train=True, 
        **kwargs
    ):
        super().__init__(
            root, 
            train, 
            download=True, 
            transform=Compose([
                ToTensor(),
                Normalize((0.5,), (0.5,))
            ])
        )
        self.targets = torch.tensor(self.targets)
        self.data_var = np.var(self.data / 255)

    def __getitem__(self, index):
        X, y = super().__getitem__(index)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]
        data.data_var = self.data_var

        return data

    def __len__(self):
        return super().__len__()


class FMNISTDataset(FashionMNIST):
    def __init__(
        self, 
        root="data/", 
        train=True, 
        **kwargs
    ):
        super().__init__(
            root, 
            train,
            download=True,
            transform=Compose([ToTensor()])
        )

    def __getitem__(self, index):
        X, y = super().__getitem__(index)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]

        return data

    def __len__(self):
        return super().__len__()


class FER2013Dataset(VisionDataset):
    def __init__(
        self, 
        root="data/", 
        train=True,
        **kwargs
    ):
        super().__init__(
            root, 
            transform=Compose([
                ToTensor(),
                Normalize((.5), (.5))
            ]),
            target_transform=None
        )

        base_folder = pathlib.Path(self.root) / "fer2013"
        file_name = "train.csv" if train else "test.csv"
        data_file = base_folder / file_name

        with open(data_file, "r", newline="") as file:
            raw_data = list(csv.DictReader(file))
            self.data = [
                torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48) 
                for row in raw_data
            ]

            self.targets = torch.tensor([
                int(row["emotion"]) if "emotion" in row.keys() else None
                for row in raw_data
            ])

        self.data_var = np.var(torch.stack([d for d in self.data]).numpy() / 255)

    def __getitem__(self, index):
        X_tensor = self.data[index]
        X = Image.fromarray(X_tensor.numpy())

        if self.transform is not None:
            X = self.transform(X)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]
        data.data_var = self.data_var

        return data

    def __len__(self):
        return len(self.data)


class PathMNISTDataset(PathMNIST):
    def __init__(self, split='train', download=True, **kwargs):
        super().__init__(
            split=split,
            transform=Compose([
                ToTensor(),
                Normalize((.5), (.5))
            ]),
            download=download
        )

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)

        X = X.permute(1, 2, 0) * 255
        X = Image.fromarray(X.numpy().astype(np.uint8))

        if self.transform is not None:
            X = self.transform(X)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]

        return data

    def __len__(self):
        return super().__len__()

class CelebADataset(CelebA):
    def __init__(
        self, 
        root="data/", 
        split="train", 
        **kwargs
    ):
        super().__init__(
            root, 
            split, 
            download=False, 
            transform=Compose([
                ToTensor(), 
                Normalize((0.5,), (0.5,))
            ])
        )

    def __getitem__(self, index):
        X, y = super().__getitem__(index)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]

        return data

    def __len__(self):
        return super().__len__()

class EmbeddedDataset(Dataset):
    def __init__(
        self, 
        base_dir,
        ker_width=10,
        graph_connectivity="fully_connected",
        **kwargs
    ):
        super().__init__()

        self.base_dir = base_dir
        self.ds_name = re.split(r"embeddings", base_dir.lower().split('/')[1])[0]
        self.train = kwargs["train"]

        self.sample_size = 5
        self.kernel_width = ker_width
        self.graph_connectivity = graph_connectivity

        # self.data, self.manfld_data = self._get_balanced_label_data()
        self.data = self._get_data()
        if kwargs["smsl_setting"]:
            self.manfld_data = self._get_manifold_data()
        else:
            self.manfld_data = self.data

        self.mean = torch.tensor(self.manfld_data[..., :-2], dtype=torch.float32).mean(dim=0) \
            if kwargs["smsl_setting"] else torch.tensor(self.data[..., :-2], dtype=torch.float32).mean(dim=0)
        self.std = torch.tensor(self.manfld_data[..., :-2], dtype=torch.float32).std(dim=0) \
            if kwargs["smsl_setting"] else torch.tensor(self.data[..., :-2], dtype=torch.float32).std(dim=0)

        @jax.jit
        def compute_edge_weight_jax(data):
            X_i = data[:, None, :]
            X_j = data[None, :, :]

            d_e = ((X_i - X_j) ** 2).sum(-1)

            edge_weights = jnp.ravel(
                jnp.exp(- d_e / (10 ** 2))
            )[:, None]

            return edge_weights

        self.compute_edge_weight_jax = compute_edge_weight_jax


    def _get_data(self):
        return np.load(Path(self.base_dir))

    def _get_manifold_data(self):
        parent_dir = '/'.join(self.base_dir.split('/')[:-1])
        manfld_dir = parent_dir + '/' + self.ds_name
        manfld_dir += "_smsl_train_full_embeddings.npy" if self.train \
            else "_smsl_test_full_embeddings.npy"

        return np.load(Path(manfld_dir))

    def _get_balanced_label_data(self):
        parent_dir = '/'.join(self.base_dir.split('/')[:-1])
        manfld_dir = parent_dir + '/' + self.ds_name
        manfld_dir += "_smsl_train_full_embeddings.npy" if self.train \
            else "_smsl_test_full_embeddings.npy"

        data = np.load(Path(self.base_dir))
        manfld_data = np.load(Path(manfld_dir))

        num_samples = 10000

        if self.train:
            idx_sample = np.random.choice(data.shape[0], num_samples, replace=False)
            blncd_data = data[idx_sample]
            blncd_manfld_data = np.append(manfld_data[idx_sample], manfld_data[data.shape[0]:], 0)
        else:
            idx_sample = np.random.choice(
                list(range(data.shape[0] - num_samples, data.shape[0])), 
                num_samples, 
                replace=False
            )
            blncd_data = data
            blncd_manfld_data = np.append(manfld_data[:data.shape[0]], manfld_data[idx_sample], 0)
    
        return (blncd_data, blncd_manfld_data)


    def _build_graph(self, graph_connectivity, index, sample_size=25):
        graph_data = Data()

        pop_size = self.manfld_data.shape[0]
        idx_prob = [1 / (pop_size - 1)] * pop_size 
        idx_prob[index] = 0
        idx_sample = np.random.choice(pop_size, sample_size-1, replace=False, p=idx_prob)
        idx_sample = np.append(idx_sample, index)
        mask = self.manfld_data[idx_sample, -1]

        graph_data.mask = torch.from_numpy(mask[:, None])
        graph_data.x = torch.tensor(self.manfld_data[idx_sample, :-2], dtype=torch.float32)
        graph_data.y = torch.tensor(
            self.manfld_data[idx_sample][np.argwhere(mask).reshape(-1), -2],
            dtype=torch.int64
        )
        graph_data.num_nodes = graph_data.x.shape[0]

        X_i = graph_data.x[:, None, :]
        X_j = graph_data.x[None, :, :]
        d_e = ((X_i - X_j) ** 2).sum(-1)

        graph_data.edge_weights = torch.flatten(
            torch.exp(- d_e / (self.kernel_width ** 2))
        )[:, None]
        graph_data.edge_weights[graph_data.edge_weights < 0.3] = 0.0

        ## Cosine sim. edge weight
        # cos_sim = torch.flatten(
        #     nn.functional.cosine_similarity(X_i, X_j, dim=-1)
        # )[:, None]

        # graph_data.edge_weights = cos_sim

        graph_data.x = (graph_data.x - self.mean) / self.std \
            if self.ds_name == "cifar10" else graph_data.x

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
            ).to(torch.int64)

        return graph_data

    def _build_graph_jax(self, graph_connectivity, index, sample_size=100):
        graph_data = Data()

        pop_size = self.manfld_data.shape[0]
        idx_prob = [1 / (pop_size - 1)] * pop_size 
        idx_prob[index] = 0
        idx_sample = np.random.choice(pop_size, sample_size-1, replace=False, p=idx_prob)
        idx_sample = np.append(idx_sample, index)
        mask = self.manfld_data[idx_sample, -1]

        graph_data.mask = torch.from_numpy(mask[:, None]) 
        graph_data.x = torch.tensor(self.manfld_data[idx_sample, :-2], dtype=torch.float32)
        graph_data.y = torch.tensor(
            self.manfld_data[idx_sample][np.argwhere(mask).reshape(-1), -2], 
            dtype=torch.int64
        )
        graph_data.num_nodes = graph_data.x.shape[0]

        edge_weights_jax = self.compute_edge_weight_jax(
            jnp.asarray(self.manfld_data[idx_sample, :-2])
        )
        graph_data.edge_weights = torch.from_numpy(np.array(edge_weights_jax))
        graph_data.edge_weights[graph_data.edge_weights < 0.3] = 0.0

        graph_data.x = (graph_data.x - self.mean) / self.std \
            if self.ds_name == 'cifar10' else graph_data.x

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
            ).to(torch.int64)

        return graph_data

    def __getitem__(self, index):
        return self._build_graph(
            self.graph_connectivity, 
            index,
            self.sample_size
        )

    def __len__(self):
        return len(self.data)

class GraphEmbeddedDataset(Dataset):
    def __init__(self, base_dir, **kwargs):
        super().__init__()

        self.data = torch.load(base_dir)
        self.train = kwargs["train"]

    def __len__(self):
        return self.data.x.shape[0]