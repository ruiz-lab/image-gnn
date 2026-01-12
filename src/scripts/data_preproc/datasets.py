import sys
import re
import csv
import pickle
import pathlib

import numpy as np

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

from PIL import Image

from medmnist import PathMNIST, OCTMNIST

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA, VisionDataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize

from torch_geometric.data import Data

from time import time
from pathlib import Path
from itertools import product
from typing import List, Dict


def smart_load(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return torch.load(f)
    elif path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def build_datasets(ds_config):
    datasets = []
    for ds_type, configs in ds_config.items():
        configs = configs.copy()
        ds_class = getattr(sys.modules[__name__], configs["dataset_class"])

        # if "base_dir" in configs and Path(configs["base_dir"]).suffix in [".pkl", ".npy"]:
        #     configs["data"] = smart_load(configs["base_dir"])
        #     del configs["base_dir"]

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
            self.data = torch.stack(
                [
                    torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48) 
                    for row in raw_data
                ], 
                dim=0
            )

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
    def __init__(self, train=True, download=True, **kwargs):
        split = 'train' if train else 'test'

        super().__init__(
            split=split,
            transform=Compose([
                ToTensor(),
                Normalize((.5), (.5))
            ]),
            download=download
        )

        self.targets = torch.from_numpy(self.labels).squeeze(dim=1)
        self.data = self.imgs

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)

        data = Data()
        data.x = X[None, ...]
        data.y = X[None, ...]

        return data

    def __len__(self):
        return self.data.shape[0]

class OCTMNISTDataset(OCTMNIST):
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

class CelebASmileBinary(CelebA):
    def __init__(
        self, 
        base_dir, 
        train=True, 
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs
    ):
        if transform is None:
            transform = Compose([Resize((64, 64)), ToTensor()])

        split = 'train' if train else 'test'

        super().__init__(root=base_dir, split=split, target_type='attr', transform=transform,
                         target_transform=target_transform, download=download)

        # Convert smile attribute to binary label (0 = not smiling, 1 = smiling)
        self.smile_idx = self.attr_names.index('Smiling')
        self.targets = (self.attr[:, self.smile_idx] == 1).long()

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.targets[index]


class CelebAGenderBinary(CelebA):
    def __init__(
        self, 
        base_dir, 
        train=True, 
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs
    ):
        if transform is None:
            transform = Compose([Resize((64, 64)), ToTensor()])

        split = 'train' if train else 'test'

        super().__init__(root=base_dir, split=split, target_type='attr', transform=transform,
                         target_transform=target_transform, download=download)

        # Convert gender attribute to binary label (0 = woman, 1 = man)
        self.gender_idx = self.attr_names.index('Male')
        self.targets = (self.attr[:, self.gender_idx] == 1).long()

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.targets[index]

class EmbeddedDataset(Dataset):
    def __init__(
        self, 
        data=None,
        base_dir=None,
        ker_width=10,
        graph_connectivity="fully_connected",
        **kwargs
    ):
        super().__init__()

        self.base_dir = base_dir
        self.train = kwargs["train"]
        self.sample_size = 5
        self.kernel_width = ker_width
        self.graph_connectivity = graph_connectivity

        if data is not None:
            self.data = data
            self.ds_name = "embedded"
        else:
            self.ds_name = re.split(r"embeddings", base_dir.lower().split('/')[1])[0]
            self.data = self._load_file(self.base_dir)

        self.manfld_data = self._get_manifold_data() if kwargs.get("smsl_setting", False) else self.data

        self.mean = torch.tensor(self.manfld_data[..., :-2], dtype=torch.float32).mean(dim=0)
        self.std = torch.tensor(self.manfld_data[..., :-2], dtype=torch.float32).std(dim=0)

        @jax.jit
        def compute_edge_weight_jax(data):
            X_i = data[:, None, :]
            X_j = data[None, :, :]
            d_e = ((X_i - X_j) ** 2).sum(-1)
            return jnp.ravel(jnp.exp(-d_e / (10 ** 2)))[:, None]

        self.compute_edge_weight_jax = compute_edge_weight_jax

    def _load_file(self, path):
        return smart_load(path)

    def _get_manifold_data(self):
        parent_dir = '/'.join(self.base_dir.split('/')[:-1])
        manfld_dir = parent_dir + '/' + self.ds_name
        manfld_dir += "_smsl_train_full_embeddings.pkl" if self.train else "_smsl_test_full_embeddings.pkl"
        return self._load_file(manfld_dir)

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

    def _build_graph(self, graph_connectivity, index, sample_size=5):
        graph_data = Data()
        pop_size = self.manfld_data.shape[0]

        idx_prob = [1 / (pop_size - 1)] * pop_size 
        idx_prob[index] = 0
        idx_sample = np.random.choice(pop_size, sample_size - 1, replace=False, p=idx_prob)
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

        graph_data.x = (graph_data.x - self.mean) / self.std \
            if self.ds_name == "cifar10" else graph_data.x

        if graph_connectivity == "fully_connected":
            graph_data.edge_index = torch.from_numpy(
                np.array(
                    list(product(range(graph_data.num_nodes), range(graph_data.num_nodes)))
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