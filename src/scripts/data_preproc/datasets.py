import numpy as np

import torch
import torch.nn as nn

from torch.nn import functional as F

from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose


from torch_geometric.data import Data, DataLoader

from tqdm import tqdm
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field

from typing import List, Dict


def build_datasets(ds_config):
    datasets  = []
    for ds_type, configs in ds_config.items():
        datasets.append(MNISTDataset(**configs))

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