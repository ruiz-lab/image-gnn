import sys
import re
import yaml
import argparse

import torch

import numpy as np

from tqdm import tqdm

from pathlib import Path

from models.models import PCAModel
from data_preproc.datasets import build_datasets

from torch_geometric.loader import DataLoader

from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset_config", 
        type=str, 
        help="Training configuration file", 
        required=True
    )

    return parser.parse_args(args)

def extract_dataset_name(class_name):
    match = re.match(r'(.*)(Dataset|Binary)$', class_name)
    if match:
        return match.group(1)
    else:
        return None

def main(sys_args, model):
    args = parse_args(sys_args)

    with open(Path(args.dataset_config), 'r') as f:
        dataset_config = yaml.safe_load(f)

    train_ds, test_ds = build_datasets(dataset_config)
    ds_name = extract_dataset_name(train_ds.__class__.__name__)

    if ds_name in ['CIFAR10', 'PathMNIST']:
        X_train = torch.from_numpy(
            train_ds.data
        ).flatten(start_dim=1) / 255.0
        X_test = torch.from_numpy(
            test_ds.data
        ).flatten(start_dim=1) / 255.0
    elif 'CelebA' in ds_name:
        X_train = []
        for idx in tqdm(range(train_ds.targets.shape[0])):
            X_train.append(train_ds[idx][0])

        X_test = []
        for idx in tqdm(range(test_ds.targets.shape[0])):
            X_test.append(test_ds[idx][0])

        X_train = torch.stack(
            X_train, 
            dim=0
        ).flatten(start_dim=1)
        X_test = torch.stack(
            X_test, 
            dim=0
        ).flatten(start_dim=1)
    else:
        X_train = train_ds.data.flatten(start_dim=1) / 255.0
        X_test = test_ds.data.flatten(start_dim=1) / 255.0

    # X_train = train_ds.data.flatten(start_dim=1) / 255.0
    # X_test = test_ds.data.flatten(start_dim=1) / 255.0

    # train_ds.data = train_ds.data[:1000]
    # train_ds.targets = train_ds.targets[:1000]

    # test_ds.data = test_ds.data[:1000]
    # test_ds.targets = test_ds.targets[:1000]

    X_train_transformed = model.fit_transform(X_train.numpy())
    X_test_transformed = model.transform(X_test.numpy())

    ds_path = "data/PathMNISTEmbeddings/pca_embeddings"
    train_ds_path = ds_path + "/pathmnist_train_embeddings.npy"
    test_ds_path = ds_path + "/pathmnist_test_embeddings.npy"

    np_z = X_train_transformed
    np_targets = np.expand_dims(train_ds.targets.numpy(), axis=1)
    np_train = np.append(np_z, np_targets, axis=-1)
    with open(Path(train_ds_path), "wb") as f:
        np.save(f, np_train)
        f.close()

    np_test_z = X_test_transformed
    np_test_targets = np.expand_dims(test_ds.targets.numpy(), axis=1)
    np_test = np.append(np_test_z, np_test_targets, axis=-1)
    with open(Path(test_ds_path), "wb") as f:
        np.save(f, np_test)
        f.close()


if __name__ == "__main__":
    model = PCAModel(n_components=8)

    main(sys.argv[3:5], model)