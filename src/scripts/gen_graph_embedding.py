import sys
import yaml
import argparse

import torch

import numpy as np

from tqdm import tqdm

from pathlib import Path

from models.models import VAEModel, CNNVAEModel
from data_preproc.datasets import build_datasets

from torch_geometric.loader import DataLoader


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

def main(sys_args, model):
    args = parse_args(sys_args)

    with open(Path(args.dataset_config), 'r') as f:
        dataset_config = yaml.safe_load(f)

    train_ds, test_ds = build_datasets(dataset_config)

    # train_ds.data = train_ds.data[:1000]
    # train_ds.targets = train_ds.targets[:1000]

    # test_ds.data = test_ds.data[:1000]
    # test_ds.targets = test_ds.targets[:1000]

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=len(train_ds),
        shuffle=False,
        num_workers=4
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=len(train_ds),
        shuffle=False,
        num_workers=4
    )

    with torch.no_grad():
        for _, train_batch in tqdm(enumerate(train_dl)):
            output, mu, logvar, z = model(train_batch)
        mu = mu.flatten(start_dim=1)

        for _, test_batch in tqdm(enumerate(test_dl)):
            test_output, test_mu, test_logvar, test_z = model(test_batch)
        test_mu = test_mu.flatten(start_dim=1)

    ds_path = "data/CIFAR10Embeddings"
    train_ds_path = ds_path + "/cifar10_train_embeddings.npy"
    test_ds_path = ds_path + "/cifar10_test_embeddings.npy"

    np_z = mu.detach().numpy()
    np_targets = np.expand_dims(train_ds.targets.numpy(), axis=1)
    np_train = np.append(np_z, np_targets, axis=-1)
    with open(Path(train_ds_path), "wb") as f:
        np.save(f, np_train)
        f.close()

    np_test_z = test_mu.detach().numpy()    
    np_test_targets = np.expand_dims(test_ds.targets.numpy(), axis=1)
    np_test = np.append(np_test_z, np_test_targets, axis=-1)
    with open(Path(test_ds_path), "wb") as f:
        np.save(f, np_test)
        f.close()


if __name__ == "__main__":
    path = "src/scripts/checkpoints/"
    model_id = "080224-180726-58.pt"
    model_path = path + model_id

    model = CNNVAEModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    main(sys.argv[3:5], model)