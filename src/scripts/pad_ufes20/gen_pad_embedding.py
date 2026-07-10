"""
Generate CNNVAE latent embeddings for the PAD-UFES-20 dataset.

Config-driven variant of src/scripts/gen_graph_embedding.py so the original
(hardcoded for PathMNIST) is left untouched. Loads the best CNNVAE checkpoint,
runs the encoder over the deterministic train/test split, and stores the mean
(mu) latent vectors with an appended label column, matching the .npy layout used
for the other datasets: rows = [latent_dims..., label].

Usage:
    python src/scripts/pad_ufes20/gen_pad_embedding.py \
        -d config/cnnvae_pad_dataset_config.yaml \
        -m config/best_cnnvae_pad_model_config.yaml \
        -c src/scripts/checkpoints/<checkpoint>.pt \
        -o data/PadUfes20Embeddings
"""
import sys
import yaml
import argparse

from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm

from torch_geometric.loader import DataLoader

# make src/scripts importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.models import CNNVAEModel
from data_preproc.datasets import build_datasets


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="data/PadUfes20Embeddings")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args(args)


def embed(model, dl, device):
    all_mu = []
    all_y = []
    with torch.no_grad():
        for batch in tqdm(dl):
            batch = batch.to(device)
            _, mu, _, _ = model(batch)
            all_mu.append(mu.detach().cpu())
    mu = torch.cat(all_mu, dim=0).numpy()
    return mu


def main(sys_args):
    args = parse_args(sys_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(Path(args.dataset_config), "r") as f:
        dataset_config = yaml.safe_load(f)
    with open(Path(args.model_config), "r") as f:
        model_config = yaml.safe_load(f)

    train_ds, test_ds = build_datasets(dataset_config)

    # DataLoader must NOT shuffle so embeddings stay aligned with .targets.
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = CNNVAEModel(**CNNVAEModel.pre_init(model_config["args"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    train_mu = embed(model, train_dl, device)
    test_mu = embed(model, test_dl, device)

    train_targets = np.expand_dims(train_ds.targets.numpy(), axis=1)
    test_targets = np.expand_dims(test_ds.targets.numpy(), axis=1)

    train_out = np.append(train_mu, train_targets, axis=-1)
    test_out = np.append(test_mu, test_targets, axis=-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "pad_train_embeddings.npy"
    test_path = out_dir / "pad_test_embeddings.npy"

    with open(train_path, "wb") as f:
        np.save(f, train_out)
    with open(test_path, "wb") as f:
        np.save(f, test_out)

    print(f"train embeddings: {train_out.shape} -> {train_path}")
    print(f"test  embeddings: {test_out.shape} -> {test_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
