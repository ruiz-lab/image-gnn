"""
Parse PAD-UFES-20 CNNVAE embeddings into the SmSL-tagged layout.

Faithful, path-parametrized variant of src/scripts/parse_graph_embedding.py.
Given the raw train/test embedding files (rows = [latent..., label]) it appends a
train/test tag column (1 = this-split node / query, 0 = other-split node) and
writes the four files the graph builder / EmbeddedDataset expect
(rows = [latent..., label, mask]):

    <ds>_smsl_train_embeddings.npy        -> train rows, tag 1
    <ds>_smsl_test_embeddings.npy         -> test  rows, tag 1
    <ds>_smsl_train_full_embeddings.npy   -> all rows, train tagged 1
    <ds>_smsl_test_full_embeddings.npy    -> all rows, test  tagged 1

Usage:
    python src/scripts/pad_ufes20/parse_pad_embedding.py \
        --in_dir data/PadUfes20Embeddings \
        --out_dir data/PadUfes20Embeddings/smsl_embeddings \
        --prefix pad
"""
import argparse

from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/PadUfes20Embeddings")
    parser.add_argument("--out_dir", type=str, default="data/PadUfes20Embeddings/smsl_embeddings")
    parser.add_argument("--prefix", type=str, default="pad")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = args.prefix

    train_data = np.load(in_dir / f"{p}_train_embeddings.npy")
    test_data = np.load(in_dir / f"{p}_test_embeddings.npy")

    train_full_tag_train_data = np.append(train_data, np.ones((train_data.shape[0], 1), dtype=np.int32), 1)
    train_full_tag_test_data = np.append(test_data, np.zeros((test_data.shape[0], 1), dtype=np.int32), 1)

    test_full_tag_test_data = np.append(test_data, np.ones((test_data.shape[0], 1), dtype=np.int32), 1)
    test_full_tag_train_data = np.append(train_data, np.zeros((train_data.shape[0], 1), dtype=np.int32), 1)

    train_full_tag_full_data = np.append(train_full_tag_train_data, train_full_tag_test_data, 0)
    test_full_tag_full_data = np.append(test_full_tag_test_data, test_full_tag_train_data, 0)

    outputs = {
        f"{p}_smsl_train_embeddings.npy": train_full_tag_train_data,
        f"{p}_smsl_test_embeddings.npy": test_full_tag_test_data,
        f"{p}_smsl_train_full_embeddings.npy": train_full_tag_full_data,
        f"{p}_smsl_test_full_embeddings.npy": test_full_tag_full_data,
    }
    for name, arr in outputs.items():
        with open(out_dir / name, "wb") as f:
            np.save(f, arr)
        print(f"{arr.shape} -> {out_dir / name}")


if __name__ == "__main__":
    main()
