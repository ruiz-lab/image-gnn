import sys

import torch

from dataclasses import dataclass

from tqdm import tqdm
from time import time
from pathlib import Path

def sample_latent_points(embeddings, n):
    z_min = embeddings.min(dim=0).values
    z_max = embeddings.max(dim=0).values

    return torch.rand(n, embeddings.size(1)) * (z_max - z_min) + z_min

def latent_to_nodes(z_samples, embeddings, ann_model):
    # Annoy expects float32 inputs
    z_np_all = z_samples.detach().cpu().numpy().astype("float32")

    idxs = [
        ann_model.get_nns_by_vector(z, 1)
        for z in tqdm(z_np_all, desc="Unif. sampling manifold")
    ]

    # Need this in tensor format
    return torch.tensor(idxs, dtype=torch.long).squeeze(-1)

def sample_manifold_nodes(embeddings, ann_model, n_nodes, allowed_mask=None):
    z_samples = sample_latent_points(embeddings, n_nodes) # creates the generated, uniformly sampled embeddings
    idxs = latent_to_nodes(z_samples, embeddings, ann_model) # gets closest actual image embeddings to the generated ones

    return idxs

def uniform_sampling(ds, embeddings, ann_model):
    # Now I'm using the nodes sampled based on the "uniformly sampled"
    # images based on the manifold embedding ranges in each dimension.
    return sample_manifold_nodes(
        embeddings,
        ann_model,
        n_nodes=ds.shape[0],
    )