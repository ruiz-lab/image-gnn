import sys

import torch

from dataclasses import dataclass

from tqdm import tqdm
from time import time
from pathlib import Path

# # Pass all of the images through the CNN encoder to
# # get every single embedding.
# @torch.no_grad()
# def compute_embeddings(encoder, data, device):
#     x = data.x.to(device) # [number of samples, 3, 32, 32]
#     z = encoder(x) # [number of samples, 128]
#     return z.cpu()

# embeddings = compute_embeddings(encoder, train_data, device)

# Then create a "bounds" (ranges of values)
# for all 128 dimensions.
# z_min = embeddings.min(dim=0).values
# z_max = embeddings.max(dim=0).values

def sample_latent_points(embeddings, n):
    # torch.rand() picks a random value between 0 and 1 for each of the 128 dimensions,
    # then we multiply by each range and add the minimum to get a uniformly random
    # value that is in the range of that embedding's possible values.
    z_min = embeddings.min(dim=0).values
    z_max = embeddings.max(dim=0).values

    return torch.rand(n, embeddings.size(1)) * (z_max - z_min) + z_min

def latent_to_nodes(z_samples, embeddings, aknn_model):
    # torch.cdist() gets the distances between each generated "uniformly sampled
    # embedding" and the actual image's embedding.
    # dists = torch.cdist(z_samples, embeddings)

    #return [aknn_model.get_nns_by_vector(z, 1, include_distances=False)\
    #         for z in z_samples]

    # Annoy expects float32 inputs
    z_np_all = z_samples.detach().cpu().numpy().astype("float32")
    
    idxs = [
        aknn_model.get_nns_by_vector(z, 1, include_distances=False)[0]
        for z in z_np_all
    ]

    # Need this in tensor format
    return torch.tensor(idxs, dtype=torch.long)

    # Then, we get the closest actual image embedding to each generated embedding.
    # return dists.argmin(dim=1)

def sample_manifold_nodes(embeddings, aknn_model, n_nodes, allowed_mask=None):
    z_samples = sample_latent_points(embeddings, n_nodes) # creates the generated, uniformly sampled embeddings
    # idx = latent_to_nodes(z_samples, embeddings) # gets closest actual image embeddings to the generated ones
    idxs = latent_to_nodes(z_samples, embeddings, aknn_model) # gets closest actual image embeddings to the generated ones

    # I added this originally because I wasn't sure if I should apply a mask to indicate which embeddings are
    # for training, and which aren't. However, since for now, I only applied it to training images, I commented it
    # out (as this is only used in the training_loader).
    # if allowed_mask is not None:
    #     idx = idx[allowed_mask[idx]]

    # Should I only keep the uniquely sampled images? Or if an image is sampled twice, should I let that be represented?
    # return idx.unique()
    return idxs

def uniform_sampling(ds, embeddings, aknn_model):
    # Now I'm using the nodes sampled based on the "uniformly sampled"
    # images based on the manifold embedding ranges in each dimension.
    return sample_manifold_nodes(
        embeddings,
        aknn_model,
        n_nodes=ds.shape[0],
        # allowed_mask=ds.mask
    )

# train_loader = NeighborLoader(
#     data=train_data,
#     input_nodes=manifold_nodes, # Using these "uniformly sampled" training nodes
#     num_neighbors=num_neighbors,
#     batch_size=batch_size,
#     shuffle=True
# )
