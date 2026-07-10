import torch

import numpy as np

import torch.nn.functional as F

from typing import Tuple


def rmse(y, y_hat):
    N = y.shape[0]
    rmse = np.sqrt(np.sum((y - y_hat) ** 2) / N)

    return rmse

def torch_rmse(y, y_hat):
    N = y.shape[0]
    rmse = torch.sqrt(torch.sum((y - y_hat) ** 2) / N)

    return rmse

def torch_kl_div(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def torch_vae_loss(x, x_hat):
    output, mu, logvar, _ = x_hat

    # reconstruction_loss = F.mse_loss(output, x, reduction="sum")
    reconstruction_loss = F.mse_loss(output, x, reduction="none")
    reconstruction_loss = reconstruction_loss.sum([1, 2, 3]).mean([0])
    kl_loss = torch_kl_div(mu, logvar)

    beta = 0.0005

    return reconstruction_loss +  beta * kl_loss

def torch_vqvae_loss(x, x_hat):
    (output, _, dict_loss, commit_loss, use_ema, data_var) = x_hat

    reconstruction_loss = F.mse_loss(output, x) / data_var[0]

    beta = 0.25

    total_loss = reconstruction_loss + beta * commit_loss if use_ema \
        else reconstruction_loss + beta * commit_loss + dict_loss

    return total_loss

def torch_ce_loss(y, y_hat):
    return F.cross_entropy(y_hat, y)