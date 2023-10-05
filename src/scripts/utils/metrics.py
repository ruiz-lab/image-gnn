import torch

import numpy as np


def rmse(y, y_hat):
    N = y.shape[0]
    rmse = np.sqrt(np.sum((y - y_hat) ** 2) / N)

    return np.mean(rmse)

def torch_rmse(y, y_hat):
    N = y.shape[0]
    rmse = torch.sqrt(torch.sum((y - y_hat) ** 2) / N)

    return rmse.mean()

def ioa(y, y_hat):
    ioa = 1 - (np.sum((y - y_hat) ** 2)) / (
        np.sum((np.abs(y_hat - np.mean(y)) + np.abs(y - np.mean(y))) ** 2)
    )

    return ioa

def torch_ioa(y, y_hat):
    ioa = 1 - (torch.sum((y - y_hat) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(y_hat - torch.mean(y, dim=0)) + torch.abs(y - torch.mean(y, dim=0)))
            ** 2,
            dim=0,
        )
    )

    return ioa.mean()