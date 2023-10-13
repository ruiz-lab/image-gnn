import sys
import yaml
import wandb

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from pathlib import Path

from torch_geometric.loader import DataLoader

from utils.metrics import torch_rmse
from data_preproc.datasets import build_datasets
from models.models import Nerov2

class Trainer():

    def __init__(
        self, 
        dataset_config, 
        training_config, 
        model_config
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_ds, self.test_ds = build_datasets(dataset_config)
        self.training_config = training_config
        self.model_config = model_config

    def run_batch(self, model, dl, mode):        
        pass

    def run_epoch(self, epochs, model, dl, mode):
        pass

    def train(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=8, 
            shuffle=True, 
            num_workers=8
        )

        test_dl = DataLoader(
            dataset=self.test_ds, 
            batch_size=8, 
            shuffle=True, 
            num_workers=4
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            test_losses = []
            train_losses = []

            for i, train_batch in tqdm(enumerate(train_dl)):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)
                train_losses.append(J.detach().cpu().numpy())

                # Backward pass
                J.backward()

                # Optimization step
                optimizer.step()

                optimizer.zero_grad()

            with torch.no_grad():
                for i, test_batch in enumerate(test_dl):
                    batch = test_batch.to(self.device)

                    # Forward pass
                    y_val = model(batch)

                    # Compute val loss
                    J = loss(batch.y, y_val)

                    test_losses.append(J.cpu().numpy())

            test_rsme = np.mean(test_losses)
            train_rsme = np.mean(train_losses)

            wandb.log({
                "test_rmse": test_rsme,
                "train_rmse": train_rsme
            })

    def train_eval(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=8, 
            shuffle=True, 
            num_workers=4
        )

        test_dl = DataLoader(
            dataset=self.test_ds, 
            batch_size=8, 
            shuffle=True, 
            num_workers=4
        )

        num_samples = len(self.train_ds)
        batch_size = train_dl.batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        train_losses = []
        for epoch in tqdm(range(epochs)):
            test_losses = []

            for i, train_batch in enumerate(train_dl):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)

                # Backward pass
                J.backward()

                # Optimization step
                optimizer.step()

                optimizer.zero_grad()

                if i % 20 == 0:
                    print('Train Epoch {}/{} [{:>5}/{} ({:>2.0f}%)] | Loss: {}'.format(
                        epoch+1, epochs, i * batch_size, num_samples, 
                        100*i / len(train_dl), J.detach())
                    )
                    train_losses.append(J.detach())

            with torch.no_grad():
                for i, test_batch in enumerate(test_dl):
                    batch = test_batch.to(self.device)

                    # Forward pass
                    y_val = model(batch)

                    # Compute val loss
                    J = loss(batch.y, y_val)

                    test_losses.append(J.cpu().numpy())

                print(f"Test Loss: {np.mean(test_losses)}")