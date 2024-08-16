import sys
import yaml
import wandb

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime

from pathlib import Path

from torch_geometric.loader import DataLoader

from utils.metrics import torch_rmse, torch_vae_loss, torch_ce_loss
from data_preproc.datasets import build_datasets
from models.models import VAEModel, CNNVAEModel, GNNModel, MLPModel


class Trainer():

    def __init__(
        self, 
        dataset_config, 
        training_config, 
        model_config,
        save_model=False
    ):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.train_ds, self.test_ds = build_datasets(dataset_config)
        self.training_config = training_config
        self.model_config = model_config

    def run_batch(self, model, dl, mode):        
        pass

    def run_epoch(self, epochs, model, dl, mode):
        pass

    def model_checkpoint(self, model, mean_loss, path="src/scripts/checkpoints/"):
        now = datetime.now().strftime("%m%d%y-%H%M%S-")
        path = path + now + format(mean_loss, ".0f") + ".pt"

        torch.save(model.state_dict(), path)

    def train(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"], 
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

            test_loss = np.mean(test_losses)
            train_loss = np.mean(train_losses)

            wandb.log({
                "test_loss": test_loss,
                "train_loss": train_loss
            })

    def train_runs(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        for _ in range(50):
            run = wandb.init(project="GNN-image-gnn_train", reinit=True)

            model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
            model = model(**model.pre_init(self.model_config["args"])).to(self.device)

            loss = getattr(sys.modules[__name__], self.training_config["loss"])
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

            for epoch in tqdm(range(epochs)):
                test_losses = []
                train_losses = []
                train_batch_acc = []
                test_batch_acc = []

                for i, train_batch in tqdm(enumerate(train_dl)):
                    batch = train_batch.to(self.device)

                    # Forward pass
                    y_hat = model(batch)

                    # Compute loss
                    J = loss(batch.y, y_hat)
                    train_losses.append(J.detach().cpu().numpy())
                    train_batch_acc.append(100 * (sum(batch.y.detach() == torch.max(y_hat, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

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
                        test_batch_acc.append(100 * (sum(batch.y.detach() == torch.max(y_val, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

                test_loss = np.mean(test_losses)
                train_loss = np.mean(train_losses)
                test_acc = np.mean(test_batch_acc)
                train_acc = np.mean(train_batch_acc)

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "train_acc": train_acc
                })


    def train_eval(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        test_dl = DataLoader(
            dataset=self.test_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        num_samples = len(self.train_ds)
        batch_size = train_dl.batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 
        save_model = self.training_config["save_model"]

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

        train_losses = []
        for epoch in tqdm(range(epochs)):
            test_losses = []

            for i, train_batch in enumerate(train_dl):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)
                # acc = 100 * (sum(batch.y.detach() == torch.max(y_hat, axis=1).indices.detach()) / batch.y.detach().shape[0]).item()

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
                    # acc = 100 * (sum(batch.y.detach() == torch.max(y_val, axis=1).indices.detach()) / batch.y.detach().shape[0]).item()

                    test_losses.append(J.cpu().numpy())

                print(f"Test Loss: {np.mean(test_losses)}")
                # print(f"{acc:.2f}%")

            scheduler.step(np.mean(test_losses))            

        # print(f"{acc:.2f}%")
        if save_model:
            self.model_checkpoint(model, np.mean(test_losses))