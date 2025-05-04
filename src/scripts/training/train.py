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
from torch_geometric.loader import NeighborLoader

from utils.metrics import torch_rmse, torch_vae_loss, torch_vqvae_loss, torch_ce_loss
from data_preproc.datasets import build_datasets
from models.models import (
    VAEModel, 
    CNNVAEModel, 
    AEModel, 
    VQVAEModel, 
    GNNModel, 
    DGMGNNModel, 
    MLPModel
)


class Trainer():

    def __init__(
        self, 
        dataset_config, 
        training_config, 
        model_config,
        save_model=False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def parse_grow_graph(self, graph_ds, batch_size):
        n0 = 1000
        n_increases = 1
        increase_rate = 10

        grow_graph_loader = []
        if graph_ds.train:
            for i in range(n_increases):
                m = n0 + (i * increase_rate)

                idx = torch.randperm(graph_ds.data.x.shape[0])[:m]
                sampled_graph_data = graph_ds.data.subgraph(idx)

                # sampled_graph_data.edge_weight[sampled_graph_data.edge_weight <= 0.95] = 0.0

                loader = NeighborLoader(
                    sampled_graph_data, 
                    num_neighbors=[25] * 2, 
                    batch_size=batch_size, 
                    input_nodes=sampled_graph_data.mask.nonzero().view(-1), # Take that off
                    shuffle=False
                )

                grow_graph_loader.append(loader)
        else:
            loader = NeighborLoader(
                graph_ds.data, 
                num_neighbors=[25] * 2, 
                # batch_size=batch_size, # Increase batch size to the whole test size
                # batch_size=int(graph_ds.data.mask.sum()),
                batch_size=int(graph_ds.data.mask.sum() // 2),
                input_nodes=graph_ds.data.mask.nonzero().view(-1),
                shuffle=False
            )
            grow_graph_loader.append(loader)

        return grow_graph_loader

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
                # batch = train_batch.to(self.device)

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
            num_workers=32
        )

        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        for _ in range(1):
            run = wandb.init(project="ICML-image-gnn_train-CIFAR10-GCN", reinit=True)

            model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
            model = model(**model.pre_init(self.model_config["args"])).to(self.device)

            loss = getattr(sys.modules[__name__], self.training_config["loss"])
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

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
                    "train_acc": train_acc,
                    "sample_size": self.train_ds.sample_size
                })

                scheduler.step(test_loss)

    def train_eval(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        test_dl = DataLoader(
            dataset=self.test_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        num_samples = len(self.train_ds)
        batch_size = train_dl.batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 
        save_model = self.training_config["save_model"]

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)
        print(type(self.train_ds))
        print(model)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

        train_losses = []
        for epoch in tqdm(range(epochs)):
            test_losses = []
            train_acc = []
            test_acc = []

            for i, train_batch in enumerate(train_dl):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)
                train_acc.append(100 * (sum(batch.y.detach() == torch.max(y_hat, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

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
                    test_acc.append(100 * (sum(batch.y.detach() == torch.max(y_val, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

                    test_losses.append(J.cpu().numpy())

                print(f"Test Loss: {np.mean(test_losses)}")
                print(f"Test Acc: {np.mean(test_acc):.2f}%")
                print(f"Train Acc: {np.mean(train_acc):.2f}%")                

            # scheduler.step(np.mean(test_losses))            

        # print(f"{acc:.2f}%")
        if save_model:
            self.model_checkpoint(model, np.mean(test_losses))

    def train_eval_grow_graph(self):
        train_dl = self.parse_grow_graph(
            self.train_ds,
            self.training_config["batch_size"]
        )
        test_dl = self.parse_grow_graph(
            self.test_ds, 
            self.training_config["batch_size"]
        )

        num_samples = len(self.train_ds)
        batch_size = train_dl[0].batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 
        save_model = self.training_config["save_model"]

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)
        print(type(self.train_ds))
        print(model)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

        n_iter_per_epoch = int(epochs // 1)
        for i in tqdm(range(1)):
            train_losses = []
            # for epoch in tqdm(range(int(epochs // 10))):
            for epoch in tqdm(range(i * n_iter_per_epoch, (i + 1) * n_iter_per_epoch)):
                test_losses = []
                train_acc = []
                test_acc = []

                for j, train_batch in enumerate(train_dl[i]):
                    batch = train_batch.to(self.device)

                    # Forward pass
                    y_hat = model(batch)

                    # Compute loss
                    J = loss(batch.y[:batch.batch_size].reshape(-1).to(torch.long), y_hat)
                    train_acc.append(
                        100 * (
                            sum
                            (
                                batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_hat, axis=1).indices.detach()
                            ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                        ).item()
                    )

                    # Backward pass
                    J.backward()

                    # Optimization step
                    optimizer.step()

                    optimizer.zero_grad()

                    if j % 2 == 0:
                        print('Train Epoch {}/{} [{:>5}/{} ({:>2.0f}%)] | Loss: {}'.format(
                            epoch+1, epochs, j * batch_size, num_samples, 
                            100*j / len(train_dl), J.detach())
                        )
                        train_losses.append(J.detach())

                with torch.no_grad():
                    for j, test_batch in enumerate(test_dl[0]):
                        batch = test_batch.to(self.device)

                        # Forward pass
                        y_val = model(batch)

                        # Compute val loss
                        J = loss(batch.y[:batch.batch_size].reshape(-1).to(torch.long), y_val)
                        test_acc.append(
                            100 * (
                                sum
                                (
                                    batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_val, axis=1).indices.detach()
                                ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                            ).item()
                        )

                        test_losses.append(J.cpu().numpy())

                    print(f"Test Loss: {np.mean(test_losses)}")
                    print(f"Test Acc: {np.mean(test_acc):.2f}%")
                    print(f"Train Acc: {np.mean(train_acc):.2f}%")                

                # scheduler.step(np.mean(test_losses))            

        # print(f"{acc:.2f}%")
        if save_model:
            self.model_checkpoint(model, np.mean(test_losses))