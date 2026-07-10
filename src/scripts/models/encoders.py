import torch
import torch.nn as nn

from torch.nn import functional as F

from torch_geometric.nn import MessagePassing

from typing import List, Dict, Any


class EncoderVAE(nn.Module):

    def __init__(
        self, 
        in_features, 
        hidden_size,
        latent_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, hidden_size),
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)]
            ]
        )
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.var_layer = nn.Linear(hidden_size, latent_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        mean = self.activation_fn(self.mean_layer(out))
        logvar = self.activation_fn(self.var_layer(out))

        return mean, logvar

class EncoderCNNVAE(nn.Module):

    def __init__(
        self,
        in_channels,
        latent_size,
        blocks=[1, 1, 1],
        activation_fn=nn.LeakyReLU(),
        img_size=28,
        **kwargs
    ):
        super().__init__()

        if img_size == 28:
            # Original 28x28 path (MNIST/FMNIST/PathMNIST/CIFAR10/FER2013).
            # Spatial: 28 -> 14 -> 7 -> 4 ; bottleneck 256 x 4 x 4 = 4096.
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                    *[ConvBasicBlock(64, 64) for _ in range(blocks[0])],
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=1),
                    *[ConvBasicBlock(128, 128) for _ in range(blocks[1])],
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(128, 256, kernel_size=1),
                    *[ConvBasicBlock(256, 256) for _ in range(blocks[2])],

                    # nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                    # nn.GELU(),
                    # nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    # nn.GELU(),
                    # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    # nn.GELU(),
                    # nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    # nn.GELU(),
                    # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    # nn.GELU(),

                    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # nn.Conv2d(256, 512, kernel_size=1),
                    # *[ConvBasicBlock(512, 512) for _ in range(blocks[2])],

                    # # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # # nn.Conv2d(512, 512, kernel_size=1),
                    # # *[ConvBasicBlock(512, 512) for _ in range(blocks[2])],


                    nn.Flatten(),
                    # nn.Linear(64 * 16, latent_size),
                    # nn.GELU()
                ]
            )
            self.flat_dim = 256 * 16
        elif img_size == 224:
            # 224x224 path (PAD-UFES-20). Five 2x downsamplings:
            # 224 -> 112 -> 56 -> 28 -> 14 -> 7 ; bottleneck 256 x 7 x 7 = 12544.
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                    *[ConvBasicBlock(64, 64) for _ in range(blocks[0])],
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=1),
                    *[ConvBasicBlock(128, 128) for _ in range(blocks[1])],
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(128, 256, kernel_size=1),
                    *[ConvBasicBlock(256, 256) for _ in range(blocks[2])],
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Flatten(),
                ]
            )
            self.flat_dim = 256 * 49
        else:
            raise ValueError(f"Unsupported img_size for EncoderCNNVAE: {img_size}")

        self.mean_layer = nn.Linear(self.flat_dim, latent_size)
        self.var_layer = nn.Linear(self.flat_dim, latent_size)
        # self.mean_layer = nn.Linear(latent_size, latent_size)
        # self.var_layer = nn.Linear(latent_size, latent_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        out  = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))
            # out = layer(out)

        # mean = self.activation_fn(
        #     self.mean_layer(
        #         torch.permute(out, (0, 2, 3, 1))
        #     )
        # )
        # logvar = self.activation_fn(
        #     self.var_layer(
        #         torch.permute(out, (0, 2, 3, 1))
        #     )
        # )

        # return (
        #     torch.permute(mean, (0, 3, 1, 2)), 
        #     torch.permute(logvar, (0, 3, 1, 2))
        # )

        mean = self.activation_fn(
            self.mean_layer(out)
        )
        logvar = self.activation_fn(
            self.var_layer(out)
        )

        return mean, logvar


class EncoderVQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        return torch.relu(h)

class ConvBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()                
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = F.leaky_relu(out + x)

        return out

class EncoderMLP(nn.Module):

    def __init__(
        self, 
        in_features, 
        hidden_size,
        layers=1, 
        activation_fn=nn.ReLU(),
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, hidden_size),
                *[nn.Linear(hidden_size, hidden_size) for _ in range(layers-2)]
            ]
        )
        self.activation_fn = activation_fn

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.activation_fn(layer(out))

        return out

class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.root_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def update(self, aggr_out, x):
        return self.lin(aggr_out) + self.root_lin(x)
