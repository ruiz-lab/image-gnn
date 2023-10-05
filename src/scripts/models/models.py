import sys

import torch
import torch.nn as nn

from models.encoders import TemporalEncoder
from models.gnns import NeroStemGNN
from models.decoders import MLPBasicBlock

from typing import List, Dict


class GenericModel(nn.Module):
    """
    Generic model.

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        pass

    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass

    def step(self):
        pass

    def eval(self):
        pass

    def forward(self, batch):
        pass