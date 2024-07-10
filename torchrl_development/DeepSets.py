import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Iterable, Sized, Tuple


class InvariantModel(nn.Module):

    def __init__(self, phi: nn.Module, rho: nn.Module):
        super(InvariantModel, self).__init__()

        self.phi = phi
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute representation for each data point
        x = self.phi(x)
        # aggregate the representation
        x = torch.sum(x, dim = 0, keepdim = True)

        # compute the output
        out = self.rho(x)
        return out


def create_deep_set_nn(phi_in_dim, latent_dim, out_dim, width = 16):
    """
    Create a deep set neural network
    """
    phi = nn.Sequential(
        nn.Linear(phi_in_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, latent_dim)
    )

    rho = nn.Sequential(
        nn.Linear(latent_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, out_dim)
    )

    return InvariantModel(phi, rho)


