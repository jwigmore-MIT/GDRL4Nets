from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Iterable, Sized, Tuple

class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        raise NotImplementedError("abstract methodd called")

class ExpUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 max_value: float = 1.0):
        super().__init__(in_features, out_features)
        torch.nn.init.uniform_(self.weight,a=-3.0, b=0) # weights seem to be too small using this one
        #torch.nn.init.uniform_(self.weight,a=-1, b=2.0)
        # truncated_normal_(self.bias, std=0.1)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.size = in_features
        self.max_value = max_value

    def forward(self, x):
        out = (x) @ torch.exp(self.weight) + self.bias
        return (1-0.01) * torch.clip(out, 0, self.max_value) + 0.01 * out


class FCLayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        # truncated_normal_(self.weight, mean=0.0, std=1)
        torch.nn.init.uniform_(self.weight, a=-3.0, b=0)
        #truncated_normal_(self.bias, std=0.1)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        FC = x @ torch.exp(self.weight) + self.bias
        return FC


class SwitchTypeNetwork(torch.nn.Module):
    """
    Same as ScalableMonotonicNeuralNetwork but with no confluence or Relu units
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 hidden_sizes: Tuple = (64, 64),
                 relu_max: float = 1.0,
                 exp_unit: ActivationLayer = ExpUnit,
                 fc_layer: ActivationLayer = FCLayer
                 ):
        super(SwitchTypeNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.relu_max = relu_max

        self.hidden_layers = torch.nn.ModuleList([
            exp_unit(self.input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i], max_value=relu_max) for i in range(len(hidden_sizes))
        ])

        self.fclayer = fc_layer(hidden_sizes[len(hidden_sizes)-1], output_size)

        self.network = torch.nn.Sequential(*self.hidden_layers, self.fclayer)


    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.fclayer(x)
        return out

