from SMNN import PureMonotonicNeuralNetwork as PMN
import torch
import numpy as np

input = torch.Tensor([[1, 2], [2, 1], [2,2], [3, 4], [4, 3], [4,4]])
input_size = input.shape[1]
output_size = 1
hidden_sizes = [64, 64]


model = PMN(input_size = input_size,
            output_size = output_size,
            hidden_sizes = hidden_sizes,
            )

output = model(input)



