from torch import nn
import monotonicnetworks as lmn
import torch

lip_nn = nn.Sequential(
    lmn.LipschitzLinear(2, 32, kind="one-inf"),
    lmn.GroupSort(2),
    lmn.LipschitzLinear(32, 1, kind="inf"),
)
monotonic_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[1,1]) # first input increasing, no monotonicity constraints on second input

input = torch.Tensor([[1, 2], [2, 1], [2,2], [3, 4], [4, 3], [4,4]])

output = monotonic_nn(input)