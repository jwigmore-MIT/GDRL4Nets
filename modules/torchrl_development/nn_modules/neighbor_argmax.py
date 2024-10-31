from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


class NeighborArgmax(MessagePassing):
    r"""
    Computes a component-wise argmax over the nodes immediate neighborhood

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.


    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{s})` or
          :math:`(|\mathcal{V_t}|, F_{s})` if bipartite
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],

            **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = self.in_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr="max", **kwargs)  # For the aggr parameter, we are using the max aggregation scheme

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            size: Size = None,
    ) -> Tensor:
        if self.training:
            return x
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        out = (x_r > out).float()

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class NeighborSoftmax(NeighborArgmax):
    r"""
    Computes a component-wise argmax over the nodes immediate neighborhood

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.


    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{s})` or
          :math:`(|\mathcal{V_t}|, F_{s})` if bipartite
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            temp: float = 10.0,
            **kwargs,
    ):


        super().__init__(in_channels, **kwargs)  # For the aggr parameter, we are using the max aggregation scheme
        self.temp = temp


    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.temp*self.propagate(edge_index, x=x, size=size)

        x_r = self.temp*x[1]
        out = x_r.exp()/(x_r.exp() + out.exp())

        return out

