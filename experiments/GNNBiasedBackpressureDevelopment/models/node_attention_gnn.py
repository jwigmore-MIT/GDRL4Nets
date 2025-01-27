from typing import List, Optional, Tuple, Union
from torch_geometric.nn.conv import GENConv
from torch import Tensor
from torch import cat
from torch.nn import (
    BatchNorm1d,
    Dropout,
    InstanceNorm1d,
    LayerNorm,
    ReLU,
    Sigmoid,
    Sequential,
    Linear,
    Module,
    ModuleList,
)
import torch

from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.functional import relu

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.data import Data, Batch

def initialize_weights(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         nn.init.zeros_(m.bias)
    # elif isinstance(m, nn.Conv2d):
        if isinstance(m, Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
class SDPA_layer(Module):
    r"""
    Scaled Dot Product Attention Layer for GNN to be applied to any node embedding tensor X


    """

    def __init__(self, input_dim, output_dim):
        super(SDPA_layer, self).__init__()
        if input_dim != output_dim:
            self.proj = Linear(input_dim, output_dim)
        else:
            self.proj = None

        self.qkv_mlp = Linear(output_dim, output_dim * 3, bias=False)
        self.mlp = Linear(output_dim, output_dim)
        self.apply(initialize_weights)


    def forward(self, X):
        if self.proj is not None:
            X = self.proj(X)
        QKV = self.qkv_mlp(X)
        Q, K, V = QKV.chunk(3, dim=-1)
        H = sdpa(Q, K, V)
        out = self.mlp(H+X)
        return  out +X

"""
The conv layer should go:
x = lin_proj(x) if needed (i.e. input_dim != output_dim)
if not first layer:
    edge_attr = lin_edge(edge_attr) if needed (i.e. edge_dim != output_dim)
else:
    out = SPDA_layer(x)

return self.output_func(out)
"""


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0., output_func = ReLU):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(output_func())
                m.append(Dropout(dropout))

        super().__init__(*m)

class EdgeDecoder(MessagePassing):
    """
    Need to take X and edge_index and output edge_attr for each edge
    which is simply an MLP applied to the concatentation of the node features for each
    edge in edge_index
    """
    def __init__(self, input_dim, output_dim = 1, activation = ReLU):
        super(EdgeDecoder, self).__init__()
        self.mlp = MLP([2*input_dim, output_dim], norm=None, bias=True, dropout=0.0, output_func = activation)

    def forward(self, X, edge_index):
        edge_attr = cat([X[edge_index[0]], X[edge_index[1]]], dim = -1)
        edge_attr = self.mlp(edge_attr)
        return edge_attr

class NodeAttentionConv(MessagePassing):
    r"""The GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.

    :class:`GENConv` supports both :math:`\textrm{softmax}` (see
    :class:`~torch_geometric.nn.aggr.SoftmaxAggregation`) and
    :math:`\textrm{powermean}` (see
    :class:`~torch_geometric.nn.aggr.PowerMeanAggregation`) aggregation.
    Its message construction is given by:

    .. math::
        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_i +
        \mathrm{AGG} \left( \left\{
        \mathrm{ReLU} \left( \mathbf{x}_j + \mathbf{e_{ji}} \right) +\epsilon
        : j \in \mathcal{N}(i) \right\} \right)
        \right)

    .. note::

        For an example of using :obj:`GENConv`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            (:obj:`"softmax"`, :obj:`"powermean"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)
        t (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`1.0`)
        learn_t (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`t` for softmax aggregation dynamically.
            (default: :obj:`False`)
        p (float, optional): Initial power for power mean aggregation.
            (default: :obj:`1.0`)
        learn_p (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`p` for power mean aggregation dynamically.
            (default: :obj:`False`)
        msg_norm (bool, optional): If set to :obj:`True`, will use message
            normalization. (default: :obj:`False`)
        learn_msg_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor of message normalization. (default: :obj:`False`)
        norm (str, optional): Norm layer of MLP layers (:obj:`"batch"`,
            :obj:`"layer"`, :obj:`"instance"`) (default: :obj:`batch`)
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        expansion (int, optional): The expansion factor of hidden channels in
            MLP layers. (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_channels (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, Edge feature dimensionality is expected to match
            the `out_channels`. Other-wise, edge features are linearly
            transformed to match `out_channels` of node feature dimensionality.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'mean',
        output_func: Optional[str] = ReLU, # either activation function as string or "mlp" for additional mlp layer
        pass_message: bool = True,
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_channels: Optional[int] = None,
        node_dim: int = 0, # for proper propagation with Node tensor features
        **kwargs,
    ):

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        # Override args of aggregator if `aggr_kwargs` is specified
        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t,
                                             semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        super().__init__(aggr=aggr, node_dim = node_dim, **kwargs)

        self.in_channels = in_channels # input dim per node-class i.e. F on input
        self.out_channels = out_channels # output dim per node-class i.e. D on output
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] != out_channels:
            self.lin_src = Linear(in_channels[0], out_channels, bias=bias)

        if not pass_message:
            self.pass_message = False
        else:
            self.pass_message = True
            if edge_channels is not None and edge_channels != out_channels:
                self.lin_edge = Linear(edge_channels, out_channels, bias=bias)

            if isinstance(self.aggr_module, MultiAggregation):
                aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
            else:
                aggr_out_channels = out_channels

            if aggr_out_channels != out_channels:
                self.lin_aggr_out = Linear(aggr_out_channels, out_channels,
                                           bias=bias)

            if in_channels[1] != out_channels:
                self.lin_dst = Linear(in_channels[1], out_channels, bias=bias)

            channels = [out_channels]
            for i in range(num_layers - 1):
                channels.append(out_channels * expansion)
            channels.append(out_channels)

            if msg_norm:
                self.msg_norm = MessageNorm(learn_msg_scale)

        self.spda = SDPA_layer(out_channels, out_channels)

        if output_func == "mlp":
            self.output_func = MLP(channels, norm=norm, bias=bias)
        else:
            self.output_func = output_func()


    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            self.lin_edge.reset_parameters()
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj = None,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # project input if needed
        if hasattr(self, 'lin_src'):
            x = (self.lin_src(x[0]), x[1])

        if self.pass_message:
            assert edge_index is not None
            # message passing + aggregation
            # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

            # If transformation needed after aggregation
            if hasattr(self, 'lin_aggr_out'):
                out = self.lin_aggr_out(out)

            # If message normalization is needed
            if hasattr(self, 'msg_norm'):
                h = x[1] if x[1] is not None else x[0]
                assert h is not None
                out = self.msg_norm(h, out)

            # Adding the original node features
            x_dst = x[1]
            if x_dst is not None:
                if hasattr(self, 'lin_dst'):
                    x_dst = self.lin_dst(x_dst)
                out = out + x_dst
        else:
            # if not message passing (i.e. first layer)
            out = x[0]

        out = self.spda(out)

        return self.output_func(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        No modification needed here from the original GENConv implementation
        :param x_j:
        :param edge_attr:
        :return:
        """
        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        if edge_attr is not None:
            assert x_j.size(-1) == edge_attr.size(-1)
            if x_j.shape != edge_attr.shape: # means that edge attr is only a M, Fe tensor and not M, K, Fe
                edge_attr = edge_attr.unsqueeze(1).expand(x_j.shape)

        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu() + self.eps

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


from torch_geometric.nn import DeepGCNLayer
import torch.nn as nn


class DeeperNodeAttentionGNN(Module):
    def __init__(self, node_channels, edge_channels, hidden_channels, num_layers, output_channels = 1, aggr = "mean",
                 conv_output_func = ReLU, output_activation = ReLU, edge_decoder = True):
        super().__init__()

        self.node_encoder = Linear(node_channels, hidden_channels)
        self.edge_encoder = Linear(edge_channels, hidden_channels)

        self.layers = ModuleList()
        self.layers.append(NodeAttentionConv(hidden_channels, hidden_channels, pass_message=False))
        for i in range(1, num_layers+1):

            conv = NodeAttentionConv(hidden_channels, hidden_channels,
                                     pass_message=True, edge_channels=hidden_channels,
                                     aggr = aggr, output_func = conv_output_func)


            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.0,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
        if edge_decoder:
            self.edge_decoder = EdgeDecoder(hidden_channels, output_channels, activation = output_activation)
        else:
            self.lin = Linear(hidden_channels, output_channels)
        if output_channels == 1:
            self.scale_param = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale_param = None

        # self.apply(initialize_weights)

    def forward(self, data: Optional[Union[Data,Batch]] = None,
                x : Optional[Tensor] = None, edge_index: Optional[Tensor] = None, edge_attr: Optional[Tensor] = None):
        if data:
            if hasattr(data, 'data'): # to handle weird data handling of non tensor data by tensordict
                data = data.data
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            assert x is not None and edge_index is not None and edge_attr is not None
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0](x)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[1].act(self.layers[1].norm(x))
        if hasattr(self, 'edge_decoder'):
            z = self.edge_decoder(x, edge_index)
        else:
            z = self.lin(x)
        if not self.scale_param is None:
            # concatenate the scale parameter to the output
            # z would be shape (M, K, 1), scale_param is shape []
            z = torch.cat([z, self.scale_param.expand(z.shape)], dim = -1)
        return z

def get_x_variation(x):
    """
    Look at the variation in the node embeddings x, where x[i] is the embedding of node i

    I want to look at
    (x[i] - x[j]).mean( for all i, j in the graph
    """

    return (x.unsqueeze(2) - x.unsqueeze(1)).pow(2).mean()
