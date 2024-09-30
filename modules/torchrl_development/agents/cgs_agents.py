from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
import torch
import torch.nn as nn
import torch.distributions as D
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
)
from torch_geometric.data import Batch

from typing import Tuple
from torchrl.data.tensor_specs import CompositeSpec
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch

import monotonicnetworks as lmn

from modules.torchrl_development.nn_modules.SwitchTypeNetwork import SwitchTypeNetwork as STN

from modules.torchrl_development.agents.utils import MaskedOneHotCategorical

from torchrl.modules import ReparamGradientStrategy

def create_mlp_actor_critic(
        input_shape: Tuple[int, ...],
        output_shape,
        in_keys,
        action_spec,
        actor_nn=None,
        critic_nn=None,
        actor_depth=2,
        actor_cells=32,
        dropout = 0.0
):
    # If no actor network is provided, create a default MLP
    if actor_nn is None:
        actor_nn = MLP(in_features=input_shape[-1],
                       activation_class=torch.nn.Sigmoid,
                       out_features=output_shape,
                       depth=actor_depth,
                       num_cells=actor_cells,
                       activate_last_layer=True,
                       dropout = dropout,
                       )
    # Wrap the actor network in a TensorDictModule
    actor_module = TensorDictModule(
        module=actor_nn,
        in_keys=in_keys,
        out_keys=["probs"],
    )

    # Wrap the actor module in a probabilistic actor
    actor_module = ProbabilisticActor(
        actor_module,
        in_keys=["probs"],
        distribution_class=IndependentBernoulli,
        spec = CompositeSpec(action = action_spec),
        return_log_prob=True,
        default_interaction_type = ExplorationType.RANDOM
    )

    # If no critic network is provided, create a default MLP
    if critic_nn is None:
        critic_nn = MLP(in_features=input_shape[-1],
                        activation_class=torch.nn.ReLU,
                        out_features=1,
                        depth=2,
                        num_cells=32,
                        )
    # Create value module
    value_module = ValueOperator(
        module = critic_nn,
        in_keys = in_keys,
    )

    # Actor Critic Wrapper
    actor_critic = ActorCriticWrapper(actor_module, value_module)
    return actor_critic


class GNN_TensorDictModule(TensorDictModule):

    def __init__(self, module, x_key="observation", edge_index_key="adj_sparse", out_key= "probs"):
        super(GNN_TensorDictModule, self).__init__(module=module, in_keys=[x_key, edge_index_key], out_keys=out_key)
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.out_key = out_key

    def forward(self, input):
        # Three different cases to handle: 1) input is a tensordict of size 1, 2) input is a tensordict of size > 1, 3) input is a batch
        if isinstance(input, TensorDict):
            if input[self.x_key].dim() < 3:  # Case 1: input is a tensordict of size 1
                input[self.out_key] = self.module(input[self.x_key], input[self.edge_index_key], None).squeeze(-1)
                return input
            else: # case 2: input is a tensordict of size > 1
                batch_graph = tensors_to_batch(input[self.x_key], input[self.edge_index_key])
                input[self.out_key] = self.module(batch_graph.x, batch_graph.edge_index, batch_graph.batch).view(batch_graph.batch_size,-1)
                return input
        elif isinstance(input, Batch): # case 3
            return self.module(input.x, input.edge_index, input.batch)

class GNN_ActorTensorDictModule(GNN_TensorDictModule):

    def __init__(self, module, x_key = "observation", edge_index_key = "adj_sparse", out_keys = ["probs", "logits"]):
        super(GNN_ActorTensorDictModule, self).__init__(module=module, x_key=x_key, edge_index_key=edge_index_key, out_key=out_keys)
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.outs_key = out_keys

    def forward(self, input):
        if isinstance(input, TensorDict):
            if input[self.x_key].dim() < 3: # batch size is 1
                probs, logits = self.module(input[self.x_key], input[self.edge_index_key], None)
                input[self.outs_key[0]] = probs.squeeze(-1)
                input[self.outs_key[1]] = logits.squeeze(-1)
            else:
                batch_graph = tensors_to_batch(input[self.x_key], input[self.edge_index_key])
                probs, logits = self.module(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
                input[self.out_keys[0]] = probs.view(batch_graph.batch_size, -1)
                input[self.out_keys[1]] = logits.view(batch_graph.batch_size, -1)
            return input
        elif isinstance(input, Batch):
            return self.module(input.x, input.edge_index, input.batch)


class GNN_CriticTensorDictModule(GNN_TensorDictModule):

    def __init__(self, module, x_key="observation", edge_index_key="adj_sparse", out_keys=["state_value"]):
        super(GNN_CriticTensorDictModule, self).__init__(module=module, x_key=x_key, edge_index_key=edge_index_key,
                                                        out_key=out_keys)
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.outs_key = out_keys

    def forward(self, input):
        if isinstance(input, TensorDict):
            if input[self.x_key].dim() < 3: # batch size is 1
                probs, logits = self.module(input[self.x_key], input[self.edge_index_key], None)
                input[self.outs_key[0]] = probs.squeeze(-1)
                input[self.outs_key[1]] = logits.squeeze(-1)

class IndependentBernoulli(D.Bernoulli):
    """
    For use as a distribution class for a Bernoulli distribution
    The input should be probs

    """

    def __init__(self,
                 probs = None,
                 **kwargs):

        #super().__init__(temperature = torch.Tensor([1.0]), probs = probs, logits = None, **kwargs)
        super().__init__(probs = probs, logits = None, **kwargs)
        # self.num_samples = self._param.shape[-1]

    def entropy(self):
        """
        We are treating each element as independent, thus the entropy of
        $x_1, ..., x_n \sim Bernoulli(p_1, ..., p_n) is \sum_{i=1}^{n}Ent(Bernoulli(p_i))$
        :return:
        """
        return super().entropy().sum(-1)

    def log_prob(self, value):
        """
        We are treating each element as independent, thus the log_prob of
        $x_1, ..., x_n \sim Bernoulli(p_1, ..., p_n) is \sum_{i=1}^{n}log_prob(Bernoulli(p_i))$
        :return:
        """
        return super().log_prob(value).sum(-1)




if __name__ == "__main__":
    dist = IndependentBernoulli(probs = torch.tensor([[0.1, 0.2, 0.3]]))
    s = dist.sample()
    ent = dist.entropy()



