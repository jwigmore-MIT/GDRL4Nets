from tensordict.nn.probabilistic import ProbabilisticTensorDictModule
from torchrl.modules.tensordict_module import SafeProbabilisticTensorDictSequential
from tensordict.nn import ProbabilisticTensorDictSequential
from torchrl.modules import ProbabilisticActor
from torchrl_development.maxweight import MaxWeightActor
from torchrl.data import CompositeSpec
from torchrl.envs import ExplorationType
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from torchrl.modules import Actor, MLP, ProbabilisticActor, ValueOperator, MaskedOneHotCategorical, ActorCriticWrapper, MaskedCategorical, OneHotCategorical
import torch
import torch.nn as nn
from tensordict import TensorDict
import torch.nn.functional as F
from tensordict.nn import (
    dispatch,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictModuleWrapper,
    TensorDictSequential,
)
import warnings
from typing import Optional, Sequence, Tuple, Union
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.modules.tensordict_module.common import SafeModule
from tensordict import TensorDictBase
from torchrl.data.utils import _process_action_space_spec
from torchrl.modules.tensordict_module.sequence import SafeSequential
import monotonicnetworks as lmn




def create_ia_actor(input_shape,
                    output_shape,
                    in_keys,
                    action_spec,
                    threshold):
    actor_mlp = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    activate_last_layer=True,
                    out_features=output_shape,
                    )
    # actor actor_mlp_output = actor_mlp(torch.ones(input_shape))
    actor_module = TensorDictModule(
        module=actor_mlp,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategorical,
        in_keys=["logits", "mask"],
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    ia_actor = InterventionActorWrapper(actor_module, threshold=threshold)
    return ia_actor

def create_ia_actor_critic(input_shape,
                            output_shape,
                            in_keys,
                            action_spec,
                            threshold):
    critic_mlp = MLP(in_features=input_shape[-1],
                     activation_class=torch.nn.ReLU,
                     activate_last_layer=True,
                     out_features=1,
                     )
    critic_mlp_output = critic_mlp(torch.ones(input_shape))
    value_module = ValueOperator(
        module=critic_mlp,
        in_keys=in_keys,
    )
    ia_actor = create_ia_actor(input_shape, output_shape, in_keys, action_spec, threshold)
    ia_actor_critic = InterventionActorCriticWrapper(ia_actor, value_module)
    return ia_actor_critic

class InterventionActorWrapper(TensorDictModuleWrapper):

    def __init__(self, actor, threshold, **kwargs):
        super().__init__(td_module = actor, **kwargs)
        self.threshold = threshold
        self.intervention_policy = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

    def forward(self, td):
        # with probabily 0.5 use intervention policy
        intervene = td["backlog"] > self.threshold
        td["intervene"] = intervene
        if intervene:
            return self.intervention_policy.forward(td)
        else:
            return self.td_module.forward(td)


class InterventionActorCriticWrapper(ActorCriticWrapper):

    def __init__(self, ia_actor, critic):
        super().__init__(ia_actor.td_module, critic)
        self.ia_actor = ia_actor


# For standard actors
def create_actor_critic(input_shape,
             output_shape,
             in_keys,
             action_spec,
             actor_nn = None,
             critic_nn = None,
             temperature = 1.0,
             device="cpu",
             actor_depth = 2,
             actor_cells = 32,
             ):

    if actor_nn is None:
        actor_nn = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    out_features=output_shape,
                    depth=actor_depth,
                    num_cells=actor_cells,
                    )

    actor_mlp_output = actor_nn(torch.ones(input_shape))
    actor_module = TensorDictModule(
        module=actor_nn,
        in_keys=in_keys,
        out_keys=["logits"],
    )

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},#{"temperature": 0.5},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    if critic_nn is None:
        critic_nn = MLP(in_features=input_shape[-1],
                        activation_class=torch.nn.ReLU,
                        out_features=1,
                        )

    critic_mlp_output = critic_nn(torch.ones(input_shape))


    # Create Value Module
    value_module = ValueOperator(
        module=critic_nn,
        in_keys=in_keys,
    )

    #
    actor_critic = ActorCriticWrapper(actor_module, value_module)


    return actor_critic



def create_maxweight_actor_critic(input_shape,
                in_keys,
                action_spec,
                temperature = 1.0,
                device="cpu",
                init_weights = None):
    weight_size = input_shape[-1]//2
    actor_network = MaxWeightNetwork(weight_size, temperature=temperature, weights=init_weights)
    actor_module = NN_Actor(module=actor_network, in_keys=["Q", "Y"], out_keys=["logits"])

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},  # {"temperature": 0.5},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    # input = TensorDict({"Q": torch.ones(1,weight_size), "Y": torch.ones(1,weight_size),
    #                     "mask": torch.Tensor([[1,0,0,0,0,0]]).bool()}, batch_size=(1,))

    # actor_output = actor_module(input)


    critic_mlp = MLP(in_features=input_shape[-1],
                     activation_class=torch.nn.ReLU,
                     out_features=1,
                     )


    # Create Value Module
    value_module = ValueOperator(
        module=critic_mlp,
        in_keys=in_keys,
    )

    #
    actor_critic = ActorCriticWrapper(actor_module, value_module)

    actor_critic.mwn_weights = actor_network.get_weights
    return actor_critic


class NN_Actor(TensorDictModule):

    def __init__(self, module, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td):
        #x = torch.cat([td[in_key] for in_key in self.in_keys]).unsqueeze(0)
        td["logits"] = self.module(td["Q"], td["Y"])
        return td

class Ind_NN_Actor(TensorDictModule):

        def __init__(self, module, N, d,  in_keys = ["observation"], out_keys = "logits" ):
            super().__init__(module= module, in_keys = in_keys, out_keys=out_keys)
            self.N =N
            self.d = d
        def forward(self, td):
            """
            td["observation"] is a B x (d * N) tensor
            We want to convert this to a B x N x d tensor before passing to the module
            :param td:
            :return:
            """
            x = td["observation"].view(-1, self.d, self.N)
            td[self.out_keys[0]] = self.module(x)
            return td


class MaxWeightNetwork(nn.Module):
    def __init__(self, weight_size, temperature=1.0, weights = None):
        super(MaxWeightNetwork, self).__init__()
        # randomly initialize the weights to be gaussian with mean 1 and std 0.1
        if weights is None:
            weights = 1+torch.randn(weight_size)/10**0.5
        if weights.dim() == 1:
            weights.unsqueeze_(0)

        # make sure weights have shape (1, weight_size)
        self.weights = nn.Parameter(weights)

        self.bias_0 = nn.Parameter(torch.ones(1,1))
        self.temperature = temperature
    def forward(self, Q, Y):
        # First get dimensions right for the Q and Y tensors
        if Q.dim == 1:
            Q.unsqueeze_(0)
        if Y.dim == 1:
            Y.unsqueeze_(0)

        z_0 = self.bias_0 - (Q * Y).sum(dim=-1, keepdim=True)
        z_i = Y * Q * self.weights
        z = torch.cat([z_0, z_i], dim=-1)
        # z = F.softmax(z, dim=-1)


        return torch.relu(z).squeeze()

    def get_weights(self):
        "returns a copy of bias_0 and weights as a single vector"
        return torch.cat([self.bias_0.detach(), self.weights.detach()], dim = -1)



def create_independent_actor_critic(number_nodes,
                actor_input_dimension,
                actor_in_keys,
                critic_in_keys,
                action_spec,
                temperature = 1.0,
                actor_depth = 2,
                actor_cells = 32,
                type = 1,
                network_type = "MLP",
                relu_max= 10):

    if type == 1:
        actor_network = IndependentNodeNetwork(actor_input_dimension, number_nodes,actor_depth, actor_cells, network_type, relu_max)
    elif type == 2:
        actor_network = IndependentNodeNetwork2(number_nodes, actor_depth, actor_cells)
    elif type == 3:
        actor_network = SharedIndependentNodeNetwork(actor_input_dimension, number_nodes,actor_depth, actor_cells, network_type, relu_max)

    actor_module = Ind_NN_Actor(actor_network, N = number_nodes, d = actor_input_dimension, in_keys=actor_in_keys, out_keys=["logits"])

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},  # {"temperature": 0.5},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    # input = TensorDict({"Q": torch.ones(1,weight_size), "Y": torch.ones(1,weight_size),
    #                     "mask": torch.Tensor([[1,0,0,0,0,0]]).bool()}, batch_size=(1,))

    # actor_output = actor_module(input)


    critic_mlp = MLP(in_features=number_nodes*actor_input_dimension,
                     activation_class=torch.nn.ReLU,
                     out_features=1,
                     )


    # Create Value Module
    value_module = ValueOperator(
        module=critic_mlp,
        in_keys=critic_in_keys,
    )

    #
    actor_critic = ActorCriticWrapper(actor_module, value_module)

    return actor_critic

class IndependentNodeNetwork(nn.Module):
    """
    This network takes the node states $\mathbf s = (s_i)$ where $s_i = (q_i, y_i)$
    and passes each s_i through its own independent neural network
    f_{\theta_i}(s_i) to compute logits z_i
    """
    def __init__(self, input_dim, num_nodes, depth, cells, network_type = "MLP", relu_max = 1):
        super(IndependentNodeNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        # create num_nodes independent neural networks, each with depth and cells
        if network_type == "MLP":
            self.node_nns = nn.ModuleList([MLP(in_features=input_dim,
                                               out_features=1,
                                               depth=depth,
                                               num_cells=cells,
                                               activation_class=nn.ReLU,
                                               ) for _ in range(num_nodes)])
        elif network_type == "LMN":
            module_list = []
            for _ in range(num_nodes):
                lip_nn = torch.nn.Sequential(
                lmn.LipschitzLinear(input_dim, cells, kind="one", lipschitz_const=1),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(cells,  cells, kind="one", lipschitz_const=1),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(cells, 1, kind="one", lipschitz_const=1))
                mono_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[1]*input_dim)
                module_list.append(mono_nn)
            self.node_nns= torch.nn.ModuleList(module_list)
        elif network_type == "PMN":
            from torchrl_development.SMNN import PureMonotonicNeuralNetwork as PMN
            from copy import deepcopy
            module_list = []
            base_pmn = PMN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max= relu_max)
            for _ in range(num_nodes):
                # pmn = PMN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max=relu_max)
                # pmn.load_state_dict(deepcopy(base_pmn.state_dict()))
                module_list.append(PMN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max=10))
                # module_list.append(pmn)
            self.node_nns = torch.nn.ModuleList(module_list)



        # self.node_nns = nn.ModuleList([nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # ) for _ in range(num_nodes)])
        self.bias_0 = nn.Parameter(torch.ones(1, 1))
    def forward(self, X):
        """
        X will be a B x d x N tensor where N = self.num_nodes and D = input_dim
        Want to pass each B x d x 1 tensor through the corresponding neural network
        The output will be a B x N tensor of logits

        :param X:
        :return:
        """
        # Handle the case where X is a N x D tensor
        if X.dim() == 2:
            X.unsqueeze_(0)

        # Create a B x 1 Tensor of zeros
        zeros = torch.zeros(X.shape[0], 1)
        # Initialize an empty list to store the output of each node's neural network
        output_list = [zeros]

        # Iterate over each node
        for i in range(self.num_nodes):
            # Extract the input for the current node
            node_input = X[:, :, i]

            # Pass the node's input through its corresponding neural network
            node_output = self.node_nns[i](node_input)

            # Append the output to the list
            output_list.append(node_output)

        # Stack the outputs to form a tensor of shape (B, N+1)
        output_tensor = torch.stack(output_list, dim=-2)

        return output_tensor.squeeze(-1)


class SharedIndependentNodeNetwork(nn.Module):
    """
    The state-space is broken up into (s_11, s_12, ... s_1d, s_21, s_22, ... s_2d, ... s_N1, s_N2, ... s_Nd)
    We want to run each s_i1:s_id through a shared neural network independently
    e.g. We get input s, and must reshape it to (B, D, N) where N is the number of nodes and D is the dimension of each node
    Then we pass each (B, D) tensor through the same neural network to get (B, N) logits
    Then take the softmax over dimension 1 to get the probabalities
    """
    def __init__(self, input_dim, num_nodes, depth, cells, network_type = "MLP", relu_max = 1):
        super(SharedIndependentNodeNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        # create num_nodes independent neural networks, each with depth and cells
        if network_type == "MLP":
            self.module = MLP(in_features=input_dim,
                                                  out_features=1,
                                                    depth=depth,
                                                    num_cells=cells,
                                                    activation_class=nn.ReLU,
                                                    )
        # elif network_type == "LMN":
        #     module_list = []
        #     for _ in range(num_nodes):
        #         lip_nn = torch.nn.Sequential(
        #         lmn.LipschitzLinear(input_dim, cells, kind="one", lipschitz_const=1),
        #         lmn.GroupSort(2),
        #         lmn.LipschitzLinear(cells,  cells, kind="one", lipschitz_const=1),
        #         lmn.GroupSort(2),
        #         lmn.LipschitzLinear(cells, 1, kind="one", lipschitz_const=1))
        #         mono_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[1]*input_dim)
        #         module_list.append(mono_nn)
        #     self.node_nns= torch.nn.ModuleList(module_list)
        elif network_type == "PMN":
            from torchrl_development.SMNN import PureMonotonicNeuralNetwork as PMN
            from copy import deepcopy

            self.module = PMN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max = relu_max)


        self.bias_0 = nn.Parameter(torch.ones(1, 1))
    def forward(self, X):
        """
        X will be a B x N x d tensor where N = self.num_nodes and D = input_dim
        Want to pass each B x 1  x d tensor through the neural shared self.module
        This will produce a B x N x 1 tensor of logits

        :param X:
        :return:
        """
        # Handle the case where X is a N x D tensor
        if X.dim() == 2:
            X.unsqueeze_(0)

        # Create a B x 1 Tensor of zeros
        zeros = torch.zeros(X.shape[0], 1)
        # Initialize an empty list to store the output of each node's neural network
        output_list = [zeros]

        # Pass the node inputs through the shared neural network
        for i in range(self.num_nodes):
            output_list.append(self.module(X[:, :, i]))




        # Stack the outputs to form a tensor of shape (B, N+1)
        output_tensor = torch.stack(output_list, dim=-2)

        return output_tensor.squeeze(-1)


class IndependentLMNNetwork(nn.Module):
    """
    Like IndependentNodeNetwork but using Lipsc
    """

class IndependentNodeNetwork2(nn.Module):
    """
    This network takes the node states $\mathbf s = (s_i)$ where $s_i = (q_i, y_i)$
    and passes each s_i through its own independent neural network
    f_{\theta_i}(s_i) to compute logits z_i
    """
    def __init__(self, num_nodes, depth, cells):
        super(IndependentNodeNetwork2, self).__init__()
        self.num_nodes = num_nodes
        # create num_nodes independent neural networks, each with depth and cells
        self.node_nns = nn.ModuleList([MLP(in_features=1,
                                           out_features=1,
                                           depth=depth,
                                           num_cells=cells,
                                           activation_class=nn.ReLU,
                                           ) for _ in range(num_nodes)])



        # self.node_nns = nn.ModuleList([nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # ) for _ in range(num_nodes)])
        self.bias_0 = nn.Parameter(torch.ones(1, 1))
    def forward(self, Q, Y):
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        # Elementwise multiplication of Q and Y
        s = [(Q[:,i]* Y[:,i]).unsqueeze(1) for i in range(Q.shape[-1])]


        z_i = [nn(s_i) for s_i, nn in zip(s, self.node_nns)]
        z_i = torch.column_stack(z_i)
        z_0 = self.bias_0 - (Q * Y).sum(dim=-1, keepdim=True)
        z = torch.column_stack([z_0, z_i])
        # normalize z
        z = F.softmax(z, dim=-1)
        return torch.relu(z).squeeze(dim=-1)

def create_weight_function_network(input_shape,
                arrival_rates,
                service_rates,
                in_keys,
                action_spec,
                temperature = 1.0,
                actor_depth = 2,
                actor_cells = 32,
                type = 1):

    num_nodes = input_shape[-1]//2

    actor_network = WeightFunctionNetworkNode(num_nodes, arrival_rates, service_rates, actor_depth, actor_cells)

    actor_module = NN_Actor(module=actor_network, in_keys=["Q", "Y"], out_keys=["logits"])

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},  # {"temperature": 0.5},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    # input = TensorDict({"Q": torch.ones(1,weight_size), "Y": torch.ones(1,weight_size),
    #                     "mask": torch.Tensor([[1,0,0,0,0,0]]).bool()}, batch_size=(1,))

    # actor_output = actor_module(input)


    critic_mlp = MLP(in_features=input_shape[-1],
                     activation_class=torch.nn.ReLU,
                     out_features=1,
                     )


    # Create Value Module
    value_module = ValueOperator(
        module=critic_mlp,
        in_keys=in_keys,
    )

    #
    actor_critic = ActorCriticWrapper(actor_module, value_module)

    return actor_critic

class WeightFunctionNetworkNode(nn.Module):
    """
    This network takes the node states $\mathbf s = (s_i)$ where $s_i = (q_i, y_i)$
    and passes each s_i through its own independent neural network
    f_{\theta_i}(s_i) to compute logits z_i
    """
    def __init__(self, num_nodes, arrival_rates, service_rates, depth, cells):
        super().__init__()
        self.num_nodes = num_nodes
        # create num_nodes independent neural networks, each with depth and cells
        self.NN = MLP(in_features=4,
                       out_features=1,
                       depth=depth,
                       num_cells=cells,
                       activation_class=nn.ReLU,
                       )

        self.arrival_rates = arrival_rates
        self.service_rates = service_rates


        self.bias_0 = nn.Parameter(torch.ones(1, 1))
    def forward(self, Q, Y):
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        # Stack Q, Y, Arrival Rate i, Service Rate i
        s = [torch.column_stack([Q[:,i], Y[:,i], torch.ones_like(Q[:,i])*self.arrival_rates[i], torch.ones_like(Q[:,i])*self.service_rates[i]]) for i in range(Q.shape[-1])]


        z_i = [self.NN(s_i) for s_i in s]
        z_i = torch.column_stack(z_i)
        z_0 = self.bias_0 - (Q * Y).sum(dim=-1, keepdim=True)
        z = torch.column_stack([z_0, z_i])
        # normalize z
        z = F.softmax(z, dim=-1)
        return torch.relu(z).squeeze(dim=-1)


class GNNMaxWeightNetwork(nn.Module):
    """
    A GNN-like network that takes in a set of Q and Y values,
    computes a function f(q_i,y_i) for each pair of Q and Y values
    and then computes the max weight of the resulting values.
    """

    def __init__(self):
        super(GNNMaxWeightNetwork, self).__init__()
        self.gnn = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias_0 = nn.Parameter(torch.ones(1, 1))

    def forward(self, Q, Y):
        # First get dimensions right for the Q and Y tensors
        if Q.dim == 1:
            Q = Q.unsqueeze(0)
        if Y.dim == 1:
            Y = Y.unsqueeze(0)
        Q = Q.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        # Compute the pairwise interactions
        # Q and Y have shape (batch_size, N)
        # Need to pass to the GNN a tensor of shape (batch_size, N, 2), where (b, n, 0) = Q[b, n] and (b, n, 1) = Y[b, n]
        #z_i = self.gnn(torch.cat([Q.unsqueeze(-1), Y.unsqueeze(-1)], dim=-1))
        z_i = self.gnn(torch.cat([Q, Y], dim=-1))
        # Compute the z_0 term
        z_0 = self.bias_0 - (Q * Y).sum(dim=-2, keepdim=True)

        # Concatenate the z_0 term with the pairwise interactions
        z = torch.cat([z_0, z_i], dim=-2)

        # output logits
        return torch.relu(z).squeeze(dim=-1)


class GNNMaxWeightNetwork2(nn.Module):
    """
    A GNN-like network that takes in a set of Q and Y values,
    computes a function f(q_i,y_i) for each pair of Q and Y values

    """

    def __init__(self):
        super(GNNMaxWeightNetwork2, self).__init__()
        self.message_nn1 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        self.bias_0 = nn.Parameter(torch.ones(1, 1))

        self.message_nn2 = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.updater = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def message(self, S_Nu):
        """
        S_Nu (B, N-1, F) is the set of all states of the neighbors of node u

        :param s_u:
        :param s_v:
        :return:
        """
        m1 = self.message_nn1(S_Nu) # (batch_size, N-1, F) -> (batch_size, N-1, 5)
        m2 = m1.max(dim=-2).values # (batch_size, 1, 5)
        m3 = self.message_nn2(m2) # (batch_size, 1, 1)
        return m3

    def update(self, S, M):
        """
        S (B, N, F) is the set of all states of the nodes
        M (B, N) is the set of all messages

        For each n, we want to concatenate S(B, n, F) and M(B, n) along the last dimension,
        Then pass this through a neural network to get the updated state

        The output should be of shape (B, N, 1)


        :param S:
        :param M:
        :return:
        """
        U = []
        for n in range(S.shape[-2]):
            if S.dim() == 2:
                U.append(self.updater(torch.cat([S[n], M[-1]], dim=-1)))
            elif S.dim() == 3:
                U.append(self.updater(torch.cat([S[:, n, :], M[:, n]], dim=-1)))
        U = torch.stack(U, dim=-2)

        return U

    def forward(self, Q, Y):
        # First get dimensions right for the Q and Y tensors
        if Q.dim == 1:
            Q = Q.unsqueeze(0)
        if Y.dim == 1:
            Y = Y.unsqueeze(0)
        Q = Q.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        S = torch.cat([Q, Y], dim=-1)
        # Message Layer
        """
        For each connected pair (u,v) we want to compute a mesage m_{u,v}
        m_{u,v} = message_nn1(
        """
        M = []
        for i in range(Q.shape[1]):
            if S.dim() == 2:
                S_Nu = S[torch.arange(Q.shape[-2]) != i,:] # how do we do this when S can be shape (B, N, F) or (N,F)
            elif S.dim() == 3:
                S_Nu = S[:, torch.arange(Q.shape[-2]) != i] # how do we do this when S can be shape (B, N, F) or (N,F)
            M.append(self.message(S_Nu)) # Each element, (b,i) is the aggregated message from all neighbors to node i for batch b
        M = torch.stack(M, dim=-2) # (B, N, 1)
        U = self.update(S, M)

        # Repeat m_i1 along the second dimension

        # m_i1_repeated = m_i1.repeat_interleave(m_i1.shape[1], dim=2)


        # Compute the z_0 term
        z_0 = self.bias_0 - (Q * Y).sum(dim=-2, keepdim=True)

        # Concatenate the z_0 term with the pairwise interactions
        z = torch.cat([z_0, U], dim=-2)

        # output logits
        return torch.relu(z).squeeze(dim=-1)



def create_gnn_maxweight_actor_critic(input_shape,
                in_keys,
                action_spec,
                temperature = 1.0,
                type = 1):
    if type == 1:
        actor_network = GNNMaxWeightNetwork()
    elif type == 2:
        actor_network = GNNMaxWeightNetwork2()
    actor_module = NN_Actor(module=actor_network, in_keys=["Q", "Y"], out_keys=["logits"])

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},  # {"temperature": 0.5},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    # input = TensorDict({"Q": torch.ones(1,weight_size), "Y": torch.ones(1,weight_size),
    #                     "mask": torch.Tensor([[1,0,0,0,0,0]]).bool()}, batch_size=(1,))

    # actor_output = actor_module(input)


    critic_mlp = MLP(in_features=input_shape[-1],
                     activation_class=torch.nn.ReLU,
                     out_features=1,
                     )


    # Create Value Module
    value_module = ValueOperator(
        module=critic_mlp,
        in_keys=in_keys,
    )

    #
    actor_critic = ActorCriticWrapper(actor_module, value_module)

    return actor_critic

class MDP_actor(TensorDictModule):

    def __init__(self, mdp_module, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= mdp_module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["Y"])
        return td

class MDP_module(torch.nn.Module):

    def __init__(self, mdp, policy_type = "VI"):
        super().__init__()
        self.mdp = mdp
        self.policy_type = policy_type


    def forward(self, Q, Y):
        state = torch.concatenate([Q, Y]).tolist()
        if self.policy_type == "VI":
            return self.mdp.use_vi_policy(state)
        elif self.policy_type == "PI":
            return self.mdp.use_pi_policy(state)
        else:
            raise ValueError(f"Unknown policy type {self.policy_type}")




class MinQValueModule(TensorDictModuleBase):
    """Q-Value TensorDictModule for Q-value policies.

    This module processes a tensor containing action value into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions, action values and chosen action value.
            Defaults to ``["action", "action_value", "chosen_action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Returns:
        if the input is a single tensor, a triplet containing the chosen action,
        the values and the value of the chose action is returned. If a tensordict
        is provided, it is updated with these entries at the keys indicated by the
        ``out_keys`` field.

    Examples:
        >>> from tensordict import TensorDict
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> actor = QValueModule(action_space, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.zeros(4)
        >>> value[-1] = 1
        >>> actor(my_action_value=value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated in v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
        self.action_space = action_space
        self.var_nums = var_nums
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}, got {action_space}"
            )
        if action_value_key is None:
            action_value_key = "action_value"
        self.action_mask_key = action_mask_key
        in_keys = [action_value_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = ["action", action_value_key, "chosen_action_value"]
        elif action_value_key not in out_keys:
            raise RuntimeError(
                f"Expected the action-value key to be '{action_value_key}' but got {out_keys[1]} instead."
            )
        self.out_keys = out_keys
        action_key = out_keys[0]
        if not isinstance(spec, CompositeSpec):
            spec = CompositeSpec({action_key: spec})
        super().__init__()
        self.register_spec(safe=safe, spec=spec)

    register_spec = SafeModule.register_spec

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    @property
    def action_value_key(self):
        return self.in_keys[0]

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).max
            )

        action = self.action_func_mapping[self.action_space](action_values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return tensordict

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.min(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmin(value, dim=-1).to(torch.long)

    def _mult_one_hot(
        self, value: torch.Tensor, support: torch.Tensor = None
    ) -> torch.Tensor:
        if self.var_nums is None:
            raise ValueError(
                "var_nums must be provided to the constructor for multi one-hot action spaces."
            )
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(
                    _value,
                )
                for _value in values
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return (action * values).sum(-1, True)

    @staticmethod
    def _categorical_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return values.gather(-1, action.unsqueeze(-1))
        # if values.ndim == 1:
        #     return values[action].unsqueeze(-1)
        # batch_size = values.size(0)
        # return values[range(batch_size), action].unsqueeze(-1)

class MinQValueActor(SafeSequential):
    """A Q-Value actor class.

    This class appends a :class:`~.QValueModule` after the input module
    such that the action values are used to select an action.

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space. If the class provided is not compatible
            with :class:`tensordict.nn.TensorDictModuleBase`, it will be
            wrapped in a :class:`tensordict.nn.TensorDictModule` with
            ``in_keys`` indicated by the following keyword argument.

    Keyword Args:
        in_keys (iterable of str, optional): If the class provided is not
            compatible with :class:`tensordict.nn.TensorDictModuleBase`, this
            list of keys indicates what observations need to be passed to the
            wrapped module to get the action values.
            Defaults to ``["observation"]``.
        spec (TensorSpec, optional): Keyword-only argument.
            Specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): Keyword-only argument.
            If ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): if the input module
            is a :class:`tensordict.nn.TensorDictModuleBase` instance, it must
            match one of its output keys. Otherwise, this string represents
            the name of the action-value entry in the output tensordict.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).

    .. note::
        ``out_keys`` cannot be passed. If the module is a :class:`tensordict.nn.TensorDictModule`
        instance, the out_keys will be updated accordingly. For regular
        :class:`torch.nn.Module` instance, the triplet ``["action", action_value_key, "chosen_action_value"]``
        will be used.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueActor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> # with a regular nn.Module
        >>> module = nn.Linear(4, 4)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)
        >>> # with a TensorDictModule
        >>> td = TensorDict({'obs': torch.randn(5, 4)}, [5])
        >>> module = TensorDictModule(lambda x: x, in_keys=["obs"], out_keys=["action_value"])
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module,
        *,
        in_keys=None,
        spec=None,
        safe=False,
        action_space: Optional[str] = None,
        action_value_key=None,
        action_mask_key: Optional[NestedKey] = None,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)

        self.action_space = action_space
        self.action_value_key = action_value_key
        if action_value_key is None:
            action_value_key = "action_value"
        out_keys = [
            "action",
            action_value_key,
            "chosen_action_value",
        ]
        if isinstance(module, TensorDictModuleBase):
            if action_value_key not in module.out_keys:
                raise KeyError(
                    f"The key '{action_value_key}' is not part of the module out-keys."
                )
        else:
            if in_keys is None:
                in_keys = ["observation"]
            module = TensorDictModule(
                module, in_keys=in_keys, out_keys=[action_value_key]
            )
        if spec is None:
            spec = CompositeSpec()
        if isinstance(spec, CompositeSpec):
            spec = spec.clone()
            if "action" not in spec.keys():
                spec["action"] = None
        else:
            spec = CompositeSpec(action=spec, shape=spec.shape[:-1])
        spec[action_value_key] = None
        spec["chosen_action_value"] = None
        qvalue = MinQValueModule(
            action_value_key=action_value_key,
            out_keys=out_keys,
            spec=spec,
            safe=safe,
            action_space=action_space,
            action_mask_key=action_mask_key,
        )

        super().__init__(module, qvalue)


class ProbabilisticMinQValueModule(TensorDictModuleBase):
    """Q-Value TensorDictModule for Q-value policies.

    This module processes a tensor containing action value into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions, action values and chosen action value.
            Defaults to ``["action", "action_value", "chosen_action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Returns:
        if the input is a single tensor, a triplet containing the chosen action,
        the values and the value of the chose action is returned. If a tensordict
        is provided, it is updated with these entries at the keys indicated by the
        ``out_keys`` field.

    Examples:
        >>> from tensordict import TensorDict
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> actor = QValueModule(action_space, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.zeros(4)
        >>> value[-1] = 1
        >>> actor(my_action_value=value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated in v0.4.0,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
        self.action_space = action_space
        self.var_nums = var_nums
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}, got {action_space}"
            )
        if action_value_key is None:
            action_value_key = "action_value"
        self.action_mask_key = action_mask_key
        in_keys = [action_value_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = ["action", action_value_key, "chosen_action_value"]
        elif action_value_key not in out_keys:
            raise RuntimeError(
                f"Expected the action-value key to be '{action_value_key}' but got {out_keys[1]} instead."
            )
        self.out_keys = out_keys
        action_key = out_keys[0]
        if not isinstance(spec, CompositeSpec):
            spec = CompositeSpec({action_key: spec})
        super().__init__()
        self.register_spec(safe=safe, spec=spec)

    register_spec = SafeModule.register_spec

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    @property
    def action_value_key(self):
        return self.in_keys[0]

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).max
            )

        action = self.action_func_mapping[self.action_space](action_values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return tensordict

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.min(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmin(value, dim=-1).to(torch.long)

    def _mult_one_hot(
        self, value: torch.Tensor, support: torch.Tensor = None
    ) -> torch.Tensor:
        if self.var_nums is None:
            raise ValueError(
                "var_nums must be provided to the constructor for multi one-hot action spaces."
            )
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(
                    _value,
                )
                for _value in values
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return (action * values).sum(-1, True)

    @staticmethod
    def _categorical_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return values.gather(-1, action.unsqueeze(-1))
        # if values.ndim == 1:
        #     return values[action].unsqueeze(-1)
        # batch_size = values.size(0)
        # return values[range(batch_size), action].unsqueeze(-1)






from typing import Optional, Union, Sequence
from torchrl.modules import ReparamGradientStrategy
from torchrl.modules.distributions.discrete import _one_hot_wrapper
import torch.distributions as D

class MaskedOneHotCategoricalTemp(MaskedCategorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (float, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in then mask tensor when
            sparse_mask == True, the padding_value will be ignored.
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             samples gradients.
            ``ReparamGradientStrategy.RelaxedOneHot`` will use
            :class:`torch.distributions.RelaxedOneHot` to sample from the distribution.

        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedOneHotCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> sample_non_valid = torch.zeros_like(sample)
        >>> sample_non_valid[..., 1] = 1
        >>> print(dist.log_prob(sample_non_valid))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedOneHotCategorical(probs=prob, mask=mask)
        >>> s = torch.arange(10)
        >>> s = torch.nn.functional.one_hot(s, 10)
        >>> print(dist.log_prob(s))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
        grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
        temperature: float = 0.1,
    ) -> None:
        self.grad_method = grad_method
        self.has_rsample = True
        self.temperature = temperature
        logits = logits/temperature
        super().__init__(
            logits=logits,
            probs=probs,
            mask=mask,
            indices=indices,
            neg_inf=neg_inf,
            padding_value=padding_value,
        )

    @_one_hot_wrapper(MaskedCategorical)
    def sample(
        self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None
    ) -> torch.Tensor:
        ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # this calls the log_prob method of the MaskedCategorical class
        log_prob = super().log_prob(value.argmax(dim=-1, keepdim=False))
        # if value.shape[0] > 1:
        #     log_prob = log_prob.unsqueeze(-1)
        return log_prob

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            # get the argmax of the logits
            """
            logits is either (W,N) or (N,)
            logits.argmax() is (W,)
            Want to return (W,N) or (N,) tensor of one-hot vectors
            How do I do this?
            
            """
            argmax = self.logits.argmax(dim=-1, keepdim=True)
            argmax_one_hot = torch.zeros_like(self.logits)
            argmax_one_hot.scatter_(-1, argmax, 1)
            return argmax_one_hot.to(torch.long)

        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def rsample(self, sample_shape: Union[torch.Size, Sequence] = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        if hasattr(self, "logits") and self.logits is not None:
            logits = self.logits
            probs = None
        else:
            logits = None
            probs = self.probs
        if self.grad_method == ReparamGradientStrategy.RelaxedOneHot:
            if self._sparse_mask:
                if probs is not None:
                    probs_extended = torch.full(
                        (*probs.shape[:-1], self.num_samples),
                        0,
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    probs_extended = torch.scatter(
                        probs_extended, -1, self._mask, probs
                    )
                    logits_extended = None
                else:
                    probs_extended = torch.full(
                        (*logits.shape[:-1], self.num_samples),
                        self.neg_inf,
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    logits_extended = torch.scatter(
                        probs_extended, -1, self._mask, logits
                    )
                    probs_extended = None
            else:
                probs_extended = probs
                logits_extended = logits

            d = D.relaxed_categorical.RelaxedOneHotCategorical(
                1, probs=probs_extended, logits=logits_extended
            )
            out = d.rsample(sample_shape)
            out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
            return out
        elif self.grad_method == ReparamGradientStrategy.PassThrough:
            if logits is not None:
                probs = self.probs
            else:
                probs = torch.softmax(self.logits, dim=-1)
            if self._sparse_mask:
                probs_extended = torch.full(
                    (*probs.shape[:-1], self.num_samples),
                    0,
                    device=probs.device,
                    dtype=probs.dtype,
                )
                probs_extended = torch.scatter(probs_extended, -1, self._mask, probs)
            else:
                probs_extended = probs

            out = self.sample(sample_shape)
            out = out + probs_extended - probs_extended.detach()
            return out
        else:
            raise ValueError(
                f"Unknown reparametrization strategy {self.reparam_strategy}."
            )




