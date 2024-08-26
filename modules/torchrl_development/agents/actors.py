
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
)
from typing import Tuple
from torchrl.data.tensor_specs import CompositeSpec

import monotonicnetworks as lmn

from modules.torchrl_development.nn_modules.SwitchTypeNetwork import SwitchTypeNetwork as STN

from modules.torchrl_development.agents.utils import MaskedOneHotCategorical

from torchrl.modules import ReparamGradientStrategy





# For standard actors
def create_actor_critic(input_shape: Tuple[int, ...],
         output_shape,
         in_keys,
         action_spec,
         actor_nn = None,
         critic_nn = None,
         temperature = 1.0,
         actor_depth = 2,
         actor_cells = 32,
         ):
    """
    Function to create an actor-critic model.

    Args:
        input_shape (Tuple[int, ...]): The shape of the input data.
        output_shape: The shape of the output data.
        in_keys: The keys to access the input data.
        action_spec: The specification of the action space.
        actor_nn (nn.Module, optional): The neural network for the actor. If None, a default MLP is created.
        critic_nn (nn.Module, optional): The neural network for the critic. If None, a default MLP is created.
        temperature (float, optional): The temperature parameter for the softmax function in the actor. Default is 1.0.
        actor_depth (int, optional): The depth of the actor's MLP if actor_nn is None. Default is 2.
        actor_cells (int, optional): The number of cells in the actor's MLP if actor_nn is None. Default is 32.

    Returns:
        actor_critic (ActorCriticWrapper): The created actor-critic model.
    """

    # If no actor network is provided, create a default MLP
    if actor_nn is None:
        actor_nn = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    out_features=output_shape,
                    depth=actor_depth,
                    num_cells=actor_cells,
                    )

    # Generate output from the actor network
    actor_mlp_output = actor_nn(torch.ones(input_shape))

    # Wrap the actor network in a TensorDictModule
    actor_module = TensorDictModule(
        module=actor_nn,
        in_keys=in_keys,
        out_keys=["logits"],
    )

    # Wrap the actor module in a ProbabilisticActor
    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategorical,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": temperature},
        spec=CompositeSpec(action=action_spec),
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # If no critic network is provided, create a default MLP
    if critic_nn is None:
        critic_nn = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    out_features=1,
                    )

    # Generate output from the critic network
    critic_mlp_output = critic_nn(torch.ones(input_shape))

    # Create Value Module
    value_module = ValueOperator(
        module=critic_nn,
        in_keys=in_keys,
    )

    # Wrap the actor and value modules in an ActorCriticWrapper
    actor_critic = ActorCriticWrapper(actor_module, value_module)

    return actor_critic




class NN_Actor(TensorDictModule):
    """
    A class that represents an actor in a reinforcement learning model.

    Args:
        module (nn.Module): The neural network module that the actor uses to generate actions.
        in_keys (list[str], optional): The keys that the actor uses to retrieve input from the TensorDict. Defaults to ["Q", "Y"].
        out_keys (list[str], optional): The keys that the actor uses to store output in the TensorDict. Defaults to ["action"].
    """

    def __init__(self, module: nn.Module, in_keys: list[str] = ["Q", "Y"], out_keys: list[str] = ["action"]):
        super().__init__(module= module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict) -> TensorDict:
        """
        The method that the actor uses to generate actions.

        Args:
            td (TensorDict): The TensorDict that the actor retrieves input from and stores output in.

        Returns:
            TensorDict: The TensorDict with the generated action stored under the specified output keys.
        """
        td["logits"] = self.module(*[td[key] for key in self.in_keys])
        return td

class Ind_NN_Actor(TensorDictModule):
    """
    A class that represents an actor that uses a single neural network on each substate vector,
    where each state is comprised of N substate vectors e.g. (q_i, y_i, lambda_i, mu_i) for i in range(N)

    Args:
        module (nn.Module): The neural network module that the actor uses to generate actions.
        N (int): The number of nodes in the input.
        d (int): The number of features (dimension) of each node.
        in_keys (list[str], optional): The keys that the actor uses to retrieve input from the TensorDict. Defaults to ["observation"].
        out_keys (str, optional): The key that the actor uses to store output in the TensorDict. Defaults to "logits".
    """

    def __init__(self, module, N, d,  in_keys = ["observation"], out_keys = "logits" ):
        super().__init__(module= module, in_keys = in_keys, out_keys=out_keys)
        self.N =N
        self.d = d

    def forward(self, td) -> TensorDict:
        """
        We can pass each substate to the neural network in parallel by reshaping the input tensor from B x (d * N) to B x N x d
        where B is the batch_size (infered), d is the number of features per node, and N is the number of nodes.
        We then pass this tensor to the module to get the logits


        Args:
            td (TensorDict): The TensorDict that the actor retrieves input from and stores output in.

        Returns:
            TensorDict: The TensorDict with the generated action stored under the specified output keys.
        """
        # Reshape the observation tensor from B x (d * N) to B x N x d before passing to the module
        x = td["observation"].view(-1, self.d, self.N)
        td[self.out_keys[0]] = self.module(x)
        return td


def create_independent_actor_critic(number_nodes: int,
                actor_input_dimension: int,
                actor_in_keys: list[str],
                critic_in_keys: list[str],
                action_spec,
                temperature: float = 1.0,
                actor_depth: int = 2,
                actor_cells: int = 32,
                type: int= 1,
                network_type: str = "STN",
                relu_max: float = 10.0,
                add_zero: bool = True):
    """Function to create an independent actor-critic model.

    Args:
        number_nodes (int): The number of nodes in the input.
        actor_input_dimension (int): The dimension of the input for the actor.
        actor_in_keys (list): The keys to access the actor input data.
        critic_in_keys (list): The keys to access the critic input data.
        action_spec: The specification of the action space.
        temperature (float, optional): The temperature parameter for the softmax function in the actor. Default is 1.0.
        actor_depth (int, optional): The depth of the actor's neural network. Default is 2.
        actor_cells (int, optional): The number of cells in the actor's neural network. Default is 32.
        type (int, optional): The type of actor network to create. Default is 1.
        network_type (str, optional): The type of network to use ("MLP" or "STN"). Default is "MLP".
        relu_max (int, optional): The maximum value for the ReLU activation function. Default is 10.
        add_zero (bool, optional): Whether to add a zero node to the network. Default is True.

    Returns:
        actor_critic (ActorCriticWrapper): The created actor-critic model."""



    if type == 1:
        actor_network = IndependentNodeNetwork(actor_input_dimension, number_nodes,actor_depth, actor_cells, network_type, relu_max)
    elif type == 2:
        raise NotImplementedError
    elif type == 3:
        actor_network = SharedIndependentNodeNetwork(actor_input_dimension, number_nodes,actor_depth, actor_cells, network_type, relu_max, add_zero)

    actor_module = Ind_NN_Actor(actor_network, N = number_nodes, d = actor_input_dimension, in_keys=actor_in_keys, out_keys=["logits"])

    actor_module = ProbabilisticActor(
        actor_module,
        distribution_class=MaskedOneHotCategorical,
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
        elif network_type == "STN":
            module_list = []
            for _ in range(num_nodes):
                module_list.append(STN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max=10))
            self.node_nns = torch.nn.ModuleList(module_list)
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
    def __init__(self, input_dim, num_nodes, depth, cells, network_type = "MLP", relu_max = 1, add_zero = True):
        super(SharedIndependentNodeNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.network_type = network_type
        self.add_zero = add_zero
        # create num_nodes independent neural networks, each with depth and cells
        if network_type == "MLP":
            self.module = MLP(in_features=input_dim,
                                                  out_features=1,
                                                    depth=depth,
                                                    num_cells=cells,
                                                    activation_class=nn.ReLU,
                                                    )

        elif network_type == "STN":
            self.module = STN(input_size = input_dim, output_size = 1, hidden_sizes = [cells]*depth, relu_max = relu_max)


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
        if self.add_zero:
        # Create a B x 1 Tensor of zeros
            zeros = torch.zeros(X.shape[0], 1)
            # Initialize an empty list to store the output of each node's neural network
            output_list = [zeros]
        else:
            output_list = []

        # Pass the node inputs through the shared neural network
        for i in range(self.num_nodes):
            output_list.append(self.module(X[:, :, i]))


        # Stack the outputs to form a tensor of shape (B, N+1)
        output_tensor = torch.stack(output_list, dim=-2)

        return output_tensor.squeeze(-1)















