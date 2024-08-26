
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.data.tensor_specs import CompositeSpec



from utils import MaskedOneHotCategorical

from torchrl.modules import ReparamGradientStrategy





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



