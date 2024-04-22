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
             temperature = 1.0,
             device="cpu",
                        actor_depth = 2,
                        actor_cells = 32,):
    actor_mlp = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    out_features=output_shape,
                    depth=actor_depth,
                    num_cells=actor_cells,

                    )
    actor_mlp_output = actor_mlp(torch.ones(input_shape))
    actor_module = TensorDictModule(
        module=actor_mlp,
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


    critic_mlp = MLP(in_features=input_shape[-1],
                     activation_class=torch.nn.ReLU,
                     out_features=1,
                     )

    critic_mlp_output = critic_mlp(torch.ones(input_shape))


    # Create Value Module
    value_module = ValueOperator(
        module=critic_mlp,
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


        return torch.relu(z).squeeze()

    def get_weights(self):
        "returns a copy of bias_0 and weights as a single vector"
        return torch.cat([self.bias_0.detach(), self.weights.detach()], dim = -1)


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
            argmax = self.logits.argmax()
            argmax_one_hot = torch.zeros_like(self.logits).squeeze()
            argmax_one_hot[argmax] = 1
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




