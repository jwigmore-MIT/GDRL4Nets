
from torchrl.envs import ExplorationType
import torch
import torch.nn as nn

from torchrl.data.tensor_specs import CompositeSpec
from torchrl.modules import ReparamGradientStrategy
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper


from utils import MaskedOneHotCategorical
from actors import NN_Actor



def create_cgs_maxweight_actor_critic(input_shape,

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

    actor_critic.mwn_weights = actor_network.get_weights
    return actor_critic


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
