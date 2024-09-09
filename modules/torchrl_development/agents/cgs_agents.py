from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
import torch
import torch.nn as nn
import torch.distributions as D
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

def create_mlp_actor_critic(
        input_shape: Tuple[int, ...],
        output_shape,
        in_keys,
        action_spec,
        actor_nn=None,
        critic_nn=None,
        actor_depth=2,
        actor_cells=32,
):
    # If no actor network is provided, create a default MLP
    if actor_nn is None:
        actor_nn = MLP(in_features=input_shape[-1],
                       activation_class=torch.nn.Sigmoid,
                       out_features=output_shape,
                       depth=actor_depth,
                       num_cells=actor_cells,
                       activate_last_layer=True,
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



