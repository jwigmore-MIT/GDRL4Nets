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
             device="cpu"):

    actor_mlp = MLP(in_features=input_shape[-1],
                    activation_class=torch.nn.ReLU,
                    out_features=output_shape,
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
                output_shape,
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
            self.weights = nn.Parameter(torch.randn(weight_size)) #torch.ones(weight_size)*
        else:
            self.weights = nn.Parameter(weights)
        self.temperature = temperature
    def forward(self, Q, Y):
        # split x into Q and Y
        #Q = x[:, :x.shape[1]//2]
        #Y = x[:, x.shape[1]//2:]
        z = Y * Q * self.weights
        # add a column of 0.1 to z

        #z = z.squeeze() if z.dim() > 1 else z
        if z.dim() == 1:
            z = torch.cat([torch.ones(1), z], dim=0)
        else:
            z = torch.cat([torch.ones(z.shape[0],1), z], dim=1)
        # # make the first element of z to be one and move the rest to the right
        # if z.shape.__len__() > 1:
        #     z = torch.cat([torch.ones(z.shape[0],1), z], dim=1)
        # else:
        #     z = torch.cat([torch.ones(1), z], dim=0)

       # z = torch.cat([torch.ones((z.shape)), z], dim=1)
        #z = torch.cat([torch.zeros((z.shape[0],1)), z], dim=1)
        # normalize z
        #z = z / z.sum(dim=1).unsqueeze(1)
        if self.training:
            A = z
            #A = torch.nn.functional.softmax(z/self.temperature, dim=-1)
        else:
            A = torch.zeros_like(z)
            if z.dim() == 1:
                A[torch.argmax(z)] = 1
            else:
                A[torch.arange(z.shape[0]), torch.argmax(z, dim=-1)] = 1
        #A_one_hot = F.one_hot(A, num_classes=z.shape[1] + 1)
        return A


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




