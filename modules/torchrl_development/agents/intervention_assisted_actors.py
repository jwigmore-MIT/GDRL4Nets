import torch

from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, MaskedOneHotCategorical, ActorCriticWrapper
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleWrapper)


from modules.torchrl_development.maxweight import MaxWeightActor




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

