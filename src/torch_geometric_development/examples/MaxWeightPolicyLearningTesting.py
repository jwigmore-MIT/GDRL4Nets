from torchrl_development.envs.SingleHopGraph import SingleHopGraph
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl_development.actors import MaxWeightActor, MaskedOneHotCategoricalTemp
import torch
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b
from torch_geometric_development.MaxWeightGNN import GNNTensorDictModule
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec
from torchrl.modules import ReparamGradientStrategy
from torchrl_development.utils.metrics import compute_lta
from torch_geometric_development.GraphSage import GraphSageModule
import matplotlib.pyplot as plt




if __name__ == "__main__":
    env_config_path = "SH2u.json"
    env_para = parse_env_json(env_config_path)
    env = SingleHopGraph(env_para)
    check_env_specs(env)

    # Create Max Weight Actor to collect Rollout
    actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

    td = env.rollout(max_steps = 10, policy = actor)

    # Now lets try to train a GNNMaxWeightPolicy using the data collected

    GNN_Actor = GNNTensorDictModule(in_keys = ["graph"], out_keys = ['logits']) # create actor
    #td = GNN_Actor(td) # forward pass

    GNN_Actor = ProbabilisticActor(
        GNN_Actor,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": 1},  # {"temperature": 0.5},
        spec=CompositeSpec(action=env.action_spec),
        return_log_prob=True,
    )


    #set_exploration_type(ExplorationType.MODE)
    td = env.rollout(max_steps = 10000, policy = GNN_Actor)
    #
    lta = compute_lta(td["backlog"])
    #
    fig, ax = plt.subplots()
    ax.plot(lta)
    ax.set(xlabel='time', ylabel='LTA',
           title='LTA of MaxWeight GNN Policy')
    ax.grid()
    plt.show()