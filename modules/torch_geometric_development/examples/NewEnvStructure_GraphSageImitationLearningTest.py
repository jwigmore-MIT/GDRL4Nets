from torchrl_development.envs.SingleHopGraph1 import SingleHopGraph
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl_development.actors import MaxWeightActor, MaskedOneHotCategoricalTemp
import torch
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b
from torch_geometric_development.gnn_modules import GNNTensorDictModule, GNN_Critic
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec
from torchrl.modules import ReparamGradientStrategy
from torchrl_development.utils.metrics import compute_lta
from torch_geometric_development.GraphSage import GraphSageModule
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from GCN_creation_test import GCN
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphSAGE, summary
from torchrl_development.envs.env_generators import make_env


def train_GNN_module(module, td, in_keys = ["graph"], num_training_epochs = 1000, lr = 0.001,
                     loss_fn = nn.BCEWithLogitsLoss(), mini_batch_size = 32):

    optimizer = Adam(module.parameters(), lr=lr)
    pbar = tqdm(range(num_training_epochs), desc="Training GNN Module")
    last_n_losses = []
    loss_fn = loss_fn
    data_loader = DataLoader(td, batch_size=mini_batch_size, shuffle=True, collate_fn=lambda x : x)
    for epoch in pbar:
        for mb in data_loader:
            optimizer.zero_grad()
            mb = module(mb)
            loss = loss_fn(mb["logits"].float(), mb["action"].float())
            loss.backward()
            optimizer.step()
        # optimizer.zero_grad()
        # td = module(td)
        # loss = loss_fn(td["logits"].float(), td["action"].float())
        # loss.backward()
        # optimizer.step()
        if epoch % 1 == 0:
            last_n_losses.append(loss.item())
            pbar.set_postfix({f"Epoch": epoch, f"Loss": loss.item()})
            if len(last_n_losses) > 10:
                last_n_losses.pop(0)
                if np.std(last_n_losses) < 1e-6:
                    break
    return module

if __name__ == "__main__":
    env_config_path = "SH2u.json"
    env_para = parse_env_json(env_config_path)
    make_env_keywords = {
        "graph": True,
        "observe_lambda": False,
        "device": 'cpu',
        "terminal_backlog": 1000,
        "inverse_reward": True,
    }


    env = make_env(env_para, **make_env_keywords)
    check_env_specs(env)



    # Create Max Weight Actor to collect Rollout
    actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

    td = env.rollout(max_steps = 10000, policy = actor)

    # Now lets try to train a GNNMaxWeightPolicy using the data collected
    #init_weights = torch.Tensor([[1, -1]])
    init_weights = torch.randn((1, 2))*.1+1
    GNN_model = GraphSAGE(in_channels = env.observation_spec["x"].shape[1] , hidden_channels=32, out_channels=1, num_layers=2, aggr='max')
    print(summary(GNN_model, x = td["x"][0], edge_index = td["edge_index"][0]))
    GNN_Actor = GNNTensorDictModule(GNN_model, in_keys=["x", "edge_index"], out_keys=['logits'])  # create actor
    #td = GNN_Actor(td) # forward pass

    #Create a critic network
    GNN_Critic_Network = GNN_Critic(in_channels = env.observation_spec["x"].shape[1] , hidden_channels=32, out_channels=1)
    GNN_Critic_Module = GNNTensorDictModule(GNN_Critic_Network, in_keys=["x", "edge_index"], out_keys=['value'])






    GNN_Module = train_GNN_module(GNN_Actor, td, in_keys = ["graph"], num_training_epochs = 20, lr = 0.001,)

    # Now lets run the GNN MaxWeight Actor on the environment
    GNN_Actor = ProbabilisticActor(
        GNN_Module,
        distribution_class=MaskedOneHotCategoricalTemp,
        in_keys=["logits", "mask"],
        distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                             "temperature": 1},  # {"temperature": 0.5},
        spec=CompositeSpec(action=env.action_spec),
        return_log_prob=True,
    )

    #set_exploration_type(GNN_Actor, ExplorationType.EPSILON_GREEDY, epsilon=0.1)
    td = env.rollout(max_steps = 10000, policy = GNN_Actor)

    # Compute the LTA
    lta = compute_lta(td["backlog"])

    lta = compute_lta(td["backlog"])
    #
    fig, ax = plt.subplots()
    ax.plot(lta)
    ax.set(xlabel='time', ylabel='LTA',
           title='LTA of MaxWeight GNN Policy')
    ax.grid()
    plt.show()