import tensordict

from NetworkRunner import create_network_runner
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP, create_sp_bias
import json
from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import DeeperNodeAttentionGNN
from experiments.GNNBiasedBackpressureDevelopment.modules.modules import NormalWrapper
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal
from tensordict.nn import InteractionType
# import optimizer
import torch.optim as optim
import tensordict
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torch_geometric.data import Data, Batch
import networkx as nx
import torch_geometric as pyg
import matplotlib.pyplot as plt
import math
from copy import deepcopy

from experiments.GNNBiasedBackpressureDevelopment.utils import tensors_to_batch, plot_nx_graph
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


NUM_ENVS = 2
FRAMES_PER_BATCH = 8
TOTAL_FRAMES = FRAMES_PER_BATCH*1000
SEED = 13

if __name__ == '__main__':
    # set all random seeds
    torch.manual_seed(SEED)

    file_path = "../envs/grid_5x5.json"
    env_info = json.load(open(file_path, 'r'))
    # env_info['class_info'] = [env_info['class_info'][0]]

    env_info["action_func"] = "bpi"
    env = MultiClassMultiHopBP(**env_info)
    sp_bias, link_sp_dist = create_sp_bias(env)
    network_specs = env.get_rep()

    # Create the Model
    model = DeeperNodeAttentionGNN(
        node_channels = network_specs["X"].shape[-1],
        edge_channels = network_specs["edge_attr"].shape[-1],
        hidden_channels =8,
        num_layers = 4,
        output_channels=1,
        output_activation=None,
        edge_decoder=True
    )
    # TODO: Use Tanh normal wrapper instead to keep bias within -10,10
    norm_module = NormalWrapper(model, scale_bias=0.2)

    actor = ProbabilisticActor(norm_module,
                               in_keys=["loc", "scale"],
                               out_keys=["bias"],
                               distribution_class=IndependentNormal,
                               distribution_kwargs={},
                               return_log_prob=True,
                               default_interaction_type=ExplorationType.DETERMINISTIC
                               )

    create_env_func = lambda: create_network_runner(env=env, max_steps=2000, graph=True)
    baseline_runner = create_network_runner(env=env, max_steps=2000, graph=False)

    collector = MultiSyncDataCollector(
            create_env_fn=[create_env_func]*NUM_ENVS,
            policy=actor,
            frames_per_batch=NUM_ENVS*8,
            total_frames=NUM_ENVS*1000,
            reset_at_each_iter=True,
            )


    optimizer = optim.Adam(actor.parameters(), lr=1e-2)

    writer = SummaryWriter()

    mean_reward = None

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        base_td = baseline_runner.get_run(bias = env.bias.clone())
        print(f"Backpressure Backlog: {-base_td['reward']}")
        baseline_reward = base_td["reward"]

        sp_td = baseline_runner.get_run(bias = sp_bias)
        print(f"SPBP Backlog: {-sp_td['reward']}")
        sp_reward = sp_td["reward"]
    for epoch, samples in enumerate(collector):

        if isinstance(samples["next","data"], list):
            # combine all lists all lists
            # flatten list
            flattened = [item for sublist in samples["next", "data"] for item in sublist]
            batch = Batch.from_data_list(flattened)  # needed to get non-tensor data
        else:
            batch = Batch.from_data_list([data[0] for data in samples["next", "data"]])  # needed to get non-tensor data

        # update the running average mean reward
        if mean_reward is None:
            mean_reward = samples["reward"].mean()
        else:
            mean_reward = 0.9*mean_reward + 0.1*samples["reward"].mean()

        dist_td = TensorDict({"X": batch.X, "edge_index": batch.edge_index, "edge_attr": batch.edge_attr})
        dist = actor.get_dist(dist_td)
        log_prob = dist.log_prob(batch["bias"].unsqueeze(-1))


        loss = -log_prob * (batch["link_rewards"] - mean_reward)
        total_loss = loss.sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # log gradients
        # for name, param in actor.named_parameters():
        #     writer.add_histogram(name, param.grad, epoch)

        log_info = {
            "Loss": total_loss.item(),
            "Mean Backlog": samples["backlog"].mean(),
            "Mean Reward": samples["reward"].mean(),
            "Max Reward": samples["reward"].max(),
            "Min Reward": samples["reward"].min(),
            "Std Reward": samples["reward"].std(),
            # "Mean Location": batch["tanh_loc"].mean().item(),
            "Mean Scale": batch["scale"].mean().item(),
            "Std Location": batch["loc"].std().item(),
            "Std Scale": batch["scale"].std().item(),
            "Mean Loc Logits": batch["logits"][0, :].mean().item(),
            "Mean Scale Logits": batch["logits"][1, :].mean().item(),

        }

        # print everything in log_info
        print(f"------Epoch {epoch} -----")
        for key, value in log_info.items():
            print(f"{key}: {value:.2f}")
            writer.add_scalar(key, value, epoch)

        # Plot the location for a single graph
        if epoch % 5 ==0:

            graph = batch.get_example(0)

            # Get the tanh mean parameter of the distribution i.e. tanh(loc))/scale + range_scaling
            graph_dist =  actor.get_dist(TensorDict({"X": graph.X, "edge_index": graph.edge_index, "edge_attr": graph.edge_attr}))
            graph["dist_loc"] = graph_dist.deterministic_sample.squeeze(-1)

            # set the range of colors for the plot
            erange = (-2, 2)
            vrange = (0, 5)
            subtitle = f"Epoch {epoch}; Mean Backlog: {samples['backlog'].mean().item():.2f}; Sample Backlog: {graph['backlog'].mean().item():.2f}"


            # Plotting Edge Location
            fig, ax = plot_nx_graph(graph, edge_attr=graph.dist_loc, node_attr=graph["Qavg"], subtitle = subtitle, title="Edge Loc", erange=erange, vrange=vrange)
            # fig.show()
            writer.add_figure("Edge Dist Loc", fig, epoch)

