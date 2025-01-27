from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
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
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torch_geometric.data import Data, Batch
from NetworkRunner import create_network_runner
from experiments.GNNBiasedBackpressureDevelopment.utils import plot_nx_graph
from tensordict import TensorDict
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
# only keep first class
env_info['class_info'] = [env_info['class_info'][0]]
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
network_specs = env.get_rep()

# Create the Model
model = DeeperNodeAttentionGNN(
    node_channels = network_specs["X"].shape[-1],
    edge_channels = network_specs["edge_attr"].shape[-1],
    hidden_channels =8,
    num_layers = 4,
    output_channels=2,
    output_activation=None,
    edge_decoder=True
)

norm_module = NormalWrapper(model, scale_bias=0.25)
actor = ProbabilisticActor(norm_module,
                            in_keys = ["loc", "scale"],
                            out_keys = ["bias"],
                            distribution_class=IndependentNormal,
                           distribution_kwargs={"tanh_loc": True},
                            return_log_prob=True,
                            default_interaction_type = InteractionType.RANDOM
                            )
if True:
    # single env training
    create_env_func = lambda: create_network_runner(env=env, max_steps=2000, graph=True, link_rewards="shared")
    baseline_runner = create_network_runner(env=env, max_steps=2000, graph=False)
    collector = SyncDataCollector(
        create_env_fn=create_env_func,
        policy=actor,
        frames_per_batch=1,
        total_frames=100,
        reset_at_each_iter=True,
    )
    optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    writer = SummaryWriter()

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        base_td = baseline_runner.get_run(bias = env.bias.clone())
        print(f"Baseline Reward: {base_td['reward']}")
        baseline_reward = base_td["reward"]

    for epoch, sample in enumerate(collector):
        batch = sample["next","data"]
        dist_td = TensorDict({"X": batch.X, "edge_index": batch.edge_index, "edge_attr": batch.edge_attr})
        dist = actor.get_dist(dist_td)
        log_prob = dist.log_prob(batch["bias"].unsqueeze(-1))
        loss = -log_prob * batch["link_rewards"]
        mean_loss = loss.mean()
        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

        log_info = {
            "Loss": mean_loss.item(),
            "Mean Reward": sample["reward"].mean(),
            "Max Reward": sample["reward"].max(),
            "Min Reward": sample["reward"].min(),
            "Std Reward": sample["reward"].std(),
            "Mean Location": batch["loc"].mean().item(),
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
        if epoch % 1 == 0:
            # graph = batch.get_example(0)
            graph = batch
            graph.loc = graph.loc.round(decimals=2)
            erange = (-3,3)
            vrange = (0,5)

            # nx.set_edge_attributes(nx_graph, graph.edge_attr)
            fig, ax = plot_nx_graph(graph, edge_attr=graph.bias, node_attr=graph["Qavg"], epoch=epoch,
                                    reward=sample["reward"].mean(), title = "Edge Bias", erange = erange, vrange = vrange)
            # fig.show()
            writer.add_figure("Edge Loc", fig, epoch)

            fig2, ax2 = plot_nx_graph(graph, edge_attr=graph["link_rewards"], node_attr=graph["Qavg"], epoch=epoch,
                                    reward=sample["reward"].mean(), title="Link Rewards", erange = erange, vrange = vrange)
            # fig2.show()
            writer.add_figure("Link Rewards", fig2, epoch)

            fig3, a3 = plot_nx_graph(graph, edge_attr=loss.detach(), node_attr=graph["Qavg"], epoch=epoch,
                                    reward=sample["reward"].mean(), title="Link Loss", erange = erange, vrange = vrange)
            # fig3.show()
            writer.add_figure("Link Loss", fig3, epoch)
            # plt.show()


else:
    runner =  create_network_runner(env = env, max_steps = 2000, graph= True, link_rewards = "Qdiff")

    result = runner._step()


    fig,ax = plot_nx_graph(result["data"], edge_attr= result["data"]["reward"].squeeze(),
                        node_attr = result["data"]["Q"].squeeze(),
                        K = 4, epoch = 1, reward = 1)

    fig.show()