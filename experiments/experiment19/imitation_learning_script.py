import torch
import pickle
import os
import sys
from torchrl_development.mdp_actors import MDP_actor, MDP_module
import json
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import numpy as np
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_independent_actor_critic
from tqdm import tqdm
from torchrl_development.SMNN import DeepSetScalableMonotonicNeuralNetwork as DSMNN
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def plot_data(plots, suptitle=""):
    num_plots = len(plots)
    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
    if num_plots == 1:
        axes = [axes]


    for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        if title == "MaxWeightNetwork Weights":
            data = torch.stack(data).squeeze().detach().numpy()
            print("Weights shape: ", data.shape)
            for j in range(data.shape[1]):
                if j == 0:
                    continue
                ax.plot(data[:, j], label=f"W{j}")
            ax.legend()
        else:
            ax.plot(data)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ax.set_xlabel("Minibatch")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()

def eval_agent(agent, env_generator, num_rollouts = 3, rollout_length = 10000):
    results = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for i in range(num_rollouts):
            env = env_generator.sample()
            td = env.rollout(policy=agent, max_steps = rollout_length)
            results[i] = {"td": td, "lta": compute_lta(td["backlog"])}
        results["mean_lta"] = torch.stack([torch.tensor(results[i]["lta"]) for i in range(num_rollouts)]).mean(dim = 0)
        results["std_lta"] = torch.stack([torch.tensor(results[i]["lta"]) for i in range(num_rollouts)]).std(dim = 0)
        env_generator.reseed()
    return results

def supervised_train(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.CrossEntropyLoss(), weight_decay = 1e-5, lr_decay = False, reduce_on_plateau = False,
                to_plot = ["all_losses"], suptitle = "",all_losses = None, all_lrs = None, all_weights = None):
    loss_fn = loss_fn
    # optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)

    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    if all_losses is None:
        all_losses = []
    if all_lrs is None:
        all_lrs = []
    if all_weights is None:
        all_weights = []
    last_n_losses = []
    for epoch in pbar:
        # add learning rate decay
        if not reduce_on_plateau and lr_decay:
            alpha = 1 - (epoch / num_training_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr * alpha

        for mb, td in enumerate(replay_buffer):
            optimizer.zero_grad()
            td["Q"] = td["Q"].float()
            td["Y"] = td["Y"].float()
            td = module(td)
            loss = loss_fn(td['logits'], td["target_action"])
            loss.backward(retain_graph = False)
            optimizer.step()
            all_losses.append(loss.detach().item())
            all_lrs.append(optimizer.param_groups[0]["lr"])
            if module.get_policy_operator().module[0].module.__str__() == "MaxWeightNetwork()":
                actor_weights = module.get_policy_operator().module[0].module.get_weights()
                all_weights.append(actor_weights)
            if reduce_on_plateau:
                scheduler.step(loss)
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.detach().item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
    if len(to_plot) > 0:
        # check if all_weights is empty
        # def plot_data(plots, suptitle=""):
        #     num_plots = len(plots)
        #     fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
        #     if num_plots == 1:
        #         axes = [axes]
        #
        #
        #     if all_weights is not None:
        #         plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        #
        #     for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        #         if title == "MaxWeightNetwork Weights":
        #             data = torch.stack(data).squeeze().detach().numpy()
        #             print("Weights shape: ", data.shape)
        #             for j in range(data.shape[1]):
        #                 if j == 0:
        #                     continue
        #                 ax.plot(data[:, j], label=f"W{j}")
        #             ax.legend()
        #         else:
        #             ax.plot(data)
        #         ax.set_ylabel(ylabel)
        #         ax.set_title(title)
        #
        #     if num_plots == 2:
        #         ax[1].set_xlabel("Minibatch")
        #
        #     fig.suptitle(suptitle)
        #     fig.tight_layout()
        #     fig.show()

        plots = []
        if "all_losses" in to_plot:
            plots.append((all_losses, "Loss", "Training Loss"))
        if "all_lrs" in to_plot:
            plots.append((all_lrs, "Learning Rate", "Learning Rate Schedule"))
        if "all_weights" in to_plot:
            plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        plot_data(plots, suptitle=suptitle)
    return all_losses, all_lrs, all_weights
        # stop training if loss converges






PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

test_context_set_path = "SH2u2_context_set_20_07091947.json"
context_set = json.load(open(test_context_set_path, 'rb'))
mlp_cfg = load_config(os.path.join(SCRIPT_PATH, 'MLP_PPO_settings.yaml'))


make_env_parameters = {"graph": getattr(mlp_cfg.training_env, "graph", False),
                                    "observe_lambda": getattr(mlp_cfg.training_env, "observe_lambda", True),
                                    "observe_mu": getattr(mlp_cfg.training_env, "observe_mu", True),
                                    "terminal_backlog": None,
                                    "observation_keys": getattr(mlp_cfg.training_env, "observation_keys", ["Q", "Y"]),
                                    "observation_keys_scale": getattr(mlp_cfg.training_env, "observation_keys_scale", None),
                                    "negative_keys": getattr(mlp_cfg.training_env, "negative_keys", ["Y"]),
                                    "symlog_obs": getattr(mlp_cfg.training_env, "symlog_obs", False),
                                    "symlog_reward": getattr(mlp_cfg.training_env, "symlog_reward", False),
                                    "inverse_reward": getattr(mlp_cfg.training_env, "inverse_reward", False),
                                    "cost_based": getattr(mlp_cfg.training_env, "cost_based", True),
                                    "reward_scale": getattr(mlp_cfg.training_env, "reward_scale", 1.0),
                                    "stat_window_size": getattr(mlp_cfg.training_env, "stat_window_size", 5000),}


env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=0)
base_env = env_generator.sample(0)
input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n
action_spec = base_env.action_spec
N = int(base_env.base_env.N)
D = int(input_shape[0]/N)


# Create MLP Agent
mlp_agent = create_actor_critic(
                input_shape,
                output_shape,
                in_keys=["observation"],
                action_spec=action_spec,
                temperature=mlp_cfg.agent.temperature,
                actor_depth=mlp_cfg.agent.hidden_sizes.__len__(),
                actor_cells=mlp_cfg.agent.hidden_sizes[-1],
            )
mlp_agent.load_state_dict(torch.load("MLP_d_SH2u2_Context_0_agent.pt"))


nn = DSMNN(N,
                    D,
                    latent_dim = 64,
                    deepset_width=16,
                    deepset_out_dim=16,
                    exp_unit_size= (64, 64, 64),
                    relu_unit_size = (64, 64, 64),
                    conf_unit_size = (64, 64, 64),
                    )
DSMNN_agent = create_actor_critic(
    input_shape,
    output_shape,
    actor_nn=nn,
    in_keys=["observation"],
    action_spec=action_spec,
    temperature=5,

)

# Collect data from the MLP agent for environment 0
results = eval_agent(mlp_agent, env_generator, num_rollouts=3, rollout_length=10000)

td = torch.cat([results[i]["td"] for i in range(3)])
td["target_action"] = td["action"].int().argmax(dim = 1).long()
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
replay_buffer.extend(td)

# Train the DSMNN agent on the data from the MLP agent
all_losses, all_lrs, all_weights = supervised_train(DSMNN_agent, replay_buffer, num_training_epochs=500, lr=0.0001, lr_decay=True, reduce_on_plateau=False, to_plot=["all_losses"], suptitle="DSMNN Agent Training")
