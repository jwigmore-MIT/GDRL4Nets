# %%
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from torchrl_development.actors import MDP_module, MDP_actor
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import pickle
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic, create_gnn_maxweight_actor_critic
from torchrl.envs.utils import ExplorationType, set_exploration_type


# %%

def sec_order_train(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr = 1):
    """
    Second order optimization for training a module with replay buffer
    :param module:
    :param replay_buffer:
    :param in_keys:
    :param num_training_epochs:
    :param lr:
    :return:
    """

    loss_values = []
    def closure(td, loss_fn):
        optimizer.zero_grad()
        td["Q"] = td["Q"].float()
        td["Y"] = td["Y"].float()
        td = module(td)
        loss = loss_fn(td['logits'], td["target_action"])
        loss.backward()
        loss_values.append(loss.item())
        return loss



    optimizer = optim.LBFGS(module.parameters(), lr=0.01)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    for epoch in pbar:
        for mb, td in enumerate(replay_buffer):
            optimizer.step(lambda: closure(td, nn.CrossEntropyLoss()))
            if mb % 10 == 0:
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb, "Loss": loss_values[-1]})
    return loss_values



def supervised_train(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.CrossEntropyLoss(), weight_decay = 1e-5, reduce_on_plateau = False):
    loss_fn = loss_fn
    # optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-2, verbose=True)

    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    last_n_losses = []
    for epoch in pbar:
        # add learning rate decay
        if not reduce_on_plateau:
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
            if reduce_on_plateau:
                scheduler.step(loss)
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.detach().item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
        # stop training if loss converges

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

# %%

rollout_length = 10000
q_max = 50
num_rollouts = 3
env_generator_seed = 5031997

# Configure training params
num_training_epochs = 300
lr = 0.0001
pickle_string = f"IL_SH1b_nr{num_rollouts}_rl{rollout_length}"

# results storage
results = {}

# MLP Actor Parameters
actor_params = {
  "actor_depth": 2,
  "actor_cells": 64,
}

# Configure Environment Generator
base_env_params = parse_env_json("SH1B.json")

make_env_parameters = {"observe_lambda": False,
                       "device": "cpu",
                       "terminal_backlog": 5000,
                       "inverse_reward": True,
                       "stat_window_size": 100000,
                       "terminate_on_convergence": False,
                       "convergence_threshold": 0.1,
                       "terminate_on_lta_threshold": False}

env_generator = EnvGenerator(base_env_params, make_env_parameters, env_generator_seed)
base_env = env_generator.sample()
input_shape = int(base_env.base_env.N*2)
output_shape = int(base_env.base_env.N+1)



# %% Create MDP Module
mdp = SingleHopMDP(base_env, name = "SH1B", q_max = q_max)
mdp.load_tx_matrix(f"tx_matrices/SH1Bb_qmax50_discount0.99_computed_tx_matrix.pkl")
mdp.load_VI(f"saved_mdps/SH1Bb_qmax50_discount0.99_VI_dict.p")
mdp_actor = MDP_actor(MDP_module(mdp))
# %% Generate Trajectories from mdp_actor
results["MDP"] = eval_agent(mdp_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)


fig, ax = plt.subplots(1,1)
ax.plot(results["MDP"]["mean_lta"], label = "MDP Policy")
ax.fill_between(range(len(results["MDP"]["mean_lta"])), results["MDP"]["mean_lta"] - results["MDP"]["std_lta"], results["MDP"]["mean_lta"] +results["MDP"]["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

# %% Create ReplayBuffer (Dataset)
td = torch.cat([results["MDP"][i]["td"] for i in range(num_rollouts)])
td["target_action"] = td["action"].int().argmax(dim = 1).long()
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
replay_buffer.extend(td)

# %% Create MLP Module
mlp_agent = create_actor_critic(
        [input_shape],
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        actor_depth=actor_params["actor_depth"],
        actor_cells=actor_params["actor_cells"],
    )

# %% Create MaxWeightNetwork Agent
mwn_agent = create_maxweight_actor_critic(input_shape=[input_shape], output_shape=output_shape,
                                                action_spec=base_env.action_spec, in_keys=["Q", "Y"],
                                                temperature=10
                                                )
# %% Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
results["MaxWeight"] = eval_agent(mw_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# %% Train MLP Agent
supervised_train(mlp_agent,
                 replay_buffer,
                 num_training_epochs=num_training_epochs,
                 lr=lr,
                 loss_fn=nn.CrossEntropyLoss(),
                 weight_decay=0)
results["MLP"] = eval_agent(mlp_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)


# %% Train MaxWeightNetwork Agent
mwn_lr = 0.001
supervised_train(mwn_agent,
                 replay_buffer,
                 num_training_epochs=num_training_epochs,
                 lr=lr,
                 loss_fn=nn.CrossEntropyLoss(),
                 reduce_on_plateau = False,
                 weight_decay=0)
# %% Evaluate MWN Agent
results["MWN"] = eval_agent(mwn_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)


#%% Second Order Training of MaxWeightNetwork Agent
mwn_agent2 = create_maxweight_actor_critic(input_shape=[input_shape], output_shape=output_shape,
                                                action_spec=base_env.action_spec, in_keys=["Q", "Y"],
                                                temperature=10
                                                )
losses = sec_order_train(mwn_agent2, replay_buffer, num_training_epochs=num_training_epochs)

#%% Evaluate MWN Agent2
results["MWN2"] = eval_agent(mwn_agent2, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)

# %% Plot the results
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy")
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

# %% Plot Specific Policies
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    if agent_name not in ["MDP", "MaxWeight"]:
        continue
    if agent_name == "MDP":
        agent_name = "VI"
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy")
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()



