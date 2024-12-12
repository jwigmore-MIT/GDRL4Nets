import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Batch



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
        # if title == "Training Loss":
        #     ax.set_ylim(0, 1)

    ax.set_xlabel("Minibatch")


    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()


def supervised_train(module, replay_buffer, num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.BCELoss, weight_decay = 0.0, lr_decay = False, reduce_on_plateau = False,
                to_plot = ["all_losses"], suptitle = "",all_losses = None, all_lrs = None, all_weights = None):
    loss_fn = loss_fn(reduction = "none")
    # loss_fn = nn.MSELoss(reduction = "none")
    # optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)

    pbar = tqdm.tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
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
            if isinstance(td, Batch): # want to pass td directly to the GNN actor module
                probs, logits = module[0](td)
                # td.batch is a batch_size*num_nodes tensor that specifies which graph each node belongs to
                # i want to regroup all probs by their respective graph
                probs = probs.reshape(td.num_graphs, -1)
                target_action = td.target_action.reshape(td.num_graphs, -1)
                all_loss = loss_fn(probs, target_action.float())

            else:
                td = module(td)
                all_loss = loss_fn(td['probs'], td["target_action"].float())
            all_loss = all_loss*td["mask"]
            loss = all_loss.mean()



            loss.backward(retain_graph = False)

            optimizer.step()
            # if epoch > 9:
            #     # Get the max in all_loss
            #     if loss > all_losses[-1]*1.1:
            #         values, indices = torch.topk(all_loss.sum(dim=1), 5)
            #         # create a table from values, td["q"][indices], td["target_action"][indices], td["probs"][indices]
            #         df = pd.DataFrame({"Values": values.detach()})
            #         df["q"] = [q.detach().numpy() for q in td["q"][indices]]
            #         df["target_action"] = [a.detach().numpy() for a in td["target_action"][indices]]
            #         df["probs"] = [p.detach().numpy().round(decimals = 4) for p in td["probs"][indices]]
            #         df


            all_losses.append(loss.detach().item())
            all_lrs.append(optimizer.param_groups[0]["lr"])
            policy_operator = module.get_policy_operator() if hasattr(module, "get_policy_operator") else None
            if policy_operator:
                max_weight_network = policy_operator.module[0].module
                if str(max_weight_network) == "MaxWeightNetwork()":
                    actor_weights = max_weight_network.get_weights()
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


from torchrl.objectives.value.functional import generalized_advantage_estimate as gae
from torchrl.objectives.value import GAE
def supervised_train_w_critic(module, replay_buffer, num_training_epochs=5, lr=0.0001,
                  weight_decay = 0.0, lr_decay = False, reduce_on_plateau = False,
                to_plot = ["all_policy_losses", "all_critic_losses", "lr"], suptitle = "",all_policy_losses = None, all_critic_losses = None, all_lrs = None, all_weights = None):

    policy_loss_fn = nn.BCELoss(reduction = "none")
    critic_loss_fn = nn.MSELoss(reduction = "none")
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    gae_estimator = GAE(gamma = 0.99, lmbda = 0.95, value_network = module.get_value_operator(), vectorized=False)

    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)

    pbar = tqdm.tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    if all_policy_losses is None:
        all_policy_losses = []
    if all_critic_losses is None:
        all_critic_losses = []
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
            # if isinstance(td, Batch): # want to pass td directly to the GNN actor module
            #     probs, logits = module[0](td)
            #     # td.batch is a batch_size*num_nodes tensor that specifies which graph each node belongs to
            #     # i want to regroup all probs by their respective graph
            #     probs = probs.reshape(td.num_graphs, -1)
            #     target_action = td.target_action.reshape(td.num_graphs, -1)
            #     all_loss = loss_fn(probs, target_action.float())
            #
            # else:
            td = module(td)
            td["next"] = module(td["next"])

            adv, td["target_value"] = gae(0.99, 0.95, td["state_value"], td["next", "state_value"], td["next","reward"], td["done"], td["terminated"])
            all_policy_loss = policy_loss_fn(td['probs'], td["target_action"].float())
            all_critic_loss = critic_loss_fn(td['state_value'], td["target_value"].float())

            loss = all_policy_loss.mean() + all_critic_loss.mean()



            loss.backward(retain_graph = False)

            optimizer.step()
            # if epoch > 9:
            #     # Get the max in all_loss
            #     if loss > all_losses[-1]*1.1:
            #         values, indices = torch.topk(all_loss.sum(dim=1), 5)
            #         # create a table from values, td["q"][indices], td["target_action"][indices], td["probs"][indices]
            #         df = pd.DataFrame({"Values": values.detach()})
            #         df["q"] = [q.detach().numpy() for q in td["q"][indices]]
            #         df["target_action"] = [a.detach().numpy() for a in td["target_action"][indices]]
            #         df["probs"] = [p.detach().numpy().round(decimals = 4) for p in td["probs"][indices]]
            #         df


            all_policy_losses.append(all_policy_loss.detach().mean().item())
            all_critic_losses.append(all_critic_loss.detach().mean().item())
            all_lrs.append(optimizer.param_groups[0]["lr"])
            policy_operator = module.get_policy_operator() if hasattr(module, "get_policy_operator") else None
            if policy_operator:
                max_weight_network = policy_operator.module[0].module
                if str(max_weight_network) == "MaxWeightNetwork()":
                    actor_weights = max_weight_network.get_weights()
                    all_weights.append(actor_weights)
            if reduce_on_plateau:
                scheduler.step(loss)
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Policy Loss": all_policy_loss.detach().mean().item(), "Critic Loss": all_critic_loss.detach().mean().item()})
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
        if "all_policy_losses" in to_plot:
            plots.append((all_policy_losses, "Loss", "Policy Training Loss"))
        if "all_critic_losses" in to_plot:
            plots.append((all_critic_losses, "Loss", "Critic Training Loss"))
        if "all_lrs" in to_plot:
            plots.append((all_lrs, "Learning Rate", "Learning Rate Schedule"))
        if "all_weights" in to_plot:
            plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        plot_data(plots, suptitle=suptitle)
    return all_policy_losses, all_critic_losses, all_lrs, all_weights
        # stop training if loss converges

def eval_agent(agent, env_generator, max_steps, rollouts = 1, cat = True):
    agent.eval()
    tds = []
    for r in range(rollouts):
        env = env_generator.sample()
        with torch.no_grad() and set_exploration_type(ExplorationType.DETERMINISTIC):
            td = env.rollout(max_steps = max_steps, policy = agent)
            td["rollout_id"] =torch.ones_like(td)*r
            tds.append(td)
    if cat:
        tds = TensorDict.cat(tds)
    agent.train()
    return tds

def create_action_map(tensors: list, keys: list):
    ### Create a pandas dataframe that maps each observation to an action
    import pandas as pd
    df = pd.DataFrame()
    for tensor, key in zip(tensors, keys):
        temp_df = pd.DataFrame(tensor.numpy(), columns = [f"{key}{i}" for i in range(tensor.shape[1])])
        df = pd.concat([df, temp_df], axis = 1)
    # df = pd.DataFrame(observations.numpy(), columns = [f"q{i}" for i in range(observations.shape[1])])
    # df2 = pd.DataFrame(actions.numpy(), columns = [f"a{i}" for i in range(actions.shape[1])])
    # df = pd.concat([df, df2], axis = 1)
    # remove non-unique rows
    df = df.drop_duplicates()
    # sort by q values
    df = df.sort_values(by = [f"q{i}" for i in range(tensors[0].shape[1])])
    return df

def create_training_dataset(env, agent, q_max = 10):
    # get the number of nodes
    num_nodes = env.observation_spec["observation"].shape[0]
    # enumerate all possible queue states from 0 to q_max for the set of nodes
    # i.e for 3 nodes (0,0,0), (0,0,1)..., (0,0,q_max), (0,1,0), (0,1,1), ..., (q_max, q_max, q_max)
    queue_states = torch.stack(torch.meshgrid([torch.arange(q_max+1) for i in range(num_nodes)])).view(num_nodes, -1).T
    # Get the agents action for each possible queue state
    tds = []
    for i, q in enumerate(queue_states):
        td = env.reset()
        td["q"] = q
        td = env.transform(td)
        td = agent(td)
        tds.append(td)
    tds = TensorDict.stack(tds)
    return tds


