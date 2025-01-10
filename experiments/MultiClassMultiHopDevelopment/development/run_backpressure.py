import random
import string
import json
import os
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import matplotlib.pyplot as plt
import torch
from modules.torchrl_development.utils.metrics import compute_lta
from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
from torchrl.envs.transforms import Transform

def test_backpressure(env_info = None, net = None, rollout_length = 10000, runs = 3):
    # Create environment
    if env_info is not None:
        net = MultiClassMultiHop(**env_info)
        # net = Transform(net, MCMHPygLinkGraphTransform())
    elif net is None:
        raise ValueError("Either env_info or net must be provided")
    # Create backpressure actor
    bp_actor = BackpressureActor(net=net)
    # generate 3 different rollouts
    fig, ax = plt.subplots()
    tds =[]
    ltas = []
    for i in range(runs):
        td = net.rollout(max_steps=rollout_length, policy=bp_actor)
        ltas.append(compute_lta(td["Q"].sum((1,2))))
        tds.append(td)
        ax.plot(ltas[-1], label=f"Rollout {i}")
    # plot the mean ltas of the new network
    ltas = torch.stack(ltas)
    mean_lta = ltas.mean(0)
    ax.plot(mean_lta, label="Mean LTA", color="black")
    fig.show()
    return mean_lta[-1], tds

file_path = "../envs/grid_5x5.json"
env_info = json.load(open(file_path, 'r'))

mean_lta, tds = test_backpressure(env_info = env_info, rollout_length=10000, runs=3)

#get all ltas
ltas = []
for td in tds:
    ltas.append(compute_lta(td["Q"].sum((1,2))))

mean_ltas = torch.stack(ltas).mean(dim=0)

# plot percent abs(mean_ltas[-1] - mean_ltas[i])/mean_ltas[-1] for all i
fig, ax = plt.subplots()
norm_lta = (mean_ltas - mean_ltas[-1])/mean_ltas[-1]
ax.plot(norm_lta, label="Normalized LTA")
fig.show()
