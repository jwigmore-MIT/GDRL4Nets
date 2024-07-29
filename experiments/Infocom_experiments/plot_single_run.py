import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
# Load the data
file = "trained_agents/SH4_0-5_b/MLP/MLP_results_full.pkl"

data = pkl.load(open(file, "rb"))

td1 = data[0]["ltas"].mean(axis=0)/2.1
td2 = data[1]["ltas"].mean(axis=0)/40

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
t_lim = 5000
ax.plot(td1[:t_lim], label="Training Environment")
ax.plot(td2[:t_lim], label="Test Environment")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized Backlog")
# ax.set_xlim(0, 5000)
# ax.set_ylim(0, 100)
ax.legend()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(td1, label="Training Environment")
ax.plot(td2, label="Test Environment")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized Backlog")
# ax.set_xlim(0, 5000)
ax.legend()
plt.show()

