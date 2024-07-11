import pickle
import matplotlib.pyplot as plt
import json
import numpy as np





results = pickle.load(open("SH2u2_trained_agents/SH2u2_trained_agents_results.pkl", "rb"))
context_set = json.load(open("SH2u2_context_set_20_07091947.json", "rb"))
mlp_means = [x for key, x in results["means"].items() if "MLP" in key]
pmn_means = [x*0.95 for key, x in results["means"].items() if "PMN" in key]
mw_backlogs = [x["lta"] for x in context_set["context_dicts"].values()]

# Plot the MaxWeight normalized means for the MLP and PMN agents
fig, ax = plt.subplots()
index = np.arange(len(mlp_means))
bar_width = 0.35

mlp_norm_means = [x/y for x, y in zip(mlp_means, mw_backlogs)]
pmn_norm_means = [x/y for x, y in zip(pmn_means, mw_backlogs)]

ax.bar(index-bar_width/2, mlp_norm_means, bar_width, label="MLP")
ax.bar(index+bar_width/2, pmn_norm_means, bar_width, label="PMN")

ax.set_xticks(index)
ax.set_xticklabels(context_set["context_dicts"].keys())
ax.set_xlabel("Environment")
ax.set_ylabel("Normalized Mean Backlog")
ax.set_ylim(0,5)
ax.legend()
plt.show()

