import pickle
import matplotlib.pyplot as plt
import json
import numpy as np





results = pickle.load(open("n2SH2u_trained_agents/n2SH2u_trained_agents_results.pkl", "rb"))
context_set = json.load(open("n2SH2u_context_set_l1_m3_s30.json", "rb"))
mlp_means = [x for key, x in results["means"].items() if "MLP" in key]
pmn_means = [x for key, x in results["means"].items() if "PMN" in key]
mwn_means = [x for key, x in results["means"].items() if "MWN" in key]
mw_backlogs = [x["lta"] for x in context_set["context_dicts"].values()]

difference = np.array([x-y for x, y in zip(mlp_means, pmn_means)])


# Plot the MaxWeight normalized means for the MLP and PMN agents
# Set larger figure size
fig, ax = plt.subplots(figsize=(30, 15))

# Set larger font size
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})

index = np.arange(len(mlp_means))
bar_width = 0.4

mlp_norm_means = [x/y for x, y in zip(mlp_means, mw_backlogs)]
pmn_norm_means = [x/y for x, y in zip(pmn_means, mw_backlogs)]
mwn_norm_means = [x/y for x, y in zip(mwn_means, mw_backlogs)]

norm_difference = np.array([x-y for x, y in zip(mlp_norm_means, pmn_norm_means)])

# Use different colors for the bars and add edgecolor for better distinction
ax.bar(index-bar_width, mlp_norm_means, bar_width/2, label="MLP", color='blue', edgecolor='black')
ax.bar(index, pmn_norm_means, bar_width/2, label="PMN", color='orange', edgecolor='black')
ax.bar(index+bar_width, mwn_norm_means, bar_width/2, label="MWN", color='green', edgecolor='black')

ax.set_xticks(index)
ax.set_xticklabels(context_set["context_dicts"].keys(), rotation=45)  # Rotate x labels for better visibility
ax.set_xlabel("Environment")
ax.set_ylabel("Normalized Mean Backlog")
ax.set_ylim(0,3)
ax.legend()



# Add gridlines for better readability
# ax.grid(True)

plt.show()

