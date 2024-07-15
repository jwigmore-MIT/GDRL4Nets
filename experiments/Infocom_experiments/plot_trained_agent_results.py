import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import matplotlib as mpl

training_set_folder = "SH4_65_5"
context_set_file_name = "SH4_context_set_l3_m3_s100.json"
trained_agent_folder = os.path.join("trained_agents", training_set_folder)
test_context_set_path = os.path.join("context_sets", context_set_file_name)
context_set = json.load(open(test_context_set_path, 'rb'))
training_ids = list(range(0, 6))
test_ids = [x for x in range(0,context_set["num_envs"]) if x not in training_ids]

# iterate through each folder in the trained_agents folder and get the results from the respective pickle file
results = {}
for folder in os.listdir(trained_agent_folder):
    if os.path.isdir(os.path.join(trained_agent_folder, folder)):
        results[folder] = pickle.load(open(os.path.join(trained_agent_folder, folder, f"{folder}_results.pkl"), "rb"))

mw_backlogs = [x["lta"] for x in context_set["context_dicts"].values()]

# Plot the MaxWeight Normalized Backlog for each agent in results
fig, ax = plt.subplots(figsize=(30, 15))
# change fonts for the axis labels and for the legend:
mpl.rcParams.update({'font.size': 22, "font.family": "Arial"})


# mpl.rcParams.update({'font.size': 36, "font.serif": "Times New Roman"})

index = np.arange(len(mw_backlogs)) * 1.5  # Multiply index with a factor larger than 3
bar_width = 0.4
offset_dict  = {"MLP": -bar_width, "PMN": bar_width, "MWN": 0}
for agent, agent_results in results.items():
    if agent == "PMN":
        label = "STN"
    else:
        label  = agent
    offset= offset_dict[agent]
    agent_means = [context_results["mean"] for context_id, context_results in agent_results.items()]
    agent_norm_means = [x/y for x, y in zip(agent_means, mw_backlogs)]
    # save the normalized means for each agent
    for i in range(len(agent_norm_means)):
        agent_results[i]["norm_mean"] = agent_norm_means[i]
    ax.bar(index+ offset, agent_norm_means, bar_width, label=label, edgecolor='black')
    agent_results["training_set_performance"] = np.mean(np.array(agent_norm_means)[training_ids])# how to get agent_norm_means for each ind in training ind
    agent_results["test_set_performance"] = np.mean(np.array(agent_norm_means)[test_ids])
    # take the mean again but excluding any values greater than 5
    agent_results["test_set_performance_excluding_outliers"] = np.mean([x for x in agent_norm_means if x < 5])
    agent_results["context_set_performance"] = np.mean(agent_norm_means)
    agent_results["context_set_performance_excluding_outliers"] = np.mean([x for x in agent_norm_means if x < 5])
    print('*'*50)
    print(f"{agent} training set performance: {agent_results['training_set_performance']}")
    print(f"{agent} test set performance: {agent_results['test_set_performance']}")
    print(f"{agent} context set performance: {agent_results['context_set_performance']}")
    print(f"{agent} test set performance excluding outliers: {agent_results['test_set_performance_excluding_outliers']}")
    print(f"{agent} context set performance excluding outliers: {agent_results['context_set_performance_excluding_outliers']}")
    print("\n")

ax.set_xticks(index)  # Set x-ticks at the center of each group of bars
ax.hlines(1, -3*bar_width, len(mw_backlogs)*1.5, colors='r', linestyles='dashed', label="MaxWeight")  # Add a horizontal line at y=1
ax.set_xticklabels(context_set["context_dicts"].keys(), rotation=45)  # Rotate x labels for better visibility
ax.set_xlabel("Environment", fontsize=24)
ax.set_ylabel("Normalized Mean Backlog", fontsize=24)
ax.set_ylim(0,1.5)
ax.legend()

plt.show()
# Save figure as a pdf
fig.savefig(os.path.join(trained_agent_folder, f"{training_set_folder}_comparison.pdf"), bbox_inches='tight')

