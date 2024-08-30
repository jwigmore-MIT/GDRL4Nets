import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import matplotlib as mpl

training_set_folder = "SH4_0-5_b"
context_set_file_name = "SH4_context_set_l3_m3_s100.json"
trained_agent_folder = os.path.join("trained_agents", training_set_folder)
test_context_set_path = os.path.join("context_sets", context_set_file_name)
context_set = json.load(open(test_context_set_path, 'rb'))
training_ids = list(range(0, 6))
test_ids = [x for x in range(0,context_set["num_envs"]) if x not in training_ids]

# iterate through each folder in the trained_agents folder and get the results from the respective pickle file
results = {}
for folder in os.listdir(trained_agent_folder):
    # if folder == "PMN":
    #     continue
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
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
bxplot_data = []
unclipped_data = []
clipped_data = []
below_threshold_data = []
above_threshold_data = []
threshold = 3
for agent, agent_results in results.items():
    if agent == "PMN":
        label = "NST"
    else:
        label  = agent
    offset= offset_dict[agent]
    agent_means = [context_results["mean"] for context_id, context_results in agent_results.items()]
    agent_norm_means = [x/y for x, y in zip(agent_means, mw_backlogs)]
    # save the normalized means for each agent
    for i in range(len(agent_norm_means)):
        agent_results[i]["norm_mean"] = agent_norm_means[i]
    clipped_norm_means = np.clip(agent_norm_means, 0, threshold)
    # create a 0-1 vector that indicates if the value is clipped
    clipped = np.array(agent_norm_means) > threshold
    # count the number clipped
    ax.bar(index+ offset, agent_norm_means, bar_width, label=label, edgecolor='black')
    agent_results["training_set_performance"] = np.mean(np.array(agent_norm_means)[training_ids])# how to get agent_norm_means for each ind in training ind
    agent_results["training_set_performance_std"] = np.std(np.array(agent_norm_means)[training_ids])# how to get agent_norm_means for each ind in training ind
    agent_results["clipped_training_set_performance"] = np.mean(np.array(clipped_norm_means)[training_ids])# how to get agent_norm_means for each ind in training ind
    agent_results["training_set_performance_excluding_outliers"] = np.mean(np.array([x for x in np.array(agent_norm_means)[training_ids] if x < 5]))

    agent_results["test_set_performance"] = np.mean(np.array(agent_norm_means)[test_ids])
    agent_results["test_set_performance_std"] = np.std(np.array(agent_norm_means)[test_ids])
    agent_results["clipped_test_set_performance"] = np.mean(np.array(clipped_norm_means)[test_ids])
    # take the mean again but excluding any values greater than 5
    agent_results["test_set_performance_excluding_outliers"] = np.mean(np.array([x for x in np.array(agent_norm_means)[test_ids] if x < 5]))
    agent_results["test_set_performance_excluding_outliers_std"] = np.std([x for x in agent_norm_means if x < 5])
    agent_results["context_set_performance"] = np.mean(agent_norm_means)
    agent_results["context_set_performance_std"] = np.std(agent_norm_means)
    agent_results["clipped_context_set_performance"] = np.mean(clipped_norm_means)
    agent_results["context_set_performance_excluding_outliers"] = np.mean([x for x in agent_norm_means if x < 5])
    agent_results["context_set_performance_excluding_outliers_std"] = np.std([x for x in agent_norm_means if x < 5])
    print('*'*50)
    print(f"{agent} training set performance: {agent_results['training_set_performance']}/{agent_results['training_set_performance_std']}")
    print(f"{agent} clipped training set performance: {agent_results['clipped_training_set_performance']}")
    print(f"{agent} clipped training count {clipped[training_ids].sum()} ")
    print(f"{agent} test set performance: {agent_results['test_set_performance']}/{agent_results['test_set_performance_std']}")
    print(f"{agent} clipped test set performance: {agent_results['clipped_test_set_performance']}")
    print(f"{agent} clipped test count {clipped[test_ids].sum()}")
    print(f"{agent} context set performance: {agent_results['context_set_performance']}/{agent_results['context_set_performance_std']}")
    print(f"{agent} clipped context set performance: {agent_results['clipped_context_set_performance']}")
    print(f"{agent} clipped context count {clipped.sum()}")
    print(f"{agent} training set performance excluding outliers: {agent_results['training_set_performance_excluding_outliers']}/{agent_results['test_set_performance_excluding_outliers_std']}")

    print(f"{agent} test set performance excluding outliers: {agent_results['test_set_performance_excluding_outliers']}/{agent_results['test_set_performance_excluding_outliers_std']}")
    print(f"{agent} context set performance excluding outliers: {agent_results['context_set_performance_excluding_outliers']}/{agent_results['context_set_performance_excluding_outliers_std']}")
    print("\n")

    unclipped_data.append(agent_norm_means)
    clipped_data.append(clipped_norm_means)
    # create below threshold where x is replaced with 10 if x > 10
    below_threshold_data.append(np.clip(agent_norm_means, 0, threshold+1))
    above_threshold_data.append(np.clip(agent_norm_means, threshold, None))

    # below_threshold_data.append(np.array([x for x in agent_norm_means if x < threshold]))
    # above_threshold_data.append(np.array([x for x in agent_norm_means if x >= threshold] ))
    # Create a histogram of the training, test, and context set performance for each agent
    # fig2, ax2 = plt.subplots(3,1,figsize=(30, 10), sharex=True)
    # # ax2.set_title(f"{agent} Performance")
    # # get norm means
    # temp_norm_means = np.array(agent_norm_means).clip(max=5)
    # ax2[0].hist(temp_norm_means[training_ids], bins=10, alpha=0.5, label='Training Set Performance')
    # ax2[1].hist(temp_norm_means[test_ids], bins=20, alpha=0.5, label='Test Set Performance')
    # ax2[2].hist(temp_norm_means, bins=20, alpha=0.5, label='Context Set Performance')
    # ax2[0].set_xlim(0, 5)
    # ax2[1].set_xlim(0, 5)
    # ax2[2].set_xlim(0, 5)
    # ax2[0].set_title(f"Training Set Performance")
    # ax2[1].set_title(f"Testing Set Performance")
    # ax2[2].set_title(f"Context Set Performance")
    # # fig2.legend()
    # fig2.suptitle(f"{label} Performance")
    # fig2.show()

    # ax2.set_title(f"{agent} Performance")
    # get norm means
#     temp_norm_means = np.array(agent_norm_means).clip(max=threshold)
#     bxplot_data.append(agent_norm_means)
#     ax2.hist(temp_norm_means[test_ids], alpha=0.8, label=f"{label}", bins=20)
#     ax2.set_xlim(0, threshold)
#     ax2.set_ylim(0,30)
#     ax2.set_ylabel("Frequency")
#     ax2.set_xlabel("Normalized Mean Backlog")
#     # fig2.legend()
#     # fig2.suptitle(f"{label} Performance")
# # add more space between the xticks and the x-axis
# ax2.tick_params(axis='both', which='major', pad=15)
# fig2.legend()
# fig2.show()


# # box and whisker plot using the unclipped data
# fig1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
# ax1.boxplot(unclipped_data, vert=False, patch_artist=True, showfliers=True,
#             boxprops=dict(facecolor='skyblue', color='black'),
#             whiskerprops=dict(color='black'),
# capprops=dict(color='black'),
# flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none'),
# medianprops=dict(color='red'))
# ax1.set_xlabel("Normalized Mean Backlog", fontsize=24)
# ax1.set_yticks([1, 2], ["MLP", "STN"])
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # overlay histogram of the clipped data
# fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
# ax2.hist(clipped_data, alpha=0.8, label=["MLP", "STN"])
# ax2.set_xlim(0, threshold)
# ax2.set_ylim(0, 30)
# ax2.set_ylabel("Frequency")
# ax2.set_xlabel("Normalized Mean Backlog")
# fig2.legend()
# fig2.show()
#
# # Violen plot of the unclipped data
# fig3, ax3 = plt.subplots(1, 1, figsize=(15, 10))
# ax3.violinplot(unclipped_data, vert=False, showmedians=True, showextrema=True)
# ax3.set_xlabel("Normalized Mean Backlog", fontsize=24)
# ax3.set_yticks([1, 2], ["MLP", "STN"])
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
#
# # scatter plot with jitter
# fig4, ax4 = plt.subplots(1, 1, figsize=(15, 10))
# for i, data in enumerate(unclipped_data):
#     ax4.scatter([i]*len(data), data, alpha=0.5)
# ax4.set_xlabel("Agent", fontsize=24)
# ax4.set_ylabel("Normalized Mean Backlog", fontsize=24)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
#
# # CDF plot
# fig5, ax5 = plt.subplots(1, 1, figsize=(15, 10))
# for data in unclipped_data:
#     data = np.sort(data)
#     y = np.arange(len(data))/len(data)
#     ax5.plot(data, y)
# ax5.set_xlabel("Normalized Mean Backlog", fontsize=24)
# ax5.set_ylabel("CDF", fontsize=24)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# Histogram with log scale

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(15, 10))
#increase font size for the axis labels
mpl.rcParams.update({'font.size': 24, "font.family": "Arial"})

ax6a.hist(below_threshold_data, bins=30, alpha=0.8, label=["MLP", "STN"])
ax6a.set_xlim(0, threshold)
# ax6a.set_ylim(0, 30)
ax6a.set_ylabel("Frequency", fontsize=32, labelpad=10)
# ax6a.set_xlabel("Normalized Mean Backlog")
# ax6a.set_title("Below Threshold")


ax6b.hist(above_threshold_data, bins=30, alpha=0.8)
ax6b.set_xlim(threshold+20)
# ax6b.set_ylim(0, 30)
# ax6b.set_ylabel("Frequency")
# ax6b.set_xlabel("Normalized Mean Backlog")
# ax6b.set_title("Above Threshold")

ax6a.spines.right.set_visible(False)
ax6b.spines.left.set_visible(False)
ax6b.tick_params(left=False)
# hide y-ticks for the right plot
ax6b.set_yticks([])
# remove the space between the two plots
plt.subplots_adjust(wspace=0.05)

d = 1  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=24,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax6a.plot([1, 1], [0, 1], transform=ax6a.transAxes, **kwargs)
ax6b.plot([0, 0], [1, 0], transform=ax6b.transAxes, **kwargs)

#put an xlabel between the two plots
fig6.text(0.5, 0.04, 'Normalized Mean Backlog', ha='center', va='center', fontsize=32)
# place legend inside the main plot
fig6.legend(bbox_to_anchor=(0.8, 0.8), loc='center', fontsize=24)
# fig.tight_layout()
fig6.show()

# Save figure as a pdf
fig6.savefig(os.path.join(trained_agent_folder, f"{training_set_folder}_comparison.pdf"), bbox_inches='tight')

diff = np.array(unclipped_data[0]) - np.array(unclipped_data[1])
# count the number of environments where the difference is greater than 0
print(f"Number of environments where MLP outperforms STN: {np.sum(diff < 0)}")



# # Create a boxplot of the test set performance for each agent
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# ax.boxplot(bxplot_data,  vert=False, patch_artist=True, showfliers=True,
#             boxprops=dict(facecolor='skyblue', color='black'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='black'),
#             flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none'),
#             medianprops=dict(color='red'))
# ax.set_xlabel("Normalized Mean Backlog", fontsize=24)
# ax.set_yticks([1, 2], ["MLP", "STN"])
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
#
# ax.set_xticks(index)  # Set x-ticks at the center of each group of bars
# ax.hlines(1, -3*bar_width, len(mw_backlogs)*1.5, colors='r', linestyles='dashed', label="MaxWeight")  # Add a horizontal line at y=1
# ax.set_xticklabels(context_set["context_dicts"].keys(), rotation=45)  # Rotate x labels for better visibility
# ax.set_xlabel("Environment", fontsize=24)
# ax.set_ylabel("Normalized Mean Backlog", fontsize=24)
# ax.set_ylim(0,1.5)
# ax.legend()
#
#
# plt.show()
# # Save figure as a pdf
# fig.savefig(os.path.join(trained_agent_folder, f"{training_set_folder}_comparison.pdf"), bbox_inches='tight')

# create a histogram of the training, test, and context set performance for each agent

# for agent, agent_results in results.items():
#     fig2, ax2 = plt.subplots(3,1,figsize=(10, 5))
#     # ax2.set_title(f"{agent} Performance")
#     # get norm means
#     fig.suptitle(f"{agent} Performance")
#     norm_means = np.array([r["norm_mean"] for key, r in agent_results.items()])
#     ax2[0].hist(norm_means[training_ids], bins=10, alpha=0.5, label='Training Set Performance')
#     ax2[1].hist(norm_means[test_ids], bins=10, alpha=0.5, label='Test Set Performance')
#     ax2[2].hist(norm_means, bins=10, alpha=0.5, label='Context Set Performance')
#     ax2.legend()
#     # fig2.savefig(os.path.join(trained_agent_folder, f"{agent}_performance_histogram.pdf"), bbox_inches='tight')
#     fig2.show()
