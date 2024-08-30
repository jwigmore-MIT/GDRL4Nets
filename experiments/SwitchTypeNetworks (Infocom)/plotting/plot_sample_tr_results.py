import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({'font.size': 20, "font.family": "Arial"})
# Plot directory
plt_dir = 'training_data/MP_Single_Training_plots'

# Load the data
csv_data = pd.read_csv("training_data/MP_single_env_training_data.csv")
# Context set: MP5
context_set = "MP5"

# Time cap to plot until
time_cap = 500_000

# Start by getting the step column
df = csv_data["trainer/step"]

# rename step column to step
df = df.rename("step")

# get all columns in csv_data that contain train/avg_mean_normalized_backlog
for column in csv_data.columns:
    if "train/avg_mean_normalized_backlog" in column:
        if 'train/avg_mean_normalized_backlog_' in column:
            continue
        df = pd.concat([df, csv_data[column]], axis=1)

"""
Rename all column headers. Want to extract the 
"""
for column in df.columns:
    if 'step' in column:
        continue
    splitted = column.split('_')
    arch = splitted[0]
    env_id = None
    for i in range(1, len(splitted)):
        if context_set not in splitted[i]:
            continue
        env_id = f"{splitted[i+1]}"
    df = df.rename(columns={column: f"{env_id}_{arch}"})

# Sort the columns by the environment id except for the first column
df = df.reindex(sorted(df.columns), axis=1)


# Only plot up to time time_cap
df = df[df['step'] <= time_cap]

"""
Archs that share the same environment id will share the same color, but different line styles
"""

# get the list of standard colors for matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# get list of env_ids
env_ids = []
for column in df.columns:
    if 'step' in column:
        continue
    env_id, arch = column.split('_')
    if env_id not in env_ids:
        env_ids.append(env_id)



for id in env_ids:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for column in df.columns:
        if 'step' in column:
            continue
        env_id, arch = column.split('_')
        if env_id != id:
            continue
        color = colors[env_ids.index(env_id)]
        linestyle = '--' if 'MLP' in arch else '-'
        label = arch if 'MLP' in arch else f"STN"
        ax.plot(df['step'], np.log(df[column]), label=label, linestyle=linestyle, color=color)
        # if env_ids.index(env_id)+1 == 5:
        ax.set_xlabel("Training Step")
        # change the x-ticks to be in scientific notation
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # if env_ids.index(env_id)+1 == 1:
        ax.set_ylabel("Log Norm. Moving Average Cost")
        # ax.set_title(f"Environment Id: {env_ids.index(env_id)+1}")
        ax.legend()
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{plt_dir}/env_{env_ids.index(env_id)+1}_log_cost_tr.png")
    plt.show()


#
# fig, ax = plt.subplots(5, 1, figsize=(8, 20))
# for column in df.columns:
#     if 'step' in column:
#         continue
#     env_id, arch = column.split('_')
#     # get index of env_id in env_ids
#     env_id_index = env_ids.index(env_id)
#     color = colors[env_id_index]
#     linestyle = '--' if 'MLP' in arch else '-'
#     ax[env_id_index].plot(df['step'], df[column], label=env_id, linestyle=linestyle, color = color)
#
# # ax.set_xlabel("Time")
# # ax.set_ylabel("Normalized Moving Average Cost")
# # ax.legend()
#
# plt.show()







