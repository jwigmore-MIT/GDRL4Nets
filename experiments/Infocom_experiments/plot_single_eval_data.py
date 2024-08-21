import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12, "font.family": "Arial"})

# Load the data
file = "training_data/SH_single_env_training_data_eval.csv"
csv_data = pd.read_csv(file)
context_set = "SH4"

df = csv_data["trainer/step"]
df = df.rename("step")
for column in csv_data.columns:
    if "normalized_lta_backlog_training_envs" in column:
        if 'normalized_lta_backlog_training_envs_' in column:
            continue
        df = pd.concat([df, csv_data[column]], axis=1)

for column in df.columns:
    if 'step' in column:
        continue
    splitted = column.split('_')
    arch = splitted[0]
    if "PMN" in arch:
        arch = "STN"
    env_id = None
    for i in range(1, len(splitted)):
        if context_set not in splitted[i]:
            continue
        env_id = f"{splitted[i+1]}"
    df = df.rename(columns={column: f"{env_id}_{arch}"})

df = df.reindex(sorted(df.columns), axis=1)

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

"""
Want to make a bar plot of the data. The x-axis should have labels (env_id, arch) and the y-axis should have the
normalized_lta_backlog_training_envs values. The bars should be colored by the env_id 
"""

# Melt the DataFrame to get the required format
df_melted = df.melt(id_vars='step', var_name='env_id_arch', value_name='normalized_lta_backlog_training_envs')

# Split the 'env_id_arch' column into two separate columns
df_melted[['env_id', 'arch']] = df_melted['env_id_arch'].str.split('_', expand=True)

labels = []
for i in range(len(df_melted['env_id'])):
    env_index = env_ids.index(df_melted['env_id'][i])
    labels.append(f"({env_index+1}, {df_melted['arch'][i]})")

df_melted["labels"] = labels

# Create a color mapping for the 'env_id' values
color_mapping = {env_id: color for env_id, color in zip(df_melted['env_id'].unique(), colors)}

# Create the bar plot
plt.figure(figsize=(10, 8))
plt.bar(df_melted['labels'], df_melted['normalized_lta_backlog_training_envs'],
        color=df_melted['env_id'].map(color_mapping))
plt.xlabel('Environment ID and Architecture')
plt.ylabel('Normalized Average Cost')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.title("Single-hop Evaluation Performance")

# Save Figure
plt.savefig("training_data/SH_single_env_training_data_eval.png", bbox_inches='tight')

plt.show()

