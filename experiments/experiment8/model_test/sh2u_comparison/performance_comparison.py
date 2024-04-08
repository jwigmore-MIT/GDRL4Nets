import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# increase font size for title and labels
plt.rcParams.update({'font.size': 22})
file_name = "wandb_export_2024-03-25T17_27_12.518-04_00.csv"
# load csv, only get columns 2 and 4, and rows 1:10
df = pd.read_csv(file_name, usecols=[2, 4])
# rename columns
df.columns = ["multi", "single"]

# # plot df as a bar chart
# fig, ax = plt.subplots(figsize=(15, 10))
# df.plot(kind='bar', ax=ax, label = ["multi", "single"])
# ax.legend(["multi", "single"])
# plt.show()
#
# # plot the difference between the two columns
# fig, ax = plt.subplots(figsize=(15, 10))
# (df["multi"] - df["single"]).plot(kind='bar', ax=ax)
# plt.show()
# plot the difference between the two columns

# Plot a bar chart of just the single column
fig, ax = plt.subplots(figsize=(15, 10))
df["single"].plot(kind='bar', ax=ax, label='single')
# only plot every 10th environment id
ax.set_xticks(np.arange(0, len(df), 10))
ax.hlines(np.ones_like(df["single"]), 0, len(df), color='r', label='MW')
ax.set(ylabel='value', title='Single', xlabel='Environment ID')
# ax.set({ "ylabel": "MW Normalized Performance", "xlabel": "Environment ID" })
ax.legend()
plt.show()

# Plot both the single and multi columns
fig, ax = plt.subplots(figsize=(15, 10))
#df.plot(kind='bar', ax=ax, label = ["multi", "single"])

df['single'].plot(kind='bar', ax=ax, label='single')
df['multi'].plot(kind='bar', ax=ax, label='multi', color = 'orange')

single = df['single'].to_numpy()
multi = df['multi'].to_numpy()

#min y
min_y = np.minimum(single, multi)

sorted_indices = np.argsort(min_y)

for i in sorted_indices:
    if single[i] < multi[i]:
        zorder1 = 2
        zorder2 = 1
    else:
        zorder1 = 1
        zorder2 = 2
    ax.bar(i, single[i], color='b', zorder = zorder1)
    ax.bar(i, multi[i], color='orange', zorder = zorder2)

# only plot every 10th environment id
ax.set_xticks(np.arange(0, len(df), 10))
ax.hlines(np.ones_like(df["single"]), 0, len(df), color='r', label='MW')
ax.set(ylabel='value', title='Single and Multi (5)', xlabel='Environment ID')
# ax.set({ "ylabel": "MW Normalized Performance", "xlabel": "Environment ID" })
ax.legend()
plt.show()


# Compute the the average difference between the two columns
print(f"Average difference: {np.mean(df['multi'] - df['single'])}")


