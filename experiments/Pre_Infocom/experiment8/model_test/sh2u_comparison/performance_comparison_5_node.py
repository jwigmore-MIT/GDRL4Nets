import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# increase font size for title and labels
plt.rcParams.update({'font.size': 22})
file_name = "wandb_export_2024-03-26T08_12_43.600-04_00.csv"
# load csv, only get columns 2 and 4, and rows 1:10
o_df = pd.read_csv(file_name)
# only keep columns 1, 4, 7
df = o_df.iloc[:, [1, 4, 7]]
# rename columns
df.columns = ["c", 'b', 'a']



# Plot both the single and multi columns
fig, ax = plt.subplots(figsize=(15, 10))
#df.plot(kind='bar', ax=ax, label = ["multi", "single"])

# df['single'].plot(kind='bar', ax=ax, label='single')
# df['multi'].plot(kind='bar', ax=ax, label='multi', color = 'orange')

# single = df['single'].to_numpy()
# multi = df['multi'].to_numpy()
a = deepcopy(df['a'].to_numpy())
b = deepcopy(df['b'].to_numpy())
c = deepcopy(df['c'].to_numpy())

#count the number of times c >1
count = 0
for i in range(len(df)):
    if c[i] > 1:
        count += 1


#min y
# min_y = np.minimum(a, b, c)

#sorted_indices = np.argsort(min_y)

for i in range(len(df)):
    # Sort a[i], b[i], c[i] in ascending order and get the indices of the sorted values

    sorted_indices = np.argsort([a[i], b[i], c[i]])
    sorted_indices = sorted_indices[::-1]  # Reverse the order to get the descending order
    # Plot the bars in the order of the sorted indices
    for j in sorted_indices:
        if i == 0:
            ax.bar(i, [a[i], b[i], c[i]][j], color=['b', 'orange', 'g'][j], label = ['a', 'b', 'c'][j])
        else:
            ax.bar(i, [a[i], b[i], c[i]][j], color=['b', 'orange', 'g'][j])


# only plot every 10th environment id
ax.set_xticks(np.arange(0, len(df), 10))
ax.hlines(np.ones_like(df['a']), 0, len(df), color='r', label='MW')
ax.set(ylabel='value', title='Single Environments', xlabel='Environment ID')
# ax.set({ "ylabel": "MW Normalized Performance", "xlabel": "Environment ID" })
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(15, 10))

## Repeat the same but with only 'a' and 'b' columns
for i in range(len(df)):
    # Sort a[i], b[i], c[i] in ascending order and get the indices of the sorted values

    sorted_indices = np.argsort([a[i], b[i]])
    sorted_indices = sorted_indices[::-1]  # Reverse the order to get the descending order
    # Plot the bars in the order of the sorted indices
    for j in sorted_indices:
        if i == 0:
            ax.bar(i, [a[i], b[i]][j], color=['b', 'orange'][j], label = ['a', 'b'][j])
        else:
            ax.bar(i, [a[i], b[i]][j], color=['b', 'orange'][j])


# only plot every 10th environment id
ax.set_xticks(np.arange(0, len(df), 10))
ax.hlines(np.ones_like(df['a']), 0, len(df), color='r', label='MW')
ax.set(ylabel='value', title='Single Environments', xlabel='Environment ID')
# ax.set({ "ylabel": "MW Normalized Performance", "xlabel": "Environment ID" })
ax.legend()
fig.show()

# Compute the the average difference between the two columns
# print(f"Average difference: {np.mean(df['multi'] - df['single'])}")


