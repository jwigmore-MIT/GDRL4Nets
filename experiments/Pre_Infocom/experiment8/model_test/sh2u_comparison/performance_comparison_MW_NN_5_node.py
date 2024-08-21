import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# increase font size for title and labels
plt.rcParams.update({'font.size': 22})
file_name = "wandb_export_2024-03-26T08_12_43.600-04_00.csv"
file_name2 = "wandb_export_2024-04-02T07_00_47.014-04_00.csv"
# load csv, only get columns 2 and 4, and rows 1:10
o_df = pd.read_csv(file_name)
mw_df = pd.read_csv(file_name2)
# only keep columns 1, 4, 7
df = o_df.iloc[:, [1, 4, 7]]
df2 = mw_df.iloc[:, [1]]
# rename columns
df.columns = ["c", 'b', 'a']
df2.columns = ["mw_nn"]

# create new dataframe from df["b"] and df2["mw_nn"]
df["mw_nn"] = df2["mw_nn"]




# Plot both the single and multi columns
fig, ax = plt.subplots(figsize=(15, 10))
#df.plot(kind='bar', ax=ax, label = ["multi", "single"])

# df['single'].plot(kind='bar', ax=ax, label='single')
# df['multi'].plot(kind='bar', ax=ax, label='multi', color = 'orange')

# single = df['single'].to_numpy()
# multi = df['multi'].to_numpy()
b = deepcopy(df['b'].to_numpy())
mw = deepcopy(df['mw_nn'].to_numpy())
#count the number of times c >1

better_count = 0
for i in range(len(df)):
    if b[i] > mw[i]:
        better_count += 1


#min y
# min_y = np.minimum(a, b, c)

#sorted_indices = np.argsort(min_y)

for i in range(len(df)):
    # Sort a[i], b[i], c[i] in ascending order and get the indices of the sorted values

    sorted_indices = np.argsort([b[i], mw[i]])
    sorted_indices = sorted_indices[::-1]  # Reverse the order to get the descending order
    # Plot the bars in the order of the sorted indices
    for j in sorted_indices:
        if i == 0:
            ax.bar(i, [b[i], mw[i]][j], color=['orange', 'purple' ][j], label = ['NN Agent', 'FMW Agent'][j])
        else:
            ax.bar(i, [b[i], mw[i]][j], color=['orange', 'purple'][j])


# only plot every 10th environment id
ax.set_xticks(np.arange(0, len(df), 10))
ax.hlines(np.ones_like(df['b']), 0, len(df), color='r', label='MW')
ax.set(ylabel='value', title='Single Environments', xlabel='Environment ID')
# ax.set({ "ylabel": "MW Normalized Performance", "xlabel": "Environment ID" })
ax.legend()
plt.show()

# Take the mean of the ratio of df["nn"] and df["mw"]
mean = np.mean(df["mw_nn"] / df["b"])

# Calcuate the percente improvement of mw_nn over "b" where lower is better
percent_improvement = (1 - mean) * 100
# TA