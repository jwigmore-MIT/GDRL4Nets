import pandas as pd
import matplotlib.pyplot as plt

file_name = "wandb_export_2024-03-25T17_27_12.518-04_00.csv"
# load csv, only get columns 2 and 4, and rows 1:10
df = pd.read_csv(file_name, usecols=[2, 4], nrows=10)
# rename columns
df.columns = ["multi", "single"]

# plot df as a bar chart
fig, ax = plt.subplots(figsize=(15, 10))
df.plot(kind='bar', ax=ax, label = ["multi", "single"])
ax.legend(["multi", "single"])
plt.show()

# plot the difference between the two columns
fig, ax = plt.subplots(figsize=(15, 10))
(df["multi"] - df["single"]).plot(kind='bar', ax=ax)
plt.show()
# plot the difference between the two columns

