import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



"""
Take a context set dictionary and compute statistics on the lta, load, and arrival rates
"""


def plot_arrival_rate_histogram(arrival_rates, title = "Arrival Rates Histogram"):
    # plot stacked barplot of the arrival rates,
    # an arrival rate vector has length K, so we will have N stacks of K bars,
    # use pandas dataframe to create a stacked barplot
    # where each stack corresponds to one element of the arrival rate
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))

    df = pd.DataFrame(arrival_rates, columns=[f"$\\lambda_{i+1}$" for i in range(arrival_rates.shape[1])])
    df.plot(kind='bar', stacked=True, ax=axes)
    axes.set_title(title)
    fig.show()




    # N, K = arrival_rates.shape
    # x = np.arange(N)
    # # a color list for K colors based on the K colors in the matplotlib color cycle
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # for n in range(N):
    #     bottoms = np.zeros(K)
    #     arrival_rate = arrival_rates[n]
    #     for k in range(K):
    #         axes.bar(n, arrival_rate[k], bottom=bottoms[k], label=f"Arrival rate {k}", color=colors[k])
    #         bottoms[k] += arrival_rate[k]
    #
    # axes.set_title("Arrival Rates Histogram")
    #
    # plt.show()

def plot_lta_histogram(ltas):
    plt.rcParams.update({'font.size': 22})
    plt.rcParams.update({'axes.titlesize': 22})
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    #increase font size for title and labels

    axes.hist(ltas, bins=40, color='skyblue', edgecolor='black')
    axes.set_title("Histogram of LTAs")
    axes.set_xlabel("LTA")
    axes.set_ylabel("Frequency")
    axes.grid(True)
    axes.legend(["LTAs"])
    # increase font size for axis labels
    fig.show()


if __name__ == "__main__":
    #context_set_dict = json.load(open("SH2u_context_set.json", 'rb'))
    context_set_dict = json.load(open("SH3_lf1_5.json", 'rb'))
    ltas = context_set_dict["ltas"]
    arrival_rates = np.array([context_set_dict["context_dicts"][str(i)]["arrival_rates"] for i in range(context_set_dict["num_envs"])])
    network_loads = context_set_dict["network_loads"]
    # plot histogram of ltas
    plot_lta_histogram(ltas)
    plot_arrival_rate_histogram(arrival_rates)


