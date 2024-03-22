import numpy as np
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.envs.env_generator import make_env, parse_env_json
from torchrl_development.utils.metrics import compute_lta
import json

def convex_combination(arrival_rates, scale = 1.0):
    # create a convex combination of the arrival rates
    weights = np.random.dirichlet(np.ones(len(arrival_rates))).reshape(-1,1)
    return scale*np.sum(weights*arrival_rates, axis = 0)


import math

class CircularBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []

    def append(self, value):
        popped_value = None
        if len(self.buffer) == self.buffer_size:
            popped_value = self.buffer[self.index]
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)  # Expand buffer if not full yet
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.buffer_size
        return popped_value

    @property
    def length(self):
        return len(self.buffer)

    def items(self):
        return self.buffer

class RunningStatsCalculator:
    def __init__(self, buffer_size):
        self.circularBuffer = CircularBuffer(buffer_size)
        self._mean = 0
        self._dSquared = 0

    @property
    def count(self):
        return self.circularBuffer.length

    def update(self, new_value):
        popped_value = self.circularBuffer.append(new_value)

        if self.count == 1 and popped_value is None:
            # initialize when the first value is added
            self._mean = new_value
            self._dSquared = 0
        elif popped_value is None:
            # if the buffer is not full yet, use standard Welford method
            mean_increment = (new_value - self._mean) / self.count
            new_mean = self._mean + mean_increment

            d_squared_increment = ((new_value - new_mean) * (new_value - self._mean))
            new_d_squared = self._dSquared + d_squared_increment

            self._mean = new_mean
            self._dSquared = new_d_squared
        else:
            # once the buffer is full, adjust Welford Method for window size
            mean_increment = (new_value - popped_value) / self.count
            new_mean = self._mean + mean_increment

            d_squared_increment = ((new_value - popped_value) * (new_value - new_mean + popped_value - self._mean))
            new_d_squared = self._dSquared + d_squared_increment

            self._mean = new_mean
            self._dSquared = new_d_squared

    @property
    def mean(self):
        return self._mean

    @property
    def dSquared(self):
        return self._dSquared

    @property
    def populationVariance(self):
        return self.dSquared / self.count

    @property
    def populationStdev(self):
        return math.sqrt(self.populationVariance)

    @property
    def sampleVariance(self):
        return self.dSquared / (self.count - 1) if self.count > 1 else 0

    @property
    def sampleStdev(self):
        return math.sqrt(self.sampleVariance)


def compute_incremental_mean_sd(data, window_size = 1000):
    """
    Compute the running mean and standard deviation of the data over the window size using
    an incremental algorithm
    :param data:
    :param window_size:
    :return:
    """

    # initialize the mean and variance
    mean = 0
    M2 = 0
    d_squared = 0
    mean_values = []
    sd_values = []
    buffer = []
    # compute incremental mean and variance over the window size
    for i, x in enumerate(data):
        if i < window_size: # if the buffer is not full, use the Welford's algorithm
            n = i+1
            delta = x - mean
            new_mean = mean + delta/n
            delta2 = x - new_mean
            M2 = delta*delta2
            if n > 1:
                new_sd_sqr = d_squared + M2
            else:
                new_sd_sqr = 0
        else: # if the buffer is full, use the incremental window algorithm
            popped = buffer.pop(0)
            new_mean = mean + (x - popped)/window_size
            M2 = (x-popped)*(x-new_mean+popped-mean)
            new_sd_sqr = d_squared + M2

        mean= new_mean
        d_squared = new_sd_sqr
        if i >= window_size-1:
            var = d_squared/(window_size-1)
        elif i == 0:
            var = 0
        else:
            var = d_squared/i
        buffer.append(x)
        mean_values.append(mean)
        sd_values.append(np.sqrt(var))
    return mean_values, sd_values

def tracker_test(data, window_size = 1000):
    tracker = RunningStatsCalculator(buffer_size = window_size)
    means = []
    sds = []
    for d in data:
        tracker.update(d)
        means.append(tracker.mean)
        sds.append(tracker.sampleStdev)
    return means, sds



if __name__ == "__main__":

    context_space_dict = json.load(open("SH2u_lf1.32_context_space-nondominated.json", 'rb'))
    N = context_space_dict["num_envs"]

    # get N arrival rates from the context_vertex_dict["env_params"] and create a convex combination of them
    ind = np.random.choice(context_space_dict["num_envs"], N, replace = False)
    vertex_arrival_rates = [context_space_dict["context_dicts"][str(i)]["arrival_rates"] for i in ind]

    convex_arrival_rates = convex_combination(vertex_arrival_rates, scale = 1.0)

    # create a new environment with the convex_arrival_rate
    new_params = context_space_dict["context_dicts"]["0"].copy()
    for i in range(N):
        new_params["arrival_rates"] = convex_arrival_rates

    for i, param in new_params["env_params"]["X_params"].items():
        ind = int(i)-1
        param["arrival_rate"] = convex_arrival_rates[ind]

    new_params["env_params"]["stat_window_size"] = 100000
    new_params["env_params"]["terminate_on_convergence"] = True
    new_params["env_params"]["convergence_threshold"] = 0.1

    env = make_env(new_params["env_params"],
                   observe_lambda = False,
                   seed=0,
                   device="cpu",
                   terminal_backlog=None,
                   observation_keys=["Q", "Y"],
                   inverse_reward= False)
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    # test the environment

    td = env.rollout(policy=max_weight_actor, max_steps=50000)
    ta_stdev = td["ta_stdev"].numpy()

    import matplotlib.pyplot as plt
    from torchrl_development.utils.metrics import compute_lta, compute_windowed_rmse

    lta = compute_lta(td["backlog"])
    ltas = np.array([compute_lta(q) for q in td["Q"].T])
    fig, axes = plt.subplots(3,1, figsize = (15,10))
    axes[0].plot(ltas.T, label = [f"Queue {i}" for i in range(ltas.shape[0])])
    axes[0].plot(lta, label = "LTA")

    # Compute the RMSE of the lta and plot on the second axis
    rmse = compute_windowed_rmse(lta, window_size = 1000)
    mean, sd = tracker_test(lta, window_size = 1000)
    cov = np.array(sd)/np.array(mean)
    axes[1].plot(rmse, label = "RMSE")
    axes[1].plot(sd, label = "SD")
    axes[1].plot(ta_stdev, label = "TA Stdev")
    axes[0].plot(mean, "--", label = "Mean")

    axes[2].plot(cov, label = "CoV")

    # find where rmse crosses the threshold of 0.1
    rmse_threshold = 0.1
    threshold_crosses = np.where(rmse < rmse_threshold)[0]
    # get first threshold cross greater than 1000
    if threshold_crosses.size < 2:
        first_cross = len(rmse)-1
    else:
        first_cross = threshold_crosses[np.where(threshold_crosses > 1000)[0].min()]
    # plot a vertical line at the first cross on both axis
    axes[0].axvline(first_cross, color = "red", linestyle = "--", label = f"RMSE < {rmse_threshold} at {first_cross}")
    axes[1].axvline(first_cross, color = "red", linestyle = "--", label = f"RMSE < {rmse_threshold} at {first_cross}")
    axes[2].axvline(first_cross, color = "red", linestyle = "--", label = f"RMSE < {rmse_threshold} at {first_cross}")
    axes[0].legend(loc = "best")

    axes[1].legend(loc = "best")
    plt.show()



    print("Vertex Arrival Rates: ", vertex_arrival_rates)
    print("Convex Arrival Rates: ", convex_arrival_rates)

    # print the time at which the RMSE crosses the threshold
    print("Time at which RMSE crosses the threshold: ", first_cross)
    print("RMSE at time of threshold crossing: ", rmse[first_cross])
    print("LTA backlog at time of threshold crossing: ", lta[first_cross])
    print("LTA backlog at end of simulation: ", lta[-1])


    # Compute the root