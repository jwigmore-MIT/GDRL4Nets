# test to see if poisson arrival/service rates work as intended

# import make_env
import numpy as np
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.envs.env_generators import make_env, parse_env_json
from torchrl_development.utils.metrics import compute_lta
from copy import deepcopy

print("Running Poisson Test")
env_params = parse_env_json("SH3.json")
new_params = deepcopy(env_params)

# get Y_params
Y_params = new_params["Y_params"]



# get the service rates as a numpy array from the Y_params
service_rates = np.array([params["service_rate"] for key, params in Y_params.items()])

# sample uniform random variable between 1 and 10
#service_rates = np.random.uniform(1, 10, size = len(Y_params))

#eps = 0.05
#load  = service_rates.shape[0]/(service_rates.shape[0]-1) + eps
# load = 4/3+0.05
load =1.37
# Sample a uniform rv of length equal to the number of X_params
U = np.random.uniform(0, 1, size = len(Y_params))

# Normalize the uniform rv
U = U/np.sum(U)*load
#U = 0.49*np.ones(len(Y_params))

# Compute new arrival rates by mulitplying U by the service rates
arrival_rates = U*service_rates
#arrival_rates[0] = service_rates[0]/2


# Set new_params arrival rates to the new arrival rates
for i, params in new_params["X_params"].items():
    ind = int(i)-1
    params["arrival_rate"] = arrival_rates[ind]

for i, params in new_params["Y_params"].items():
    ind = int(i)-1
    params["service_rate"] = service_rates[ind]


env = make_env(new_params,
             observe_lambda = False,
             seed=0,
             device="cpu",
             terminal_backlog=500,
             observation_keys=["Q", "Y"],
             inverse_reward= False,
             stat_window_size =  100000,
             terminate_on_convergence =  True,
             convergence_threshold = 0.1,
             terminate_on_lta_threshold =  False,

)

max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
print("Collecting Max Weight rollout")
td = env.rollout(policy=max_weight_actor, max_steps = 50000)

#plot backlog
import matplotlib.pyplot as plt
lta = compute_lta(td["backlog"])
ltas = np.array([compute_lta(q) for q in td["Q"].T])
plt.plot(ltas.T)
plt.show()

print("Arrival Rates: ", arrival_rates)
print("Service Rates: ", service_rates)
print("Load: ", load)
print("Load n", list(arrival_rates/service_rates))

