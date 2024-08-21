from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
import os
import torch

file_mods = ["base", "a", "b", "c", "d"]
time_steps = 5_000
trials = 5




# Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
results = {}
# Test the environment
for mod in file_mods:

    local_path = f"Singlehop_Two_Node_Simple_{mod}.json"
    full_path = os.path.join(os.getcwd(), local_path)
    base_env_params = parse_env_json(full_path=full_path)
    mod_results = []
    for i in range(trials):
        env = make_env(base_env_params, terminal_backlog=100, seed = i)
        td = env.rollout(policy=mw_actor, max_steps = time_steps)
        lta = compute_lta(td["backlog"])
        mod_results.append(lta)
    # average the mod_results
    results[mod] = torch.stack(mod_results).mean(dim=0)

# Plot the results
fig, ax = plt.subplots(1,1)
for mod, lta in results.items():
    ax.plot(lta, label=mod)
    print("Mod: ", mod, "LTA: ", lta[-1])
ax.legend()
plt.show()






