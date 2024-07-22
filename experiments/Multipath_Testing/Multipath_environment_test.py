from torchrl_development.envs.env_generators import make_env, parse_env_json
import tensordict as td
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
env_params = parse_env_json("MP3.json")
env= make_env(env_params,
                observation_keys=["arrivals", "Q", "Y"])

state = env.reset()

action = td.TensorDict({"action" : env.base_env.get_stable_action()}, batch_size=[])

out = env.step(action)

out = env.rollout(50000, policy = env.base_env.get_stable_action)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

ax[0].plot(compute_lta(out["backlog"]))
ax[0].set_title("Time Average Backlog")

ax[1].plot(out["ta_stdev"])
ax[1].set_title("Time Average Backlog Std")



fig.show()



