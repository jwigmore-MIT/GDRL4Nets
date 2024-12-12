from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict





file_path = "../envs/env0.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)


net = MultiClassMultiHop(**env_info)
total_arrivals = 0
total_departures = 0
for t in range(1000):
    # print(f"Step {t}")
    # print(f"Q")
    td = TensorDict({"action":net._get_random_valid_action(), "valid_action": torch.Tensor([True])})
    # print("Action")
    # print(td["action"])
    out = net._step(td)
    total_arrivals += out["arrivals"].sum().item()
    total_departures += out["departures"].sum().item()
    flag = total_arrivals == out["Q"].sum().item() + total_departures
    if not flag:
        print("------------------------------------------------")
        print(f"Step {t}, Total arrivals: {total_arrivals}, Total departures: {total_departures}, Q: {out['Q'].sum().item()}, Flag: {flag}")
        print(net.Q)
print(f"Step {t}, Total arrivals: {total_arrivals}, Total departures: {total_departures}, Q: {out['Q'].sum().item()}, Flag: {flag}")

