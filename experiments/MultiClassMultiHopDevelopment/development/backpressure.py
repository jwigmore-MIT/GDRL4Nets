import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from modules.torchrl_development.utils.metrics import compute_lta

class BackpressureActor(TensorDictModule):
    """
    All inclusive Backpressure Agent
    NOTE: The actor must know
        1. Net.M
        2. Net.K
        3. Net.link_info
    :param TensorDictModule:
    :return:
    """
    def __init__(self, net, in_keys = ["Q", "cap", "mask"], out_keys = ["action"],):
        super().__init__(module= backpressure, in_keys = in_keys, out_keys=out_keys)
        self.set_topology(net)


    def set_topology(self, net):
        self.link_info = net.link_info
        self.M = net.M
        self.K = net.K

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["cap"], td["mask"],
                                   self.M, self.K, self.link_info)
        return td


def backpressure(Q: torch.Tensor, cap: torch.Tensor, mask: torch.Tensor, M: int, K: int, link_info: dict):
    """
    Runs the backpressure algorithm given the network

    Backpressure algorithm:
        For each link:
            1. Find the class which has the greatest difference between the queue length at the start and end nodes THAT IS NOT MASKED
            2. Send the largest class using the full link capacity for each link if there is a positive differential between start and end nodes
            3. If there is no positive differential, send no packets i.e. a_i[0,1:] = 0, a_i[0,0] = Y[i]

    :param net: MultiClassMultiHop object
    :param td: TensorDict object, should contain "mask" which is a torch.Tensor of shape [M, K]

    :return: action: torch.Tensor of shape [M, K] where M is the number of links and K is the number of classes
    """

    # send the largest class using the full link capacity for each link
    action = torch.zeros([M, K])
    for m in range(M):
        diff = (Q[link_info[m]["start"]]-Q[link_info[m]["end"]])*mask[m][1:]  # mask out the classes that are not allowed to be sent
        max_class = torch.argmax(diff)
        if diff[max_class] > 0:
            action[m, max_class] = cap[m]
    return action


if __name__ == "__main__":
    """
    All included in pytest script tests/test_backpressure.py
    """
    import json
    from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    env_json = "../envs/env2.json"
    with open(env_json, 'r') as file:
        env_info = json.load(file)
    net = MultiClassMultiHop(**env_info)
    bp_actor = BackpressureActor(net)

    td = net.reset()
    T = 1000
    _print = False
    total_arrivals = td["arrivals"].sum().item()
    total_departures = 0
    backlog = torch.zeros([T])
    # TODO: FIX THIS
    for t in range(T):
        if t < 1:
            td = bp_actor(td)
        else:
            td = bp_actor(td["next"])
        if _print:
            print(f"Q({t}): {td['Q']}")
            print(f"Action({t}): {net.convert_action(td['action'])}")
        td = net.step(td)
        backlog[t] = td["next","Q"].sum()
        total_departures += td["next","departures"].sum().item()
        total_arrivals += td["next", "arrivals"].sum().item()
        if _print:
            print(f"Departures({t}: {td['next', 'departures']}")
            print(f"Arrivals to each node: {[(k, net.arrival_map[k], td['arrivals'][k].item()) for k in range(net.K)]}")
        flag = total_arrivals == td["next","Q"].sum().item() + total_departures
        if not flag:
            # Raise error if the total arrivals is not equal to the total departures
            raise ValueError("Total arrivals is not equal to total departures")
    # Get time average backlog
    lta = compute_lta(backlog)
    plt.plot(backlog)
    plt.plot(lta)
    plt.show()
    print("Done")

    # repeat with torchrls rollout function

    rollout = net.rollout(max_steps=1000, policy=bp_actor)
    # check if the sum of departures + final Q == arrivals.sum()
    error = rollout["Q"][-1].sum() + rollout["departures"].sum() !=  rollout["arrivals"].sum()
    print(f"Error using torchrl env rollout: {error}")

