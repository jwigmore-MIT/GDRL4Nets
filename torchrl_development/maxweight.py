import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict import TensorDict


class MaxWeightActor(TensorDictModule):

    def __init__(self, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= maxweight, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        td["action"] = maxweight(td["Q"], td["Y"])
        return td


def maxweight(Q, Y):
    # Q are the queue lengths
    # Y are the channel weights
    # returns the maxweight schedule as a zero-one vector of length Q.shape[0] +1, where element 0 corresponds to
    #idling
    v = Q*Y
    if torch.all(v==0):
        # first element is one, all others is zero
        return torch.Tensor(([1] + [0]*Q.shape[0])).int()
    else:
        # find the max index
        max_index = torch.argmax(v)
        # create the schedule
        schedule = torch.zeros(Q.shape[0]+1)
        schedule[max_index+1] = 1
        return schedule.int()

if __name__ == "__main__":
    # test maxweight
    Q = np.array([1,2,3])
    Y = np.array([1,2,3])
    print(maxweight(Q,Y))
    Y = np.array([3,2,1])
    print(maxweight(Q,Y))
    Y = np.array([0,0,0])
    print(maxweight(Q,Y))