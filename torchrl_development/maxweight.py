import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict import TensorDict


class MaxWeightActor(TensorDictModule):

    def __init__(self, in_keys = ["Q", "Y"], out_keys = ["action"], index = 'min'):
        maxweight_function = maxweight if index == 'min' else maxweight_high_index
        super().__init__(module= maxweight_function, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["Y"])
        return td


def maxweight(Q, Y):
    # Q are the queue lengths  -- size (B, N) where B is number in the batch and N is the number of queues
    # Y are the channel weights --  size (B, N) where B is number in the batch and N is the number of queues
    # returns the maxweight schedule as a zero-one vector of length Q.shape[0] +1, where element 0 corresponds to
    #idling
    if Q.dim() == 1:
        Q = Q.unsqueeze(0)
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)
    v = Q*Y
    all_zeros = torch.all(v == 0, dim=-1)
    max_index = torch.argmax(v, dim=-1)

    schedule = torch.zeros((Q.shape[0], Q.shape[1] + 1)).int()
    schedule[all_zeros, 0] = 1
    schedule[~all_zeros, max_index[~all_zeros] + 1] = 1
    return schedule.int()

def argmax_with_largest_index(tensor):

    N, M = tensor.shape

    # Reverse the tensor along the last dimension
    reversed_tensor = tensor.flip(-1)

    # Get the argmax of the reversed tensor
    reversed_argmax_indices = torch.argmax(reversed_tensor, dim=-1)

    # Adjust the indices to get the original indices
    argmax_indices = M - reversed_argmax_indices - 1
    return argmax_indices




def maxweight_high_index(Q,Y):
    """
    Like maxweight, but in the event of a tie, the highest index queue is selected
    :param Q:
    :param Y:
    :return:
    """
    if Q.dim() == 1:
        Q = Q.unsqueeze(0)
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)
    v = Q*Y
    all_zeros = torch.all(v == 0, dim=-1)
    # get argmax of v along the last dimension, and incase of a tie, select the highest index

    max_index = argmax_with_largest_index(v)
    schedule = torch.zeros((Q.shape[0], Q.shape[1] + 1)).int()
    schedule[all_zeros, 0] = 1
    schedule[~all_zeros, max_index[~all_zeros] + 1] = 1
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