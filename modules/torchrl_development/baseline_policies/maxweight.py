import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

class CGSLGSActor(TensorDictModule):
    """
    Local Greedy Search actor for Conflict Graph Scheduling environments
    """

    def __init__(self, in_keys = ["q", "s", "adj"], out_keys = ["action"],):

        super().__init__(module= local_greedy_search, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        mwis_action, mwis, total_ws = self.module(td["adj_sparse"], td["q"]*td["s"])
        td["action"] = torch.Tensor(mwis_action)
        return td

class CGSMaxWeightActor(TensorDictModule):
    """
    Max Weight actor for Conflict Graph Scheduling environments
    """

    def __init__(self, in_keys = ["q", "s"], out_keys = ["action"], valid_actions = None):
        if valid_actions is None:
            raise ValueError("valid_actions must be provided")
        super().__init__(module= cgs_maxweight, in_keys = in_keys, out_keys=out_keys)
        self.valid_actions = valid_actions

    def forward(self, td: TensorDict):
        td["action"] = self.module(self.valid_actions, td["q"], td["s"])
        return td
def cgs_maxweight(valid_actions, q, s):
    """
    Compute the maxweight scheduling policy
    argmax(q*y*a) where a is a valid action
    :param valid_actions: (K, N) binary tensor where K is the number of valid actions
    :param q: (B,N) tensor of queue sizes
    :param s: (B, N) tensor of service states
    :return: (B, N) tensor of the maxweight action
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
        s = s.unsqueeze(0)

    return valid_actions[torch.argmax(torch.sum(q * s * valid_actions, dim = 1), dim=0)]



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

def local_greedy_search(adj, wts):
    '''
    Return MWIS set and the total weights of MWIS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :return: mwis, total_wt
    '''
    if adj.shape[0] != adj.shape[1]:
        adj = torch.sparse.FloatTensor(adj, torch.ones(adj.shape[1])).to_dense().numpy()
    wts = np.array(wts).flatten()
    verts = np.array(range(wts.size))
    mwis = set()
    mwis_action = np.zeros(wts.size)
    ind = -1
    remain = set(verts.flatten())
    vidx = list(remain)
    nb_is = set()
    while len(remain) > 0:
        for v in remain:
            ind +=1
            # if v in nb_is:
            #     continue
            nb_set = np.nonzero(adj[v])[0]# Get neighbors
            nb_set = set(nb_set).intersection(remain)
            if len(nb_set) == 0:
                mwis.add(v)
                continue
            nb_list = list(nb_set)
            nb_list.sort()
            wts_nb = wts[nb_list]
            w_bar_v = wts_nb.max()
            if wts[v] > w_bar_v:
                mwis.add(v)
                mwis_action[ind] = 1
                nb_is = nb_is.union(set(nb_set))
            elif wts[v] == w_bar_v:
                i = list(wts_nb).index(wts[v])
                nbv = nb_list[i]
                if v < nbv:
                    mwis.add(v)
                    mwis_action[ind] = 1

                    nb_is = nb_is.union(set(nb_set))
            else:
                pass
        remain = remain - mwis - nb_is
    total_ws = np.sum(wts[list(mwis)])
    return mwis_action, mwis, total_ws

if __name__ == "__main__":
    # test maxweight
    Q = np.array([1,2,3])
    Y = np.array([1,2,3])
    print(maxweight(Q,Y))
    Y = np.array([3,2,1])
    print(maxweight(Q,Y))
    Y = np.array([0,0,0])
    print(maxweight(Q,Y))



