import torch.nn as nn
import torch.nn.functional as F
import torch
from tensordict.nn import TensorDictModule

class NN_Actor(TensorDictModule):

    def __init__(self, module, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td):
        x = torch.cat([td[in_key] for in_key in self.in_keys]).unsqueeze(0)
        td["action"] = self.module(x).squeeze(0)
        return td


class MaxWeightNetwork(nn.Module):
    def __init__(self, weight_size, temperature=1.0):
        super(MaxWeightNetwork, self).__init__()
        # randomly initialize the weights to be gaussian with mean 1 and std 0.1
        self.weights = nn.Parameter(torch.randn(weight_size)*0.1) #torch.ones(weight_size)*
        self.temperature = temperature
    def forward(self, x):
        # split x into Q and Y
        Q = x[:, :x.shape[1]//2]
        Y = x[:, x.shape[1]//2:]
        z = Y * Q * self.weights
        # add a column of 0.1 to z
        z = torch.cat([torch.ones((z.shape[0],1)), z], dim=1)
        #z = torch.cat([torch.zeros((z.shape[0],1)), z], dim=1)
        # normalize z
        #z = z / z.sum(dim=1).unsqueeze(1)
        if self.training:
            #A = z
            A = torch.nn.functional.softmax(z/self.temperature, dim=-1)
        else:
            A = torch.zeros_like(z)
            A[torch.arange(z.shape[0]), torch.argmax(z, dim=-1)] = 1
        #A_one_hot = F.one_hot(A, num_classes=z.shape[1] + 1)
        return A

class LinearNetwork(nn.Module):
    '''Simple linear network where each output is a linear combination of the inputs'''
    def __init__(self, input_size, output_size):
        super(LinearNetwork, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
    def forward(self, x):
        """ Simple linear network """
        input = x.float()
        z = torch.matmul(input, self.weights)
        if self.training:
            A = z
            #A = torch.nn.functional.softmax(z, dim=-1)
        else:
            A = torch.zeros_like(z)
            A[torch.arange(z.shape[0]), torch.argmax(z, dim=-1)] = 1

        return A


class FeedForwardNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = torch.cat([Q,Y], dim=1)
        x = F.relu(self.fc1(x.float()))
        x = self.fc2(x)
        # if training, use softmax output on z
        if self.training:

            z = x
            #z = F.softmax(x, dim=-1)
        else:
            # if not training, then use argmax
            z = torch.zeros_like(x)
            z[torch.arange(x.shape[0]), torch.argmax(x, dim=-1)] = 1
        return z

### Wrap a