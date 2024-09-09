import torch.distributions as D
import torch.nn as nn
import torch


# Create a basic neural network with 4 inputs, 4 outputs, and two hidden layers
# using sigmoid activation functions, with sigmoid output activation

mlp = nn.Sequential(
    nn.Linear(4, 8),
    nn.Sigmoid(),
    nn.Linear(8, 8),
    nn.Sigmoid(),
    nn.Linear(8, 4),
    nn.Sigmoid()
)

x = torch.randn([10,4])

# pass data through
y = mlp(x)

# use y as Bernoulli distribution parameters
dist = D.Bernoulli(probs=y)

# sample from the distribution
sample = dist.sample()

# take the logprobability of the sample
logprob = dist.log_prob(sample)


