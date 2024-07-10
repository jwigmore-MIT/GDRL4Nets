import torch
from torchrl_development.SMNN import DeepSetScalableMonotonicNeuralNetwork

# Define the parameters for the DeepSetScalableMonotonicNeuralNetwork
num_classes = 2
phi_in_dim = 3
latent_dim = 64
deepset_width = 16
deepset_out_dim = 16
exp_unit_size = (64, 64)
relu_unit_size = (64, 64)
conf_unit_size = (64, 64)

# Create an instance of the DeepSetScalableMonotonicNeuralNetwork
model = DeepSetScalableMonotonicNeuralNetwork(num_classes, phi_in_dim, latent_dim, deepset_width, deepset_out_dim, exp_unit_size, relu_unit_size, conf_unit_size)

# Create a random tensor to represent the input
input_tensor = torch.randn((10, num_classes * phi_in_dim))

# Pass the input through the model
output = model(input_tensor)

# Print the output
print(output)