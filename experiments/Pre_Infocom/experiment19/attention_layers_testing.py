import torch
from torchrl_development.attention_networks import StackedAttentionModule

# Define the input parameters
input_dim = 4
output_dim = 1
latent_dim = 7
num_layers = 3

# Create an instance of StackedAttentionModule
stacked_attention_module = StackedAttentionModule(input_dim, output_dim, latent_dim, num_layers)

# Create some random data to pass through the module
batch_size = 4
seq_len = 6
data = torch.randn(batch_size, seq_len, input_dim)

# Pass the data through the module
output = stacked_attention_module(data)

# Print the output
print(output)

# Run the test
