import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    """
    Implements a simple scaled-dot product attention module.
    """

    def __init__(self, input_dim, output_dim):
        super(AttentionModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim


        # Define the linear layers for the queries, keys, and values
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

        # Define the output linear layer
        self.output_layer = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # X has shape (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply the linear layers to the input
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # Compute the scaled dot product attention
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.output_dim ** 0.5)

        # Compute the attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=2)

        # Compute the output
        output = torch.bmm(attention_weights, values)

        # Apply the output layer
        output = self.output_layer(output)

        return output

class StackedAttentionModule(nn.Module):
    """
    Stack num_layers attention modules.
    """

    def __init__(self, input_dim, output_dim, latent_dim, num_layers):
        super(StackedAttentionModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.attention_modules = nn.ModuleList([AttentionModule(input_dim, latent_dim) if i == 0 else AttentionModule(latent_dim, latent_dim) for i in range(num_layers)])

        self.output_layer = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.attention_modules[i](x)

        output = self.output_layer(x)

        return output






