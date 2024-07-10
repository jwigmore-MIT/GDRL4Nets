from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Iterable, Sized, Tuple
from torchrl_development.DeepSets import create_deep_set_nn


def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        raise NotImplementedError("abstract methodd called")

class ExpUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 max_value: float = 1.0):
        super().__init__(in_features, out_features)
        torch.nn.init.uniform_(self.weight,a=-3.0, b=0) # weights seem to be too small using this one
        #torch.nn.init.uniform_(self.weight,a=-1, b=2.0)
        # truncated_normal_(self.bias, std=0.1)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.size = in_features
        self.max_value = max_value

    def forward(self, x):
        out = (x) @ torch.exp(self.weight) + self.bias
        return (1-0.01) * torch.clip(out, 0, self.max_value) + 0.01 * out

class NegExpUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 max_value: float = 1.0):
        super().__init__(in_features, out_features)
        torch.nn.init.uniform_(self.weight,a=-20.0, b=2)
        truncated_normal_(self.bias, std=0.5)
        self.size = in_features
        self.max_value = max_value

    def forward(self, x):
        out = (x) @ torch.exp(self.weight) + self.bias
        return (1-0.01) * torch.clip(out, -self.max_value, 0) + 0.01 * out

class ReLUnUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 max_value: float = 1.0):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)
        self.size = in_features
        self.max_value = max_value

    def forward(self, x):
        out = (x) @ self.weight + self.bias
        return (1-0.01) * torch.clip(out, 0, self.max_value) + 0.01 * out
class ReLUUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 **kwargs):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        out = (x) @ self.weight + self.bias
        return F.relu(out)

class ConfluenceUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)
        self.size = in_features

    def forward(self, x):
        out = (x) @ self.weight + self.bias
        return (1-0.01) * torch.clip(out, 0, 1) + 0.01 * out 

class FCLayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        # truncated_normal_(self.weight, mean=0.0, std=1)
        torch.nn.init.uniform_(self.weight, a=-3.0, b=0)
        #truncated_normal_(self.bias, std=0.1)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        FC = x @ torch.exp(self.weight) + self.bias
        return FC

class FCLayer_notexp(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        FC = x @ self.weight + self.bias
        return FC

class ScalableMonotonicNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_size: int, 
                 mono_size: int,
                 mono_feature,
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = (),
                 exp_unit: ActivationLayer = ExpUnit,
                 relu_unit: ActivationLayer = ReLUUnit,
                 conf_unit: ActivationLayer = ConfluenceUnit,
                 fully_connected_layer: ActivationLayer = FCLayer):

        super(ScalableMonotonicNeuralNetwork,self).__init__()
        
        self.input_size = input_size
        self.mono_size = mono_size
        self.non_mono_size = input_size - mono_size
        self.mono_feature = mono_feature
        self.non_mono_feature = list(set(list(range(input_size))).difference(mono_feature))
        self.exp_unit_size = exp_unit_size  
        self.relu_unit_size = relu_unit_size  
        self.conf_unit_size = conf_unit_size 

        self.exp_units = torch.nn.ModuleList([
            exp_unit(mono_size if i == 0 else exp_unit_size[i-1] + conf_unit_size[i-1], exp_unit_size[i])
            for i in range(len(exp_unit_size))
        ])

        self.relu_units = torch.nn.ModuleList([
            relu_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], relu_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.conf_units = torch.nn.ModuleList([
            conf_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], conf_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.fclayer = fully_connected_layer(exp_unit_size[len(exp_unit_size)-1] + conf_unit_size[len(exp_unit_size)-1] + relu_unit_size[len(relu_unit_size)-1],1)

    def forward(self,x):

        x_mono = x[:, self.mono_feature]
        x_non_mono = x[:, self.non_mono_feature]

        for i in range(len(self.exp_unit_size)):
            if i == 0 :
                exp_output = self.exp_units[i](x_mono)
                conf_output = self.conf_units[i](x_non_mono)
                relu_output = self.relu_units[i](x_non_mono)
                exp_output = torch.cat([exp_output, conf_output], dim=1)
            else :
                exp_output = self.exp_units[i](exp_output)
                conf_output = self.conf_units[i](relu_output)
                relu_output = self.relu_units[i](relu_output)
                exp_output = torch.cat([exp_output, conf_output], dim=1)

        out = self.fclayer(torch.cat([exp_output,relu_output],dim = 1)) 
        return out


class DeepSetScalableMonotonicNeuralNetwork(torch.nn.Module):
    """
    How this module works is that we have a MK length input vector where each consecutive M elements corresponds to a single output.
    Each output is obtained through the following process:
    For start in range(0, M, MK)
        Keep elements start:start+M as the monotonic feature ->
        Pass all other features into the invariant model
        Concatenate the output of the invariant model with the monotonic feature
        Pass concatenated output through the scalable monotonic neural network where the first M features are the monotonic feature, and the rest are not


    """
    def __init__(self,

                 num_classes,
                 phi_in_dim: int = 3,
                 latent_dim: int = 64,
                 deepset_width: int = 16,
                 deepset_out_dim: int = 16,
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = (),
                 exp_unit: ActivationLayer = ExpUnit,
                 relu_unit: ActivationLayer = ReLUUnit,
                 conf_unit: ActivationLayer = ConfluenceUnit,
                 fully_connected_layer: ActivationLayer = FCLayer):

        super(DeepSetScalableMonotonicNeuralNetwork,self).__init__()


        # Create Deep Set NN, which will be input into the non-nonotonic network
        self.total_input_size = num_classes*phi_in_dim
        self.mono_size = phi_in_dim
        self.invariant_model = create_deep_set_nn(phi_in_dim, latent_dim, deepset_out_dim, width = deepset_width)
        self.scalable_monotonic_network = ScalableMonotonicNeuralNetwork(int(phi_in_dim+deepset_out_dim), self.mono_size, list(range(0,self.mono_size)), exp_unit_size, relu_unit_size, conf_unit_size, exp_unit, relu_unit, conf_unit, fully_connected_layer)
        self.bias_0 = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """
        x will have shape (batch_size, phi_in_dim*num_classes)
        :param x:
        :return:
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Preallocate memory for out
        out = self.bias_0*torch.ones((x.shape[0], self.total_input_size // self.mono_size+1), device=x.device)

        for start in range(0, self.total_input_size, self.mono_size):
            mono_feature = x[:, start:start + self.mono_size]
            non_mono_feature = torch.cat((x[:, :start], x[:, start + self.mono_size:]), dim=1).reshape(x.shape[0], -1, self.mono_size)
            invariant_output = self.invariant_model(non_mono_feature)
            # Use in-place operation for concatenation
            invariant_output = torch.cat((mono_feature,invariant_output), dim=1)
            out[:, 1+ (start // self.mono_size)] = self.scalable_monotonic_network(invariant_output).squeeze()

        return out



class PureMonotonicNeuralNetwork(torch.nn.Module):
    """
    Same as ScalableMonotonicNeuralNetwork but with no confluence or Relu units
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 hidden_sizes: Tuple = (64, 64),
                 relu_max: float = 1.0,
                 exp_unit: ActivationLayer = ExpUnit,
                 fc_layer: ActivationLayer = FCLayer
                 ):
        super(PureMonotonicNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.relu_max = relu_max

        self.hidden_layers = torch.nn.ModuleList([
            exp_unit(self.input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i], max_value=relu_max) for i in range(len(hidden_sizes))
        ])

        self.fclayer = fc_layer(hidden_sizes[len(hidden_sizes)-1], output_size)

        self.network = torch.nn.Sequential(*self.hidden_layers, self.fclayer)


    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.fclayer(x)
        return out

        #return self.network(x)


class PureMonotonicPlusNeuralNetwork(torch.nn.Module):
    """
    Like the PureMonotonicNeuralNetwork but we take each state group, and pass it through its own monotonic network
    The input size is 3*N, and the output size is N+1, where N is the number of state groups
    Output n>1 corresponds with an action for state group n-1.
    Let S= (s_{1,1}, s_{1,2}, s_{1,3}, s_{2,1}, s_{2,2}, s_{2,3}, ...., s_{N,1}, s_{N,2}, s_{N,3})
    We pass S through the main network, and then pass each state group through its own monotonic network
    Main Network f_{\theta}(S): S\mapsto R^{N+1}
    Side networks f_{\phi_i}: S_i\mapsto R  where S_i = (s_{i,1}, s_{i,2}, s_{i,3})
    Output is
        f_\theta(S)_i + f_{\phi_i}(S_i))_{i=1}^{N+1}  for i=1,2,3
        f_\theta(S)_0 for i=0
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 hidden_sizes: Tuple = (64, 64),
                 side_net_sizes: Tuple = (8, 8),
                 relu_max: float = 1.0,
                 exp_unit: ActivationLayer = ExpUnit,
                 fc_layer: ActivationLayer = FCLayer
                 ):
        super(PureMonotonicNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.relu_max = relu_max

        self.hidden_layers = torch.nn.ModuleList([
            exp_unit(self.input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i], max_value=relu_max) for i in range(len(hidden_sizes))
        ])

        self.fclayer = fc_layer(hidden_sizes[len(hidden_sizes)-1], output_size)

        self.network = torch.nn.Sequential(*self.hidden_layers, self.fclayer)

        self.side_nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                exp_unit(3, side_net_sizes[0], max_value=relu_max),
                exp_unit(side_net_sizes[0], side_net_sizes[1], max_value=relu_max),
                fc_layer(side_net_sizes[1], 1)
            ) for _ in range(output_size)
        ])

    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
        main_out = self.fclayer(x)
        side_outs = [side_net(x[i:i+3]) for i, side_net in enumerate(self.side_nets)]
        out = main_out + torch.cat(side_outs, dim=1)

        return out

        #return self.network(x)


class SameStructure(torch.nn.Module):
    def __init__(self,
                 input_size: int, 
                 mono_size: int,
                 mono_feature,
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = (),
                 exp_unit: ActivationLayer = ReLUUnit,
                 relu_unit: ActivationLayer = ReLUUnit,
                 conf_unit: ActivationLayer = ReLUUnit,
                 fully_connected_layer: ActivationLayer = FCLayer_notexp):

        super(SameStructure,self).__init__()
        self.input_size = input_size
        self.mono_size = mono_size
        self.non_mono_size = input_size - mono_size
        self.mono_feature = mono_feature
        self.non_mono_feature = list(set(list(range(input_size))).difference(mono_feature))

        self.exp_unit_size = exp_unit_size   
        self.relu_unit_size = relu_unit_size  
        self.conf_unit_size = conf_unit_size 

        self.exp_units = torch.nn.ModuleList([
            exp_unit(mono_size if i == 0 else exp_unit_size[i-1] + conf_unit_size[i-1], exp_unit_size[i])
            for i in range(len(exp_unit_size))
        ])

        self.relu_units = torch.nn.ModuleList([
            relu_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], relu_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.conf_units = torch.nn.ModuleList([
            conf_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], conf_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.fclayer = fully_connected_layer(exp_unit_size[len(exp_unit_size)-1] + conf_unit_size[len(exp_unit_size)-1] + relu_unit_size[len(relu_unit_size)-1],1)
        
    def forward(self,x):

        x_mono = x[:, self.mono_feature]
        x_non_mono = x[:, self.non_mono_feature]

        for i in range(len(self.exp_unit_size)):
            if i == 0 :
                exp_output = self.exp_units[i](x_mono)
                conf_output = self.conf_units[i](x_non_mono)
                relu_output = self.relu_units[i](x_non_mono)
                exp_output = torch.cat([exp_output, conf_output], dim=1)
            else :
                exp_output = self.exp_units[i](exp_output)
                conf_output = self.conf_units[i](relu_output)
                relu_output = self.relu_units[i](relu_output)
                exp_output = torch.cat([exp_output, conf_output], dim=1)

        out = self.fclayer(torch.cat([exp_output,relu_output],dim = 1)) 
        return out
    


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: Tuple = (),
                 hidden_layer: ActivationLayer = ReLUUnit,
                 fully_connected_layer: ActivationLayer = FCLayer_notexp):

        super(MultiLayerPerceptron,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size   

        self.hidden_layers = torch.nn.ModuleList([
            hidden_layer(self.input_size if i == 0 else hidden_size[i-1], hidden_size[i])
            for i in range(len(hidden_size))
        ])

        self.fclayer = fully_connected_layer(hidden_size[len(hidden_size)-1] ,output_size)

    def forward(self,x):

        for i in range(len(self.hidden_size)):
            if i == 0 :
                output = self.hidden_layers[i](x)
            else :
                output = self.hidden_layers[i](output)          

        out = self.fclayer(output)
        return out


