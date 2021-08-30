import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, state_size, seed, hidden_units=(400, 300), activation=F.relu, final_activation=F.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            activation (function): activation function
            final_activation (function): final activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation = activation
        self.final_activation = final_activation
        self.states_normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.bn = nn.ModuleList([nn.BatchNorm1d(dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):        
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.states_normalizer(states)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.activation(self.bn[i](layer(x)))
            else:
                x = self.activation(layer(x))
        return self.final_activation(self.output(x))

    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size, state_size, seed, hidden_units=(400, 300), activation=F.relu, dropout=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            activation (function): activation function
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.states_normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1: self.layers.append(nn.Linear(dim_in+action_size, dim_out))
            else: self.layers.append(nn.Linear(dim_in, dim_out))
        self.bn = nn.ModuleList([nn.BatchNorm1d(dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.states_normalizer(states)
        xs = self.activation(self.bn[0](self.layers[0](xs)))
        x = torch.cat((xs, actions), dim=1)
        for i in range(1, len(self.layers)):
            if i < len(self.layers) - 1:
                x = self.activation(self.bn[i](self.layers[i](x)))
            else:
                x = self.activation(self.layers[i](x))
        x = self.dropout(x)
        return self.output(x)
