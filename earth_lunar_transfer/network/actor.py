import torch
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorNN(nn.Module):
    def __init__(self, input_shape, action_shape,
                 learning_rate, tau=0.001, init_w=3e-3):
        super(ActorNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        num_neurons = [input_shape,
                       64, 32,
                       action_shape]
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(num_neurons)-1):
            layer = nn.Linear(num_neurons[i], num_neurons[i+1])
            self.layers.append(layer)
            activation = nn.ReLU() if i < len(num_neurons)-2 else nn.Tanh()
            self.layers.append(activation)
        
    def forward(self, x):
        action = self.layers[0](x)
        for layer in self.layers[1:]:
            action = layer(action)
        return action
