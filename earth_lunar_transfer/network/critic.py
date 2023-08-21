import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNN(nn.Module):
    def __init__(self, input_shape, action_shape,
                 learning_rate, tau=0.001, init_w=3e-3):
        super(CriticNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        policy_shape = 1
        num_neurons = [input_shape+action_shape,
                       64, 32, 16,
                       policy_shape]
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(num_neurons)-1):
            layer = nn.Linear(num_neurons[i], num_neurons[i+1])
            self.layers.append(layer)
            activation = nn.ReLU()
            self.layers.append(activation)
        
    def forward(self, x):
        x = torch.cat(x)
        policy = self.layers[0](x)
        for layer in self.layers[1:]:
            policy = layer(policy)
        return policy