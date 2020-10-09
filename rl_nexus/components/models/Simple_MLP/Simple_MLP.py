import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Simple_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation, output_transform = 'softmax'):
        super(Simple_MLP, self).__init__()
        # nn.Module.__init__(self)
        # fcnet_hiddens = spec_tree['fcnet_hiddens']
        # fcnet_activation = spec_tree['fcnet_activation']
        # hidden_layers = hid
        # self.temperature = spec_tree['temperature']
        # num_outputs = action_space.n
        self.output_transform = output_transform
        layers = []
        input_layer = nn.Linear(input_dim, hidden_layers[0])
        if activation == 'tanh':
            activation_layer = nn.Tanh()
        elif activation == 'relu':
            activation_layer = nn.ReLU()

        layers.append(input_layer)
        layers.append(activation_layer)

        for i in range(len(hidden_layers)-1):
            hidden = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            layers.append(hidden)
            layers.append(activation_layer)
        output_layer = nn.Linear(hidden_layers[-1], output_dim)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.model(x)
        if self.output_transform == 'softmax':
            x = F.softmax(x, dim = -1)
        elif self.output_transform == 'logexp':
            x = torch.log(1+torch.exp(x))
        return x
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))




        

