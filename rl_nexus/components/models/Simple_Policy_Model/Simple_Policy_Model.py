import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Simple_Policy_Model(nn.Module):
    def __init__(self, spec_tree, obs_space, action_space):
        super(Simple_Policy_Model, self).__init__()
        # nn.Module.__init__(self)
        fcnet_hiddens = spec_tree['fcnet_hiddens']
        fcnet_activation = spec_tree['fcnet_activation']
        self.temperature = spec_tree['temperature']
        num_outputs = action_space.n
        
        layers = []
        input_layer = nn.Linear(int(np.product(obs_space.shape)), fcnet_hiddens[0])
        if fcnet_activation == 'tanh':
            activation_layer = nn.Tanh()
        elif fcnet_activation == 'relu':
            activation_layer = nn.ReLU()

        layers.append(input_layer)
        layers.append(activation_layer)

        for i in range(len(fcnet_hiddens)-1):
            hidden_layer = nn.Linear(fcnet_hiddens[i], fcnet_hiddens[i+1])
            layers.append(hidden_layer)
            layers.append(activation_layer)
        output_layer = nn.Linear(fcnet_hiddens[-1], num_outputs)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.model(x)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def sample_action(self, obs):
        # import pdb; pdb.set_trace()
        action_probs = F.softmax(self.model(torch.tensor(obs, dtype=torch.float))/self.temperature, dim=-1)
        # action = torch.multinomial(action_probs, action_probs.shape[0])
        action = torch.squeeze(torch.multinomial(action_probs, 1))
        # import pdb; pdb.set_trace()
        return action.data.numpy()
    
    def sample_action_with_prob(self, obs):
        action_probs = F.softmax(self.model(torch.tensor(obs, dtype=torch.float))/self.temperature, dim=-1)
        # action = torch.multinomial(action_probs, action_probs.shape[0])
        action = torch.squeeze(torch.multinomial(action_probs, 1))
        action_prob = action_probs[action]
        # import pdb; pdb.set_trace()
        # return action.data.numpy()[0], action_prob.data.numpy()[0]
        return action.data.numpy(), action_prob.data.numpy()
    
    def get_probabilities(self, obs):
        action_probs = F.softmax(self.model(torch.tensor(obs, dtype=torch.float))/self.temperature, dim=-1)
        return action_probs.data.numpy()
    
    def get_prob_with_act(self, obs, act):
        action_probs = F.softmax(self.model(torch.tensor(obs, dtype=torch.float))/self.temperature, dim=-1)
        return action_probs.gather(1,torch.tensor(act)).data.numpy()






        

