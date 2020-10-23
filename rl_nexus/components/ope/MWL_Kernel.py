import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
import pdb

dtype = torch.float

class MWL():
    def __init__(self, dataset, obs_dim, act_dim, k_tau, norm = None,
                hidden_layers = [64,64], activation = 'relu', 
                policy_net = None, keep_terminal_states = True,
                action_encoding_scheme = 'continuous',
                lr = 5e-3, reg_factor = 0, gamma = 0.99):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.reg_factor = reg_factor
        self.hidden_layers = hidden_layers
        self.k_tau = k_tau
        self.norm = norm
        self.policy_net = policy_net

        if keep_terminal_states:
            self.included_idx = torch.arange(dataset['obs'].shape[0])
        else:
            raise NotImplementedError
        
        self.n_episode =  dataset['init_obs'].shape[0]
        self.n_samples = self.included_idx.shape[0]
        
        self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]

        if norm == 'std':
            raise NotImplementedError
        elif norm is None:
            self.obs = torch.tensor(dataset['obs'], dtype=dtype)[self.included_idx]
            self.next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
            self.init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            self.term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)
            self.done = torch.tensor(dataset['done'], dtype=torch.bool)[self.included_idx]

        if action_encoding_scheme == 'continuous':
            encoded_actions = np.linspace(-1,1, self.act_dim)
            mean_action = np.mean(encoded_actions[self.data_acts])
            std_action = np.std(encoded_actions[self.data_acts])
            self.encoded_actions = (encoded_actions - mean_action)/ std_action
            self.act_input = self.encoded_actions[self.data_acts]
        else:
            raise NotImplementedError
        
        if action_encoding_scheme == 'continuous':
            self.obs_act = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
            