import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.utils.ope_utils import choose_estimate_from_sequence
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
from torch.nn.utils import spectral_norm
import pdb

dtype = torch.float

class Model_Core(nn.Sequential):
    def __init__(self, input_dim, hidden_layers, activation):
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
        # layers.append(nn.BatchNorm1d(num_features = hidden_layers[-1]))
        super(Model_Core, self).__init__(*layers)

class Dynamics(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_layers, activation):
        super(Dynamics, self).__init__()
        self.feature_extractor = Model_Core(input_dim, hidden_layers, activation)
        # self.next_feature = spectral_norm(nn.Linear(hidden_layers[-1]*num_actions, hidden_layers[-1]*num_actions, bias = False))
        self.next_feature = nn.Linear(hidden_layers[-1]*num_actions, hidden_layers[-1]*num_actions, bias = False)
        self.reward_predictor = nn.Linear(hidden_layers[-1]*num_actions, 1, bias = False)
        self.feature_dim = hidden_layers[-1]
        self.num_actions = num_actions
        
    def forward(self, obs, one_hot_action):
        feature = self.feature_extractor(obs)
        expanded_feature = torch.einsum('ab,ac->abc', feature, one_hot_action).permute(0,2,1).contiguous().view(-1, self.feature_dim*self.num_actions)
        next_feature = self.next_feature(expanded_feature)
        reward = self.reward_predictor(expanded_feature)
        return next_feature, reward

class Model_Based():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon, 
                policy_net, hidden_layers, activation, 
                norm = 'std', use_delayed_target = False,
                keep_terminal_states = True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.use_delayed_target = use_delayed_target
        self.n_episode =  dataset['init_obs'].shape[0]
        if keep_terminal_states:
            self.included_idx = torch.arange(dataset['obs'].shape[0])
            self.end_idx = np.arange(self.horizon-1, dataset['obs'].shape[0], self.horizon)
            self.absorbing_idx = np.where(dataset['info'][:,0] == True)[0]
            # self.absorbing_idx = np.array([])
        else:
            pass
        self.n_samples = self.included_idx.shape[0]
        self.non_absorbing_mask = torch.ones(self.n_samples, dtype=torch.bool)
        self.non_absorbing_mask[self.absorbing_idx] = False

        self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]
        if self.policy_net is not None:
            raise NotImplementedError
        else:
            self.pi_current = torch.tensor(dataset['target_prob_obs'],dtype=dtype)[self.included_idx]
            self.pi_next = torch.tensor(dataset['target_prob_next_obs'], dtype=dtype)[self.included_idx]
            self.pi_init = torch.tensor(dataset['target_prob_init_obs'], dtype=dtype)
            self.pi_term = torch.tensor(dataset['target_prob_term_obs'], dtype=dtype)
        if self.norm == 'std':
            raise NotImplementedError
        else:
            self.obs = torch.tensor(dataset['obs'], dtype = dtype)[self.included_idx]
            self.next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
            self.init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            self.term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)
        #* whiten the included observation data
        obs_mean = torch.mean(self.obs, dim=0, keepdims= True)
        obs_std = torch.std(self.obs, dim=0, keepdims= True)
        self.obs = (self.obs - obs_mean) / obs_std
        self.next_obs = (self.next_obs - obs_mean) / obs_std
        self.init_obs = (self.init_obs - obs_mean) / obs_std
        self.term_obs = (self.term_obs - obs_mean) / obs_std

        # self.feature_extractor = Model_Core(input_dim = self.obs_dim, hidden_layers = hidden_layers, activation= activation)
        self.dynamics = Dynamics(input_dim= self.obs_dim, num_actions = self.act_dim, hidden_layers=hidden_layers,\
            activation=activation)

    def train(self, num_iter = 2000, lr = 1.0e-3, batch_size = 500, tail_average=10, reg = 1e-3):
        optimizer = optim.Adam(self.dynamics.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-4)
        # optimizer = optim.RMSprop(self.dynamics.parameters(), lr = 0.01)
        if not batch_size:
            batch_size = self.n_samples #* use the whole batch if no batchsize declared
        feature_dim =  self.hidden_layers[-1]
        one_hot_acts = torch.squeeze(F.one_hot(self.data_acts, num_classes = self.act_dim))
        I_sa = torch.eye(feature_dim*self.act_dim)
        min_loss = np.inf
        value_est_list = []
        for i in range(num_iter):
            perm = torch.randperm(self.n_samples)
            num_batches = self.n_samples // batch_size
            current_loss = 0
            for j in range(num_batches):
                idx = perm[j*batch_size:(j+1)*batch_size]
                obs = self.obs[idx]
                acts = one_hot_acts[idx]
                next_obs = self.next_obs[idx]
                pi_next = self.pi_next[idx]
                rews = self.rews[idx]

                next_feature, reward = self.dynamics(obs, acts)
                Z_prime = self.dynamics.feature_extractor(next_obs).detach()
                phi_prime = torch.einsum('ab,ac->abc', Z_prime, pi_next).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
                model_loss = F.mse_loss(next_feature, phi_prime)
                reward_loss = F.mse_loss(reward, rews)
                loss = model_loss + reward_loss
                optimizer.zero_grad()
                loss.backward()
                # for param in self.dynamics.parameters():
                #     print(param.grad.data.max())
                optimizer.step()
                current_loss += loss.clone().detach()
            current_loss /= num_batches
            if current_loss < min_loss:
                min_loss = current_loss.clone().detach()
                print('iter {} NEW MIN LOSS: '.format(i), min_loss.numpy())
                #* evaluate
                with torch.no_grad():
                    P = self.dynamics.next_feature.weight
                    reward_feature = self.dynamics.reward_predictor.weight
                    Z_init = self.dynamics.feature_extractor(self.init_obs)
                    phi_init = torch.einsum('ab,ac->abc', Z_init , self.pi_init).permute(0,2,1).contiguous().view(self.n_episode, feature_dim*self.act_dim)
                    finite_horizon_correction = I_sa - torch.matrix_power(self.gamma*P.T, self.horizon)
                    transposed_transition_inverse = torch.inverse(I_sa - self.gamma*P.T)
                    accumulated_feature = phi_init @ finite_horizon_correction @ transposed_transition_inverse
                    V = accumulated_feature @ reward_feature.T
                    value_est = torch.mean(V).numpy()
                    value_est_list.append(value_est)
                    print('latest estimate: ', value_est)
                    print('\n')

            else:
                print('iter {} current loss: '.format(i), current_loss.clone().detach().numpy())
            # roll forward evaluation
            # if i % 1 == 0:
            #     #* get initial observation
            #     with torch.no_grad():
            #         P = self.dynamics.next_feature.weight
            #         reward_feature = self.dynamics.reward_predictor.weight
            #         Z_init = self.dynamics.feature_extractor(self.init_obs)
            #         phi_init = torch.einsum('ab,ac->abc', Z_init , self.pi_init).permute(0,2,1).contiguous().view(self.n_episode, feature_dim*self.act_dim)
            #         # #* we will roll forward for horizon steps
            #         # current_phi = phi_init.clone()
            #         # discount = 1
            #         # accumulated_value = torch.zeros(self.n_episode,1, requires_grad = False)
            #         # for t in range(self.horizon):
            #         #     current_reward = self.dynamics.reward_predictor(current_phi)
            #         #     accumulated_value += discount * current_reward
            #         #     next_phi = self.dynamics.next_feature(current_phi)
            #         #     discount *= self.gamma
            #         #     current_phi = next_phi.clone()

            #         finite_horizon_correction = I_sa - torch.matrix_power(self.gamma*P.T, self.horizon)
            #         transposed_transition_inverse = torch.inverse(I_sa - self.gamma*P.T)
            #         accumulated_feature = phi_init @ finite_horizon_correction @ transposed_transition_inverse
            #         V = accumulated_feature @ reward_feature.T
            #     print('current estimate: ', torch.mean(V).numpy())
            #     print('\n')
            #     # all_acts = torch.arange(self.act_dim).repeat(self.n_episode,1)
            #     # init_feature = self.dynamics()
            #     if i % 10 == 0:
            #         pdb.set_trace()

        # pdb.set_trace()
        return np.mean(value_est_list[-10:])



