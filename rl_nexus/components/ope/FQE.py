import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.utils.ope_utils import choose_estimate_from_sequence
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
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
        super(Model_Core, self).__init__(*layers)

class FQE():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon, 
                policy_net, hidden_layers, activation, 
                norm = 'std', use_delayed_target = False,
                keep_terminal_states = True, debug = True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.hidden_layers = hidden_layers
        self.use_delayed_target = use_delayed_target
        self.n_episode =  dataset['init_obs'].shape[0]
        if keep_terminal_states:
            self.included_idx = torch.arange(dataset['obs'].shape[0])
            self.end_idx = np.arange(self.horizon-1, dataset['obs'].shape[0], self.horizon)
            # self.absorbing_idx = np.where(dataset['info'][:,0] == True)[0]
            self.absorbing_idx = np.array([])
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
        if use_delayed_target:
            self.q_net = Simple_MLP(input_dim = self.obs_dim, output_dim = self.act_dim, hidden_layers = hidden_layers,\
                activation= activation, output_transform = None)
        else:
            self.q_net = Model_Core(input_dim = self.obs_dim, hidden_layers = hidden_layers, activation= activation)
        self.debug = debug
    def train(self, num_iter = 1000, lr = 1.0e-3, batch_size = 500, tail_average=10, reg = 1e-3):
        if self.use_delayed_target:
            value_est = self.train_delayed_target(num_iter, lr, batch_size, tail_average, reg)
        else:
            value_est = self.train_non_linear(num_iter, lr, batch_size, tail_average, reg)
        return value_est
    
    def train_delayed_target(self, num_iter = 1000, lr = 1.0e-3, batch_size = 500, tail_average=10,\
        reg = 1e-3, use_separate_target_net=False):
        optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay = reg)
        value_est_list = []
        if not batch_size:
            batch_size = self.n_samples #* use the whole batch if no batchsize declared

        for i in range(num_iter):
            # decayed_lr = lr / np.sqrt(i+1)
            # for param_group in optimizer_q.param_groups:
            #     param_group['lr'] = decayed_lr
            perm = torch.randperm(self.n_samples)
            num_batches = self.n_samples // batch_size
            for j in range(num_batches):
                idx = perm[j*batch_size:(j+1)*batch_size]
                obs = self.obs[idx]
                acts = self.data_acts[idx]
                next_obs = self.next_obs[idx]
                pi_next = self.pi_next[idx]
                rews = self.rews[idx]

                non_absorbing_mask = self.non_absorbing_mask[idx]
                state_action_values = self.q_net(obs).gather(1, acts)
                next_state_values = torch.zeros(batch_size,1, dtype=dtype)
                non_absorbing_next_states = next_obs[non_absorbing_mask]
                if use_separate_target_net:
                    next_state_values[non_absorbing_mask] = \
                        (self.q_net_target(non_absorbing_next_states)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims = True).detach()
                else:
                    next_state_values[non_absorbing_mask] = \
                        (self.q_net(non_absorbing_next_states)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims = True).detach()
                expected_state_action_values = (next_state_values * self.gamma) + rews
                # Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                optimizer_q.zero_grad()
                loss.backward()
                for param in self.q_net.model.parameters():
                    param.grad.data.clamp_(-1,1)
                optimizer_q.step()
            if use_separate_target_net:                
                self.q_net_target.load_state_dict(self.q_net.state_dict())
            if i % 10  == 0:
                q_s0 = self.q_net(self.init_obs)
                q_s0_pi = (q_s0 * self.pi_init).sum()
                q_sterm = self.q_net(self.term_obs)
                q_sterm_pi = (q_sterm * self.pi_term).sum()
                value_est = (q_s0_pi - self.gamma**self.horizon*q_sterm_pi) / self.n_episode
                value_est_list.append(value_est.detach().numpy())
            if i %10 == 0 and i>0 and self.debug:
                print('\n')
                print('iter {} Trailing estimate: '.format(i), np.mean(value_est_list[-10:]))
                print('loss {}'.format(loss.detach().numpy()))
        final_value_estimate = choose_estimate_from_sequence(value_est_list)        
        return final_value_estimate




    def train_non_linear(self, num_iter = 1000, lr = 1.0e-3, batch_size = 500, tail_average=10, reg = 1e-3):
        optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.0,0.99), eps=1e-8)
        value_est_list = []

        feature_dim =  self.hidden_layers[-1]
        I = torch.eye(self.act_dim*feature_dim)
        min_loss = np.inf
        if not batch_size:
            batch_size = self.n_samples #* use the whole batch if no batchsize declared
        # pdb.set_trace()
        for i in range(num_iter):
            decayed_lr = lr / np.sqrt(i+1)
            for param_group in optimizer_q.param_groups:
                param_group['lr'] = decayed_lr
            perm = torch.randperm(self.n_samples)
            num_batches = self.n_samples // batch_size
            current_iter_loss = 0
            for j in range(num_batches):
                idx = perm[j*batch_size:(j+1)*batch_size]
                obs = self.obs[idx]
                acts = self.data_acts[idx]
                next_obs = self.next_obs[idx]
                pi_next = self.pi_next[idx]
                rews = self.rews[idx]
                non_absorbing_mask = self.non_absorbing_mask[idx]
                
                #* extract the non-linear features
                Z =  self.q_net(obs)
                Z_prime = self.q_net(next_obs)
                #* solve for the last linear layer using least square regression
                Phi = torch.zeros(batch_size, feature_dim*self.act_dim, dtype=dtype)
                Phi_prime_pi = torch.zeros(batch_size, feature_dim*self.act_dim, dtype=dtype)
                for a in range(self.act_dim):
                    act_idx = torch.where(acts == a)[0]
                    Phi[act_idx, a*feature_dim:(a+1)*feature_dim] = Z[act_idx]
                    Phi_prime_pi[:, a*feature_dim:(a+1)*feature_dim] = pi_next[:, a][:,None] * Z_prime
                regularized_inverse = torch.inverse( torch.mm(Phi.T, Phi-self.gamma*Phi_prime_pi) + reg*I)
                featurized_reward = torch.mm(Phi.T, rews)
                linear_coeff = torch.mm(regularized_inverse, featurized_reward)

                #* Now that we solve the linear layer, form the loss function
                linear_layer = linear_coeff.view(-1, feature_dim).permute(1,0)
                state_action_values = (Z @ linear_layer).gather(1, acts)
                next_state_values = torch.zeros(batch_size,1, dtype=dtype)
                next_state_values[non_absorbing_mask] = ((Z_prime @ linear_layer)*pi_next).sum(dim=1, keepdims = True)[non_absorbing_mask]
                expected_state_action_values = (next_state_values * self.gamma) + rews

                # Huber loss or MSE loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                # loss = F.mse_loss(state_action_values, expected_state_action_values)
                current_iter_loss += loss.clone().detach()
                
                #* differentiate and update 
                optimizer_q.zero_grad()
                loss.backward()
                for param in self.q_net.parameters():
                    param.grad.data.clamp_(-1,1)
                optimizer_q.step()
            #* see if total loss is the best so far, if so, record the estimate
            if current_iter_loss < min_loss:
                min_loss = current_iter_loss
                Z_init = self.q_net(self.init_obs)
                Z_term = self.q_net(self.term_obs)
                q_s0 = (Z_init @ linear_layer)
                q_s0_pi = ( q_s0 * self.pi_init).sum()
                q_sterm = (Z_term @ linear_layer)
                q_sterm_pi = ( q_sterm * self.pi_term).sum()
                value_est = (q_s0_pi - self.gamma**self.horizon * q_sterm_pi) / self.n_episode
                value_est_list.append(value_est.detach().numpy())
                max_grad = 0
                max_q_weight = 0
                if self.debug:
                    for param in self.q_net.parameters():
                        # param.grad.data.clamp_(-1,1)
                        if param.grad is not None:
                            max_grad = max(max_grad, param.grad.data.max())
                        max_q_weight = max(max_q_weight, param.data.max())
                    print('\n')
                    print('iter {} Trailing estimate: '.format(i), np.mean(value_est_list[-tail_average:]))
                    print('current estimate: ', value_est)
                    print('current loss: ', current_iter_loss.detach().numpy())
                    print('max linear weight:', linear_layer.max())
                    print('max q gradient:', max_grad)
                    print('max q weight:', max_q_weight)
            if i % 100  == 0 and self.debug:
                print('Current loss in iter {}: {:4f}'.format(i, current_iter_loss.numpy()))
        return np.mean(value_est_list[-tail_average:])











