import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
import pdb

dtype = torch.float
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_layers, activation):
        # super(MLP, self).__init__()
        # nn.Module.__init__(self)
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
        
        # self = nn.Sequential(*layers)
        super(MLP, self).__init__(*layers)


class MQL():
    def __init__(self, dataset, obs_dim, act_dim, k_tau, seed = 1,norm = None, 
                policy_net = None, input_mode = 'sa', action_encoding_scheme = 'continuous',
                keep_terminal_states = True,
                hidden_layers = [64,64], activation = 'relu', lr = 5e-3, reg_factor=0,gamma = 0.99):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.reg_factor = reg_factor
        self.hidden_layers = hidden_layers
        self.k_tau = k_tau
        self.norm = norm
        
        # TODO: maybe allow for the option to load the policy / q network in
        # TODO: for now, we mainly assume the dataset already contains the probabilities for s'
        # TODO: also, we will assume that the input to network will be (s,a), with a continuous encoding
        self.policy_net = policy_net
        self.input_mode = input_mode
        if keep_terminal_states:
            self.included_idx = torch.arange(dataset['obs'].shape[0])
        else:
            raise NotImplementedError

        self.n_episode =  dataset['init_obs'].shape[0]
        self.n_samples = self.included_idx.shape[0]
        
        self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]
        #* Load in the target probabilities (compute it if the policy net is given as an argument)
        if self.policy_net is not None:
            self.pi_current = torch.tensor(self.policy_net.get_probabilities(dataset['obs']), dtype=dtype)[self.included_idx]
            self.pi_next = torch.tensor(self.policy_net.get_probabilities(dataset['next_obs']), dtype=dtype)[self.included_idx]
            self.pi_init = torch.tensor(self.policy_net.get_probabilities(dataset['init_obs']), dtype=dtype)
            self.pi_term = torch.tensor(self.policy_net.get_probabilities(dataset['term_obs']), dtype=dtype)
        else:
            self.pi_current = torch.tensor(dataset['target_prob_obs'],dtype=dtype)[self.included_idx]
            self.pi_next = torch.tensor(dataset['target_prob_next_obs'], dtype=dtype)[self.included_idx]
            self.pi_init = torch.tensor(dataset['target_prob_init_obs'], dtype=dtype)
            self.pi_term = torch.tensor(dataset['target_prob_term_obs'], dtype=dtype)
        if self.norm == 'std':
            self.obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            self.obs_std = np.std(dataset['obs'], axis=0, keepdims=True)
            self.obs = torch.tensor((dataset['obs'] - self.obs_mean) / self.obs_std, dtype=dtype)[self.included_idx]
            self.next_obs = torch.tensor((dataset['next_obs'] - self.obs_mean) / self.obs_std, dtype=dtype)[self.included_idx]
            self.init_obs = (dataset['init_obs'] - self.obs_mean) / self.obs_std
            self.term_obs = (dataset['term_obs'] - self.obs_mean) / self.obs_std
        elif self.norm is None:
            self.obs = torch.tensor(dataset['obs'], dtype=dtype)[self.included_idx]
            self.next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
            self.init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            self.term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)
        else:
            raise NotImplementedError
        if action_encoding_scheme == 'continuous':
            encoded_actions = np.linspace(-1,1, self.act_dim)
            mean_action = np.mean(encoded_actions[self.data_acts])
            std_action = np.std(encoded_actions[self.data_acts])
            self.encoded_actions = (encoded_actions - mean_action)/ std_action
            self.act_input = self.encoded_actions[self.data_acts]
        else:
            raise NotImplementedError
        #* set-up networks
        if self.input_mode == 'sa':
            self.obs_act = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
            assert action_encoding_scheme == 'continuous'
            # self.q_net = Simple_MLP(input_dim = self.obs_dim, output_dim = self.act_dim, hidden_layers = hidden_layers,\
            #     activation= activation, output_transform = None)
            self.q_net = MLP(input_dim = self.obs_dim, output_dim = self.act_dim, hidden_layers = hidden_layers,\
                activation= activation)
            # self.q_net = nn.Sequential(
            #                 nn.Linear(4,64),
            #                 nn.ReLU(),
            #                 nn.Linear(64,64),
            #                 nn.ReLU(),
            #                 nn.Linear(64,2)
            #             )
        else:
            raise NotImplementedError

    def estimate_median_distance(self, num_sample = None):
        index_set_1 = torch.randperm(self.n_samples)[:num_sample]
        index_set_2 = torch.randperm(self.n_samples)[:num_sample]
        obs_act_1 = self.obs_act[index_set_1]
        obs_act_2 = self.obs_act[index_set_2]
        return torch.median(torch.sqrt(torch.sum(torch.square(obs_act_1[None, :, :] - obs_act_2[:, None, :]), dim = -1)))


    def train(self, num_iter, lr, batch_size, q_reg = 0.0, eval_interval = 10, detach_target = True):
        # optimizer = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay=q_reg)
        # pdb.set_trace()
        optimizer = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8)
        med_dist = self.estimate_median_distance(num_sample = 5000) / self.k_tau
        n_samples_batch = batch_size**2

        value_est_list = []
        # l2_crit = nn.MSELoss(reduction = 'mean')
        for i in range(num_iter):
            #* sample a mini-batch of data
            index_set_1 = torch.randperm(self.n_samples)[:batch_size]
            index_set_2 = torch.randperm(self.n_samples)[:batch_size]
            
            obs_1 = self.obs[index_set_1]; obs_2 = self.obs[index_set_2]
            act_1 = self.data_acts[index_set_1]; act_2 = self.data_acts[index_set_2]
            next_obs_1 = self.next_obs[index_set_1]; next_obs_2 = self.next_obs[index_set_2]
            rew_1 = self.rews[index_set_1]; rew_2 = self.rews[index_set_2]
            pi_next_1 = self.pi_next[index_set_1]; pi_next_2 = self.pi_next[index_set_2]
            obs_act_1 = self.obs_act[index_set_1]; obs_act_2 = self.obs_act[index_set_2]

            #* Forming the loss function for this batch of samples
            #* There are 3 main components: bellman error for index set 1, bellman error
            #* for index set 2, and the kernel matrix evaluated on cross index set
            q_sa_1 = self.q_net(obs_1).gather(1,act_1)
            q_sa_2 = self.q_net(obs_2).gather(1,act_2)
            q_sn_pi_1 = (self.q_net(next_obs_1) * pi_next_1).sum(dim=1, keepdims = True)
            q_sn_pi_2 = (self.q_net(next_obs_2) * pi_next_2).sum(dim=1, keepdims = True)

            # with torch.no_grad():
                
            #     q_sn_pi_1 = (self.q_net(next_obs_1) * pi_next_1).sum(dim=1, keepdims = True)
            #     q_sn_pi_2 = (self.q_net(next_obs_2) * pi_next_2).sum(dim=1, keepdims = True)

            # q_sa_2 = self.q_net(obs_2).detach().gather(1,act_2)
            # # q_sn_pi[self.non_absorbing_mask] = (self.q_net(non_absorbing_sn)*self.pi_next[self.non_absorbing_mask]).sum(dim=1, keepdims=True)
            # if detach_target:
            #     q_sn_pi_1 = (self.q_net(next_obs_1).detach() * pi_next_1).sum(dim=1, keepdims = True)
            #     q_sn_pi_2 = (self.q_net(next_obs_2).detach() * pi_next_2).sum(dim=1, keepdims = True)
            # else:
            #     q_sn_pi_1 = (self.q_net(next_obs_1) * pi_next_1).sum(dim=1, keepdims = True)
            #     q_sn_pi_2 = (self.q_net(next_obs_2) * pi_next_2).sum(dim=1, keepdims = True)
            error_1 = rew_1 + self.gamma * q_sn_pi_1 - q_sa_1
            error_2 = rew_2 + self.gamma * q_sn_pi_2 - q_sa_2
            diff = obs_act_1.unsqueeze(1) - obs_act_2.unsqueeze(0)
            K = torch.exp(-torch.sum(torch.square(diff), dim = -1) / 2.0 / med_dist**2 )
            loss = torch.squeeze(error_1.T @ K @ error_2)
            loss /= n_samples_batch
            # pdb.set_trace()
            # reg_loss = l2_crit(self.q_net[-1].weight.view(-1), torch.zeros(128))
            reg_loss = torch.norm(self.q_net[-1].weight)**2 + torch.norm(self.q_net[-1].bias)**2
            loss += q_reg * reg_loss

            # l2_norm = torch.norm(self.q_net.model[-1].weight)
            # pdb.set_trace()
            # loss += 0.5*l2_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                q_s0 = self.q_net(self.init_obs)
                q_s0_pi = (q_s0 * self.pi_init).sum()
                value_est = (q_s0_pi) / self.n_episode
                value_est_list.append(value_est.detach().numpy())
                
                print('Iter: {}. Loss current: {:.4f}. MQL Estimate: {:.2f}'.format(i, loss.detach().numpy(), value_est))
                print('max weight last layer: ', self.q_net[-1].weight.max())

                # pdb.set_trace()
        pdb.set_trace()
        return value_est
                







        
        