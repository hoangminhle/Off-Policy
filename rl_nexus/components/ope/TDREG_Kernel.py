import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP

dtype = torch.double
torch.set_default_dtype(dtype)

# class W_network(nn.Module):
#     def __init__(self, input_dim, seed):
#         super(W_network, self).__init__()
#         self.seed = seed
#         torch.manual_seed(self.seed)
#         self.fc1 = nn.Linear(input_dim, 32)
#         self.fc2 = nn.Linear(32,32)
#         self.fc3 = nn.Linear(32, 1)
                
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.log(1+torch.exp(self.fc3(x)))
#         return x

class TDREG_Kernel(object):
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon,
                policy_net, value_reg,
                hidden_layers, activation, output_transform,
                default_length_scale = 0.1,
                random_feature_per_obs_dim = 250,
                norm = 'std',
                scale_length_adjustment = 'median',
                input_mode = 'sa', seed = 1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        # self.model_reg = model_reg
        # self.reward_reg = reward_reg
        self.value_reg = value_reg
        self.input_mode = input_mode

        self.n_samples = dataset['obs'].shape[0]
        self.n_episode =  dataset['init_obs'].shape[0]
        if self.policy_net is not None:
            self.pi_current = self.policy_net.get_probabilities(dataset['obs'])
            self.pi_next = self.policy_net.get_probabilities(dataset['next_obs'])
            self.pi_init = self.policy_net.get_probabilities(dataset['init_obs'])
            self.pi_term = self.policy_net.get_probabilities(dataset['term_obs'])
        else:
            self.pi_current = dataset['target_prob_obs']
            self.pi_next = dataset['target_prob_next_obs']
            self.pi_init = dataset['target_prob_init_obs']
            self.pi_term = dataset['target_prob_term_obs']
        if self.norm == 'std':
            self.obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            self.obs_std = np.std(dataset['obs'], axis=0, keepdims=True)
            self.obs = (dataset['obs'] - self.obs_mean) / self.obs_std
            self.next_obs = (dataset['next_obs'] - self.obs_mean) / self.obs_std
            self.init_obs = (dataset['init_obs'] - self.obs_mean) / self.obs_std
            self.term_obs = (dataset['term_obs'] - self.obs_mean) / self.obs_std
        elif self.norm is None:
            self.obs = dataset['obs']
            self.next_obs = dataset['next_obs']
            self.init_obs = dataset['init_obs']
            self.term_obs = dataset['term_obs']
        else:
            raise NotImplementedError
        if scale_length_adjustment == 'median':
            sample_num = 5000
            idx1 = np.random.choice(self.n_samples, sample_num); idx2 = np.random.choice(self.n_samples, sample_num)
            med_dist = np.median(np.square(self.obs[None, idx1, :] - self.obs[idx2, None, :]), axis = (0,1))
            med_dist[med_dist<0.01] = 0.01 # enforce a upperbound on the scale-length of the action component
            scale_length_vector = 1.0/med_dist
        else:
            scale_length_vector = np.ones(self.obs_dim)
        # import pdb; pdb.set_trace()
        #* set the fourier feature
        transformer_list = []
        self.z_dim = random_feature_per_obs_dim * self.obs_dim
        models = [RBFSampler(n_components = random_feature_per_obs_dim, gamma = default_length_scale*dist) for dist in scale_length_vector]
        for model in models:
            model.fit([self.obs[0]])
            transformer_list.append((str(model), model))
        self.rff = FeatureUnion(transformer_list)

        #* separate action set indexing
        act_idx = []
        for i in range(self.act_dim):
            act_idx.append(np.where(dataset['acts']==i)[0])
        #* apply transformation
        Z = self.rff.transform(self.obs); Z_prime = self.rff.transform(self.next_obs)
        Z_init = self.rff.transform(self.init_obs); Z_term = self.rff.transform(self.term_obs)
        assert self.z_dim == Z.shape[1]
        self.Phi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim))
        self.Phi_pi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim))
        self.Phi_prime_pi = np.zeros((Z_prime.shape[0], Z_prime.shape[1]* self.act_dim))
        self.Phi_init_pi = np.zeros((Z_init.shape[0], Z_init.shape[1]*self.act_dim))
        self.Phi_term_pi = np.zeros((Z_term.shape[0], Z_term.shape[1]*self.act_dim))
        for i in range(self.act_dim):
            self.Phi[act_idx[i], i*self.z_dim:(i+1)*self.z_dim] = Z[act_idx[i]]
            self.Phi_pi[:, i*self.z_dim:(i+1)*self.z_dim] = self.pi_current[:,i][:,None] * Z        
            self.Phi_prime_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_next[:,i][:,None] * Z_prime
            self.Phi_init_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_init[:,i][:,None]*Z_init
            self.Phi_term_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_term[:,i][:,None]*Z_term

        #* Some commonly used variables
        self.I_sa = np.eye(self.act_dim*self.z_dim)
        self.rews = dataset['rews']
        self.init_idx = np.arange(0, self.n_samples, self.horizon)
        self.end_idx = np.arange(self.horizon-1, self.n_samples, self.horizon)

        self.rho = dataset['ratio'] #* make sure that the importance weights are already calculated

        #* set-up network
        #! consider representing the actions better
        if self.input_mode == 'sa':
            if self.act_dim == 2:
                acts = dataset['acts'] *2 -1 # turn the actions into [-1,1] for binary action case
                self.x = torch.tensor(np.concatenate((self.obs,acts), axis=1))
                self.w_net = Simple_MLP(input_dim = self.obs_dim+1, output_dim = 1, hidden_layers = hidden_layers,\
                    activation= activation, output_transform = output_transform)
            else:
                raise NotImplementedError
        elif self.input_mode == 's':
            self.x = torch.tensor(self.obs)
            self.w_net = Simple_MLP(input_dim = self.obs_dim, output_dim = 1, hidden_layers = hidden_layers,\
                activation = activation, output_transform = output_transform)

        self.form_td_ball()
        self.prepare_torch_tensor()
    def form_td_ball(self):
        X = np.matmul(self.Phi.T, self.Phi- self.gamma*self.Phi_prime_pi)#+reg*np.eye(z_dim)
        y = np.matmul(self.Phi.T, self.rews)
        # D = np.linalg.inv(np.matmul(Z.T, Z))#+reg*np.eye(z_dim))
        D = self.I_sa
        self.M = np.linalg.inv( X.T @ D @ X +self.value_reg*self.I_sa)
        self.alpha = self.M @ X.T @ D @ y

    def prepare_torch_tensor(self):
        self.M = torch.tensor(self.M)
        self.alpha = torch.tensor(self.alpha)
        self.rews = torch.tensor(self.rews)
        self.rho = torch.tensor(self.rho)
        self.Phi_prime_pi = torch.tensor(self.Phi_prime_pi)
        self.Phi = torch.tensor(self.Phi)
        self.Phi_term_pi = torch.tensor(self.Phi_term_pi)

        #* convert to torch tensor
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        self.discount_factor = torch.tensor(discount_factor)


    def train(self, num_iter = 300, batch_size = 20000, td_ball_epsilon = 0.01, lr = 1e-3, w_reg =0.0, normalize_w = False,\
        use_var_in_loss=True, tail_average=10):
        optimizer = optim.Adam(self.w_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay=w_reg)
        
        idx = torch.arange(self.n_samples)
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)
        Phi_0_pi = torch.tensor(np.mean(self.Phi_init_pi, axis=0)[:,None])
        
        min_loss = 1e6
        w_min = None

        value_est_list = []

        for i in range(num_iter):
            optimizer.zero_grad()
            w = self.w_net(self.x)
            if normalize_w:
                w = w/torch.mean(w)
            #* calculate the 3 components that go into the f_w term
            if self.input_mode == 'sa':
                bias_1 = torch.einsum('ki, kj->ij', self.gamma*self.Phi_prime_pi[idx]-self.Phi[idx], self.discount_factor[idx]*w[idx])/horizon_normalization/self.n_episode
                trajectory_reward = (w*self.rews*self.discount_factor/horizon_normalization).reshape((self.n_episode, self.horizon)).sum(axis=1)
            elif self.input_mode =='s':
                bias_1 = torch.einsum('ki, kj->ij', self.gamma*self.rho*self.Phi_prime_pi[idx]-self.Phi[idx], self.discount_factor[idx]*w[idx])/horizon_normalization/self.n_episode
                trajectory_reward = (w*self.rho*self.rews*self.discount_factor/horizon_normalization).reshape((self.n_episode, self.horizon)).sum(axis=1)
            bias_2 = Phi_0_pi/horizon_normalization
            bias_3 = torch.einsum('ki, kj->ij', self.Phi_term_pi, w[self.end_idx])*self.gamma**self.horizon/horizon_normalization/self.n_episode
            f_w =  bias_1+bias_2-bias_3
            
            if use_var_in_loss:
                loss = (torch.sqrt(torch.mm(f_w.T, self.alpha)**2) + torch.sqrt(td_ball_epsilon*torch.mm(torch.mm(f_w.T, self.M), f_w)))**2+ torch.var(trajectory_reward)
            else:
                loss = torch.sqrt(torch.mm(f_w.T, self.alpha)**2) + torch.sqrt(td_ball_epsilon*torch.mm(torch.mm(f_w.T, self.M), f_w))
            if loss < min_loss:
                w_min = w.clone().detach()
                min_loss = loss
                if self.input_mode == 's':
                    value_est = (w_min*self.rho*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
                else:
                    value_est = (w_min*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
                value_est_list.append(value_est)

            loss.backward()
            optimizer.step()

            # if i % 10 == 0 and i>0:
            #     # import pdb; pdb.set_trace()
            #     print('Iteration {}, Current Loss {:.5f}, Tail Average: {:.2f}'.\
            #         format(i, loss.detach().numpy()[0][0], np.mean(value_est_list[-tail_average:])))
            # print('Iteration: ', i)
            # print('loss in this iteration: ', loss)
            # if self.input_mode == 's':
            #     value_est = (w*self.rho*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
            #     value_est_min = (w_min*self.rho*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
            # else:
            #     value_est = (w*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
            #     value_est_min = (w_min*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode

            # # value_est = (w*rewards*discount_factor).sum().data.numpy()/n_episode
            # # value_est_list.append(value_est)
            
            # print('current estimated value: ', value_est)
            # print('min_w value estimate: ', value_est_min)
            # print('\n')
        return np.mean(value_est_list[-tail_average:])
        # return value_est_min

            





