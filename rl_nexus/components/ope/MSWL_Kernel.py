import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
import pdb

dtype = torch.float

class MSWL():
    def __init__(self, dataset, obs_dim, act_dim, k_tau, norm = None,
                hidden_layers = [64,64], activation = 'relu', 
                policy_net = None, keep_terminal_states = True,
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

        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]
        if self.policy_net is not None:
            raise NotImplementedError
        else:
            self.ratio = torch.tensor(dataset['ratio'], dtype=dtype)[self.included_idx]
            self.factor = torch.tensor(dataset['factor'], dtype= dtype)[self.included_idx]
        
        if norm == 'std':
            raise NotImplementedError
        elif norm is None:
            self.obs = torch.tensor(dataset['obs'], dtype=dtype)[self.included_idx]
            self.next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
            self.init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            self.term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)
            self.done = torch.tensor(dataset['done'], dtype=torch.bool)[self.included_idx]
        
        self.w_net = Simple_MLP(input_dim = self.obs_dim, output_dim = 1, hidden_layers = hidden_layers,\
            activation= activation, output_transform = 'logexp')

        self.sampling_weights = torch.ones(self.n_samples) / self.n_samples
        self.sampling_weights_init = torch.ones(self.n_episode) / self.n_episode
        self.absorbing_idx = torch.tensor(np.where(dataset['info'][:,0] == True)[0])
        self.non_absorbing_mask = torch.ones(self.n_samples, dtype=torch.bool)
        self.non_absorbing_mask[self.absorbing_idx] = False

    def estimate_median_distance(self, num_sample = None):
        # index_set_1 = torch.randperm(self.n_samples)[:num_sample]
        # index_set_2 = torch.randperm(self.n_samples)[:num_sample]
        index_set_1 = torch.multinomial(self.sampling_weights, num_sample)
        index_set_2 = torch.multinomial(self.sampling_weights, num_sample)
        obs_1 = self.obs[index_set_1]
        obs_2 = self.obs[index_set_2]

        m0 = torch.median(torch.sqrt(torch.sum(torch.square(obs_1[None, :, :] - obs_2[:, None, :]), axis = -1)))

        return torch.tensor([m0] * 4, dtype = dtype)
    
    def train(self, num_iter, lr, batch_size, w_reg = 0.0, eval_interval = 10, detach_target = False):
        optimizer = optim.Adam(self.w_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay=w_reg)

        med_dist = self.estimate_median_distance(num_sample = 5000) / self.k_tau
        n_samples_batch = batch_size**2

        value_est_list = []
        horizon_normalization = (1 - self.gamma**101) / (1-self.gamma)

        for i in range(num_iter):
            #* sample a mini-batch of data
            # index_set_1 = torch.randperm(self.n_samples)[:batch_size]
            # index_set_2 = torch.randperm(self.n_samples)[:batch_size]
            index_1 = torch.multinomial(self.sampling_weights, batch_size)
            index_2 = torch.multinomial(self.sampling_weights, batch_size)
            index_init_1 = torch.multinomial(self.sampling_weights_init, self.n_episode, replacement = True)
            index_init_2 = torch.multinomial(self.sampling_weights_init, self.n_episode, replacement = True)
            
            obs_1 = self.obs[index_1]; obs_2 = self.obs[index_2]
            next_obs_1 = self.next_obs[index_1]; next_obs_2 = self.next_obs[index_2]
            # rew_1 = self.rews[index_1]; rew_2 = self.rews[index_2]
            ratio_1 = self.ratio[index_1]; ratio_2 = self.ratio[index_2]
            factor_1 = self.factor[index_1]; factor_2 = self.factor[index_2]
            init_obs_1 = self.init_obs[index_init_1]; init_obs_2 = self.init_obs[index_init_2]
            non_absorbing_mask_1 = self.non_absorbing_mask[index_1]; non_absorbing_mask_2 = self.non_absorbing_mask[index_2]

            #* Forming the loss function for this batch of samples
            w_s_1 = self.w_net(obs_1) 
            w_sn_1 = torch.ones_like(w_s_1)
            # w_sn_1[non_absorbing_mask_1] = self.w_net(next_obs_1).detach()[non_absorbing_mask_1]
            w_sn_1[non_absorbing_mask_1] = self.w_net(next_obs_1)[non_absorbing_mask_1]

            w_s_2 = self.w_net(obs_2)
            # w_s_2 = self.w_net(obs_2).detach()
            w_sn_2 = torch.ones_like(w_s_2)
            # w_sn_2[non_absorbing_mask_2] = self.w_net(next_obs_2).detach()[non_absorbing_mask_2]
            w_sn_2[non_absorbing_mask_2] = self.w_net(next_obs_2)[non_absorbing_mask_2]

            w_s0_1 = self.w_net(init_obs_1)
            w_s0_2 = self.w_net(init_obs_2)
            # w_s0_1 = self.w_net(init_obs_1).detach()
            # w_s0_2 = self.w_net(init_obs_2).detach()

            #* the kernel loss is formed by crossing the 2 tuples into 4 pairs of losses
            #* (next_obs_1, init_obs_1) x (next_obs_2, init_obs_2)

            coeff = [
                self.gamma ** 2,
                (1 - self.gamma) ** 2,
                self.gamma * ( 1 - self.gamma),
                self.gamma * ( 1 - self.gamma),
            ]

            # coeff = [
            #     self.gamma ** 2,
            #     (1/horizon_normalization) ** 2,
            #     self.gamma * 1/horizon_normalization,
            #     self.gamma * 1/horizon_normalization,
            # ]

            Kernel = [
                (next_obs_1, next_obs_2),
                (init_obs_1, init_obs_2),
                (next_obs_1, init_obs_2),
                (init_obs_1, next_obs_2)
            ]
            
            w1 = (w_s_1 * ratio_1 - w_sn_1) * factor_1
            w2 = (w_s_2 * ratio_2 - w_sn_2) * factor_2

            w_init_1 = 1 - w_s0_1
            w_init_2 = 1 - w_s0_2
            weights = [
                (w1, w2),
                (w_init_1, w_init_2),
                (w1, w_init_2),
                (w_init_1, w2),
            ]

            loss = 0
            for index in range(len(Kernel)):
                c = coeff[index]
                k1, k2 = Kernel[index]
                x1, x2 = weights[index]
                diff = k1.unsqueeze(1) - k2.unsqueeze(0)
                K = torch.exp(-torch.sum(torch.square(diff), dim = -1) / 2.0 / med_dist[index]**2 )
                sample_num = K.shape[0] * K.shape[1]
                loss += torch.squeeze( c * x1.T @ K @ x2) / sample_num


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                w_s = self.w_net(self.obs)
                value_est = torch.sum(w_s * self.factor * self.ratio * self.rews).data.numpy() / self.n_episode
                value_est_list.append(value_est)

                print('Iter: {}. Loss current: {:.4f}. MSWL Estimate: {:.2f}'.format(i, loss.detach().numpy(), value_est))
                print('max weight last layer: ', self.w_net.model[-1].weight.max())
                print('\n')
                # if  i % 1000 == 0:
                pdb.set_trace()
        pdb.set_trace()
        return value_est





