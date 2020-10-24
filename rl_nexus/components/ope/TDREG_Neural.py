import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
# from rl_nexus.utils.acgd import ACGD, OptimisticAdam, ModifiedAdam
from rl_nexus.utils.optimizer import OptimisticAdam
from itertools import chain
from rl_nexus.utils.ope_utils import choose_estimate_from_sequence
import pdb

dtype = torch.float
torch.set_default_dtype(dtype)

class TDREG_Neural():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon,
                policy_net, hidden_layers, activation, output_transform,
                norm = 'std',
                input_mode = 'sa', seed = 1,
                action_encoding_scheme = 'continuous',
                keep_terminal_states = True,
                use_separate_target_net = False):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.input_mode = input_mode
        self.use_separate_target_net = use_separate_target_net

        # self.n_samples = dataset['obs'].shape[0]
        self.n_episode =  dataset['init_obs'].shape[0]
        # self.non_terminal_idx = (dataset['info']==False)[:,0]
        # self.n_samples_non_terminal = int(self.non_terminal_idx.sum())

        self.non_absorbing_state = (dataset['info']==False)[:,0]
        self.n_samples_non_absorbing = int(self.non_absorbing_state.sum())
        if keep_terminal_states:
            self.included_idx = torch.arange(dataset['obs'].shape[0])
            self.end_idx = np.arange(self.horizon-1, dataset['obs'].shape[0], self.horizon)
            self.absorbing_idx = np.where(dataset['info'][:,0] == True)[0]
        else:
            self.included_idx = torch.tensor(self.non_absorbing_state.nonzero()[0])
            end_idx = []
            absorbing_idx = []
            accumulated_eps_length = 0
            for eps_id in range(self.n_episode):
                real_episode_duration = int(self.non_absorbing_state[eps_id*self.horizon: (eps_id+1)*self.horizon].sum())
                accumulated_eps_length += real_episode_duration
                end_idx.append(accumulated_eps_length-1)
                if real_episode_duration < self.horizon: absorbing_idx.append(accumulated_eps_length-1)
            assert accumulated_eps_length == self.n_samples_non_absorbing
            self.end_idx = np.array(end_idx)
            self.absorbing_idx = np.array(absorbing_idx)
        self.n_samples = self.included_idx.shape[0]
        self.non_absorbing_mask = torch.ones(self.n_samples, dtype=torch.bool)
        self.non_absorbing_mask[self.absorbing_idx] = False

        self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]

        # self.data_acts = torch.tensor(dataset['acts'][self.non_terminal_idx], dtype=torch.long)
        # self.rews = torch.tensor(dataset['rews'][self.non_terminal_idx], dtype=dtype)

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
            obs = torch.tensor(dataset['obs'], dtype = dtype)[self.included_idx]
            next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
            init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)

            # obs = torch.tensor(dataset['obs'][self.non_terminal_idx], dtype = dtype)
            # next_obs = torch.tensor(dataset['next_obs'][self.non_terminal_idx], dtype=dtype)
            # init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
            # term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)
        #* re-whiten the (possibly) non-terminal data frames
        #* should have no effect if all indices are included and if the data is already whitened
        obs_mean = torch.mean(obs, dim=0, keepdims= True)
        obs_std = torch.std(obs, dim=0, keepdims= True)
        self.obs = (obs - obs_mean) / obs_std
        self.next_obs = (next_obs - obs_mean) / obs_std
        self.init_obs = (init_obs - obs_mean) / obs_std
        self.term_obs = (term_obs - obs_mean) / obs_std

        # #* whiten the non-terminal data frames
        # obs_mean = torch.mean(obs, dim=0, keepdims= True)
        # obs_std = torch.std(obs, dim=0, keepdims= True)
        # self.obs = (obs - obs_mean) / obs_std
        # self.next_obs = (next_obs - obs_mean) / obs_std
        # self.init_obs = (init_obs - obs_mean) / obs_std
        # self.term_obs = (term_obs - obs_mean) / obs_std

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
            assert action_encoding_scheme == 'continuous'
            self.w_net = Simple_MLP(input_dim = self.obs_dim+1, output_dim = 1, hidden_layers = hidden_layers,\
                activation= activation, output_transform = output_transform)
            self.q_net = Simple_MLP(input_dim = self.obs_dim, output_dim = self.act_dim, hidden_layers = hidden_layers,\
                activation= activation, output_transform = None)
            if use_separate_target_net:
                self.q_net_target = Simple_MLP(input_dim = self.obs_dim, output_dim = self.act_dim, hidden_layers = hidden_layers,\
                    activation= activation, output_transform = None)
                for param in self.q_net_target.model.parameters():
                    param.requires_grad = False
                self.q_net_target.model.load_state_dict(self.q_net.model.state_dict())
        else:
            raise NotImplementedError

        # end_idx = []
        # accumulated_eps_length = 0
        # for eps_id in range(self.n_episode):
        #     accumulated_eps_length += int(self.non_terminal_idx[eps_id*self.horizon: (eps_id+1)*self.horizon].sum())
        #     end_idx.append(accumulated_eps_length-1)
        # assert accumulated_eps_length == self.n_samples_non_terminal
        # self.end_idx = np.array(end_idx)

    def train(self,num_iter=1000, lr = 1e-3, batch_size = 500, td_reg = 1.0e-2, w_reg = 0,tail_average=10, normalize_w = False):
        optimizer_w = OptimisticAdam(self.w_net.parameters(), lr = lr, betas = (0.0,0.999), eps=1e-8, weight_decay=w_reg)
        optimizer_q = OptimisticAdam(self.q_net.parameters(), lr = -lr, betas = (0.0,0.999), eps=1e-8, weight_decay=1e-3)
        # optimizer = ACGD(max_params=self.q_net.parameters(), min_params = self.w_net.parameters(), lr_max = lr, lr_min = lr)
        #* convert to torch tensor
        self.x = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        self.discount_factor = torch.tensor(discount_factor[self.included_idx], dtype=dtype)

        min_loss = 1e6
        w_min = None
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)

        value_est_list = []
        
        non_absorbing_sn = self.next_obs[self.non_absorbing_mask]
        for i in range(num_iter):
            #* form loss objective
            w_sa = self.w_net(self.x)
            # w_sa = w_sa/torch.mean(w_sa)
            q_sa = self.q_net(self.obs).gather(1,self.data_acts)
            # q_sa_bias = q_sa.clone().detach()
            # q_sn_pi_td = torch.zeros(self.n_samples, 1,dtype=dtype)
            # # non_absorbing_sn = self.next_obs[self.non_absorbing_mask]
            # if self.use_separate_target_net:
            #     q_sn_pi_td[non_absorbing_mask] = (self.q_net_target(non_absorbing_sn)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims=True).detach()
            # else:
            #     q_sn_pi_td[non_absorbing_mask] = (self.q_net(non_absorbing_sn)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims=True).detach()
            
            # td_error = F.smooth_l1_loss(q_sa, q_sn_pi_td*self.gamma+rews)

            q_sn_pi = torch.zeros(self.n_samples, 1,dtype=dtype)
            q_sn_pi[self.non_absorbing_mask] = (self.q_net(non_absorbing_sn)*self.pi_next[self.non_absorbing_mask]).sum(dim=1, keepdims=True)
            q_sn_pi_td = q_sn_pi.clone().detach()

            td_error = F.smooth_l1_loss(q_sa, q_sn_pi_td*self.gamma+self.rews)

            q_s0 = self.q_net(self.init_obs)
            q_s0_pi = (q_s0 * self.pi_init).sum(dim=1, keepdims = True)
            q_sterm = self.q_net(self.term_obs)
            q_sterm_pi = (q_sterm * self.pi_term).sum(dim=1, keepdims = True)
            
            
            bias_1 = ((self.gamma*q_sn_pi - q_sa)*(self.discount_factor*w_sa)).sum() / horizon_normalization / self.n_episode
            bias_2 = q_s0_pi.sum() / horizon_normalization / self.n_episode
            # bias_3 = (self.gamma**(self.horizon+1)*q_sterm_pi*w_sa[self.end_idx]).sum() / horizon_normalization / self.n_episode
            bias_3 = 0

            loss = (bias_1 + bias_2 - bias_3)**2 - td_reg * td_error
            # optimizer.zero_grad()
            # optimizer.step(loss=loss)
            # pdb.set_trace()
            optimizer_w.zero_grad()
            optimizer_q.zero_grad()

            loss.backward()

            optimizer_w.step()
            optimizer_q.step()
            
            if i % 10 == 0:
                # w = self.w_net(self.x).detach()
                value_est = (w_sa*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
                value_est_list.append(value_est)
                # print('Iteration {}, Current Loss {:.5f}, Tail Average: {:.2f}'.\
                #     format(i, loss.detach().numpy(), np.mean(value_est_list[-tail_average:])))
            # if i%1000 == 0:
            #     pdb.set_trace()
        #* with the lack of clear convergence criterion, we need to think about choosing the value to output
        #* here we will put out the most plausible candidate by binning
        final_value_estimate = choose_estimate_from_sequence(value_est_list)        
        return final_value_estimate



    def train_minibatch(self,num_iter=1000, lr = 1e-3, batch_size = 500, td_reg = 1.0e-2, w_reg = 0,tail_average=10, normalize_w = False):
        optimizer_w = OptimisticAdam(self.w_net.parameters(), lr = lr, betas = (0.0,0.999), eps=1e-8, weight_decay=w_reg)
        optimizer_q = OptimisticAdam(self.q_net.parameters(), lr = -lr, betas = (0.0,0.999), eps=1e-8, weight_decay=1e-3)
        #* convert to torch tensor
        self.x = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        self.discount_factor = torch.tensor(discount_factor[self.included_idx], dtype=dtype)

        min_loss = 1e6
        w_min = None
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)

        value_est_list = []
        for i in range(num_iter):
            # perm = torch.randperm(self.n_samples)
            eps_idx = np.random.choice(self.n_episode, batch_size)
            idx_list = list(chain([range(i*self.horizon,(i+1)*self.horizon) for i in eps_idx]))
            idx = torch.tensor(idx_list).flatten()
            end_idx = torch.tensor([(eps+1)*self.horizon-1 for eps in range(batch_size)])
            x = self.x[idx]
            obs = self.obs[idx]
            acts = self.data_acts[idx]
            next_obs = self.next_obs[idx]
            init_obs = self.init_obs[eps_idx]
            term_obs = self.term_obs[eps_idx]

            pi_next = self.pi_next[idx]
            pi_init = self.pi_init[eps_idx]
            pi_term = self.pi_term[eps_idx]
            rews = self.rews[idx]
            discount_factor = self.discount_factor[idx]
            non_absorbing_mask = self.non_absorbing_mask[idx]
            n_sample_batch = batch_size * self.horizon

            #* form loss objective
            w_sa = self.w_net(x)
            q_sa = self.q_net(obs).gather(1,acts)
            q_sn_pi_td = torch.zeros(n_sample_batch, 1,dtype=dtype)
            non_absorbing_sn = next_obs[non_absorbing_mask]
            if self.use_separate_target_net:
                q_sn_pi_td[non_absorbing_mask] = (self.q_net_target(non_absorbing_sn)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims=True).detach()
            else:
                q_sn_pi_td[non_absorbing_mask] = (self.q_net(non_absorbing_sn)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims=True).detach()
            
            td_error = F.smooth_l1_loss(q_sa, q_sn_pi_td*self.gamma+rews)

            q_sn_pi = torch.zeros(n_sample_batch, 1,dtype=dtype)
            q_sn_pi[non_absorbing_mask] = (self.q_net(non_absorbing_sn)*pi_next[non_absorbing_mask]).sum(dim=1, keepdims=True)

            q_s0 = self.q_net(init_obs)
            q_s0_pi = (q_s0 * pi_init).sum(dim=1, keepdims = True)
            q_sterm = self.q_net(term_obs)
            q_sterm_pi = (q_sterm * pi_term).sum(dim=1, keepdims = True)
            
            
            bias_1 = ((self.gamma*q_sn_pi - q_sa)*(discount_factor*w_sa)).sum() / horizon_normalization / batch_size
            bias_2 = q_s0_pi.sum() / horizon_normalization / batch_size
            # bias_3 = (self.gamma**(self.horizon+1)*q_sterm_pi*w_sa[end_idx]).sum() / horizon_normalization / batch_size
            bias_3 = 0

            loss = (bias_1 + bias_2 - bias_3)**2 - td_reg * td_error
            # pdb.set_trace()
            optimizer_w.zero_grad()
            optimizer_q.zero_grad()

            loss.backward()

            optimizer_w.step()
            optimizer_q.step()
            
            if i % 100 == 0:
                w = self.w_net(self.x).detach()
                value_est = (w*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
                value_est_list.append(value_est)
                print('Iteration {}, Current Loss {:.5f}, Tail Average: {:.2f}'.\
                    format(i, loss.detach().numpy(), np.mean(value_est_list[-tail_average:])))
            # if i %1000 == 0:
            #     pdb.set_trace()
        return np.mean(value_est_list[-tail_average:])




    def train_old(self,num_iter=1000, lr = 1e-3, batch_size = 500, td_reg = 1.0e-2, w_reg = 0,tail_average=10, normalize_w = False):
        # self.fqe()
        # pdb.set_trace()
        tau = 0.0
        # optimizer_w = optim.Adam(self.w_net.parameters(), lr = lr, betas = (0.0,0.999), eps=1e-8, weight_decay=w_reg)
        # optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.0,0.999), eps=1e-8, weight_decay=1e-2)
        optimizer_w = OptimisticAdam(self.w_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay=w_reg)
        optimizer_q = OptimisticAdam(self.q_net.parameters(), lr = -lr, betas = (0.9,0.999), eps=1e-8, weight_decay=1e-2)
        # optimizer = ACGD(max_params=self.q_net.parameters(), min_params = self.w_net.parameters(), lr_max = lr, lr_min = lr)
        # pdb.set_trace()
        #* convert to torch tensor
        self.x = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        # self.discount_factor = torch.tensor(discount_factor[self.non_terminal_idx], dtype=dtype)
        self.discount_factor = torch.tensor(discount_factor[self.included_idx], dtype=dtype)
        # self.rews = torch.tensor(self.rews, dtype=dtype)

        # idx = torch.arange(self.n_samples_non_terminal)
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)
        min_loss = 1e6
        w_min = None

        value_est_list = []
        for i in range(num_iter):
            optimizer_w.zero_grad()
            optimizer_q.zero_grad()
            # optimizer.zero_grad()
            w_sa = self.w_net(self.x)
            q_s = self.q_net(self.obs)
            q_sa = torch.gather(q_s, dim = 1, index = self.data_acts)
            q_sn = self.q_net(self.next_obs)
            q_sn_pi = (q_sn*self.pi_next).sum(dim=1, keepdims = True)
            q_s0 = self.q_net(self.init_obs)
            q_s0_pi = (q_s0 * self.pi_init).sum(dim=1, keepdims = True)
            q_sterm = self.q_net(self.term_obs)
            q_sterm_pi = (q_sterm * self.pi_term).sum(dim=1, keepdims = True)
            bias_1 = ((self.gamma*q_sn_pi - q_sa)*(self.discount_factor*w_sa)).sum() / horizon_normalization / self.n_episode
            bias_2 = q_s0_pi.sum() / horizon_normalization / self.n_episode
            bias_3 = (self.gamma**self.horizon*q_sterm_pi*w_sa[self.end_idx]).sum() / horizon_normalization / self.n_episode
            # q_sn_target = self.q_net_target(self.next_obs)
            # q_sn_pi_target = (q_sn_target*self.pi_next).sum(dim=1, keepdims = True)
            # td_error = (self.discount_factor*(q_sa - self.rews - self.gamma*q_sn_pi_target)**2).sum() / horizon_normalization / self.n_episode
            # td_error = 0
            # td_error = ((q_sa - self.rews - self.gamma*q_sn_pi_target)**2).sum() / horizon_normalization / self.n_episode
            td_error = (self.discount_factor*(q_sa - self.rews - self.gamma*q_sn_pi)**2).sum() / horizon_normalization / self.n_episode
            # td_error = ((q_sa - self.rews - self.gamma*q_sn_pi)**2).sum() / horizon_normalization / self.n_episode
            # loss_q = -(bias_1 + bias_2 - bias_3)**2 + td_reg * td_error
            # loss_w = (bias_1 + bias_2 - bias_3)**2
            loss = (bias_1 + bias_2 - bias_3)**2 - td_reg * td_error
            # if i % 100 == 0:
            #     pdb.set_trace()
            loss.backward()
            optimizer_w.step()
            optimizer_q.step()
            # optimizer.step(loss=loss)
            # for target_param, local_param in zip(self.q_net_target.model.parameters(), self.q_net.model.parameters()):
            #     target_param.data.copy_((1-tau)*target_param.data + tau*local_param.data)
            # for target_param, local_param in zip(self.q_net_target.model.parameters(), self.q_net.model.parameters()):
            #     target_param.data.copy_(local_param.data)

            if i % 10 == 0:
                # for target_param, local_param in zip(self.q_net_target.model.parameters(), self.q_net.model.parameters()):
                #     target_param.data.copy_(local_param.data)

                # self.q_net_target.model.load_state_dict(self.q_net.model.state_dict())
                value_est = (w_sa*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
                value_est_list.append(value_est)
            # if i % 10 == 0:
            #     loss_w.backward()
            #     optimizer_w.step()
            #     value_est = (w_sa*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
            #     value_est_list.append(value_est)
            #     self.q_net_target.model.load_state_dict(self.q_net.model.state_dict())
            # else:
            #     loss_q.backward()
            #     optimizer_q.step()
            #     # for target_param, local_param in zip(self.q_net_target.model.parameters(), self.q_net.model.parameters()):
            #     #     target_param.data.copy_((1-tau)*target_param.data + tau*local_param.data)

            if i % 20 == 0 and i>0:
                # import pdb; pdb.set_trace()
                print('Iteration {}, Current Loss {:.5f}, Tail Average: {:.2f}'.\
                    format(i, loss.detach().numpy(), np.mean(value_est_list[-tail_average:])))
        # pdb.set_trace()

        return np.mean(value_est_list[-tail_average:])




    def train_acgd(self, num_iter=300, lr = 1e-3, batch_size = 500, td_reg = 1.0e-2, w_reg = 0,tail_average=10, normalize_w = False):
        # optimizer_w = optim.Adam(self.w_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay=w_reg)
        # optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8)
        optimizer = ACGD(max_params=self.q_net.parameters(), min_params = self.w_net.parameters(), lr_max = lr, lr_min = lr)
        #* convert to torch tensor
        self.x = torch.tensor(np.concatenate((self.obs, self.act_input), axis=1), dtype=dtype)
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        self.discount_factor = torch.tensor(discount_factor[self.non_terminal_idx], dtype=dtype)
        self.rews = torch.tensor(self.rews, dtype=dtype)
        

        idx = torch.arange(self.n_samples_non_terminal)
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)

        min_loss = 1e6
        w_min = None

        value_est_list = []
        
        for i in range(num_iter):
            # optimizer_w.zero_grad()
            optimizer.zero_grad()
            w_sa = self.w_net(self.x)
            q_s = self.q_net(self.obs)
            q_sa = torch.gather(q_s, dim = 1, index = self.data_acts)
            #* sanity check
            error = 0
            for j in range(self.n_samples_non_terminal): error += ((q_s[j, self.data_acts[j]]) - q_sa[j])**2
            assert error <1e-6
            q_sn = self.q_net(self.next_obs)
            q_sn_pi = (q_sn*self.pi_next).sum(dim=1, keepdims = True)

            q_s0 = self.q_net(self.init_obs)
            q_s0_pi = (q_s0 * self.pi_init).sum(dim=1, keepdims = True)
            bias_1 = ((self.gamma*q_sn_pi - q_sa)*(self.discount_factor*w_sa)).sum() / horizon_normalization / self.n_episode
            bias_2 = q_s0_pi.sum() / horizon_normalization / self.n_episode
            td_error = (self.discount_factor*(q_sa - self.rews - self.gamma*q_sn_pi)**2).sum() / horizon_normalization / self.n_episode
            loss = (bias_1 + bias_2)**2 - td_reg * td_error
            optimizer.step(loss = loss)
            if i == 100:
                pdb.set_trace()
            
            if normalize_w:
                w_sa = w_sa/torch.mean(w_sa)
            # loss = torch.tensor([[0]],dtype=dtype)
            value_est = (w_sa*self.rews*self.discount_factor).sum().data.numpy()/self.n_episode
            value_est_list.append(value_est)

            if i % 10 == 0 and i>0:
                # import pdb; pdb.set_trace()
                print('Iteration {}, Current Loss {:.5f}, Tail Average: {:.2f}'.\
                    format(i, loss.detach().numpy(), np.mean(value_est_list[-tail_average:])))
        return np.mean(value_est_list[-tail_average:])

    def fqe(self, num_iter = 5000, lr = 1e-3):
        optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-4, weight_decay = 1e-2)
        #* build target update schedule:
        # update_iter = []
        # last = 10; update_iter.append(0); update_iter.append(last)
        # while update_iter[-1]+last < num_iter:
        #     last = int(last * (1+ 0.05))
        #     update_iter.append(update_iter[-1]+last)
        # update_iter = range(0, num_iter, 20)
        self.q_net_target.model.load_state_dict(self.q_net.model.state_dict())
        tau = 0.1
        # pdb.set_trace()            
        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1])
        self.discount_factor = torch.tensor(discount_factor[self.non_terminal_idx], dtype=dtype)
        horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma)

        for i in range(num_iter):
            optimizer_q.zero_grad()
            q_s = self.q_net(self.obs)
            q_sa = torch.gather(q_s, dim = 1, index = self.data_acts)
            q_sn = self.q_net_target(self.next_obs)
            q_sn_pi = (q_sn*self.pi_next).sum(dim=1, keepdims = True)
            # td_error = (self.discount_factor*(q_sa - self.rews - self.gamma*q_sn_pi)**2).sum() / horizon_normalization / self.n_episode
            td_error = ((q_sa - self.rews - self.gamma*q_sn_pi)**2).sum() / horizon_normalization / self.n_episode
            td_error.backward()
            optimizer_q.step()
            for target_param, local_param in zip(self.q_net_target.model.parameters(), self.q_net.model.parameters()):
                target_param.data.copy_((1-tau)*target_param.data + tau*local_param.data)
            # if i in update_iter:
            #     self.q_net_target.model.load_state_dict(self.q_net.model.state_dict())
            if i % 10 == 0:
                # optimizer_q = optim.Adam(self.q_net.parameters(), lr = lr, betas = (0.9,0.999), eps=1e-8, weight_decay = 1e-3)
                q_s0 = self.q_net(self.init_obs)
                q_s0_pi = (q_s0 * self.pi_init).sum()
                q_sterm = self.q_net(self.term_obs)
                q_sterm_pi = (q_sterm * self.pi_term).sum()
                value_est = (q_s0_pi - self.gamma**self.horizon*q_sterm_pi) / self.n_episode
                print('iter {} value estimate: '.format(i), value_est.detach().numpy())
                print('td error {}'.format(td_error))
                max_weight = 0
                for param in self.q_net.model.parameters(): max_weight = max(max_weight, param.data.max())
                print('model max weight: {}\n'.format(max_weight))

            # if i %100 == 0:
            #     pdb.set_trace()
        pdb.set_trace()
