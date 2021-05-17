import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.utils.ope_utils import choose_estimate_from_sequence, select_most_likely_value
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
from rl_nexus.utils.optimizer import AdaBound
import pdb

dtype = torch.float

class Mu_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation):
        super(Mu_network, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden = []
        for i in range(len(hidden_layers)-1):
            self.hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.output_dim = output_dim
    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = torch.tanh(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

class R_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation):
        super(R_network, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden = []
        for i in range(len(hidden_layers)-1):
            self.hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.output_dim = output_dim
    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = torch.tanh(x)
        x = self.output_layer(x)
        return x

class W_network(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation):
        super(W_network, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        # self.norm_input = nn.LayerNorm(hidden_layers[0])
        self.hidden = []
        # self.norm = []
        for i in range(len(hidden_layers)-1):
            self.hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            # self.norm.append(nn.LayerNorm(hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
    def forward(self, x):
        x = self.input_layer(x)
        # x = self.norm_input(x)
        x = torch.tanh(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            # x = self.norm[i](x)
            x = torch.tanh(x)
        x = self.output_layer(x)
        # x = -F.threshold(-x, -4,-4)
        # x = F.threshold(x, -16, -16)
        x = torch.log(1+torch.exp(x))
        # x = F.relu6(x)
        # x = -F.threshold(-x, -6,-6)
        # x = 10*torch.sigmoid(x)
        # x = torch.relu(x)
        # x = x / torch.mean(x)
        # x = x / avg
        return x

# class W_network(nn.Module):
#     def __init__(self, input_dim, seed):
#         super(W_network, self).__init__()
#         self.seed = seed
#         torch.manual_seed(self.seed)
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64,64)
#         self.fc3 = nn.Linear(64, 1)
                
#     def forward(self, x):
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = torch.log(1+torch.exp(self.fc3(x)))
#         # x = F.normalize(x, p=1, dim=0)
#         x = x / torch.mean(x)
#         return x

class W_Regression():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon, 
                hidden_layers, activation, policy_net = None,
                norm = 'std', use_delayed_target = False,
                keep_terminal_states = True, debug = True):
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
        else:
            pass
        self.n_samples = self.included_idx.shape[0]
        self.non_absorbing_mask = torch.ones(self.n_samples, dtype=torch.bool)
        self.non_absorbing_mask[self.absorbing_idx] = False
        self.non_terminal = torch.ones(self.n_samples,1)
        self.non_terminal[self.end_idx,0] = 0

        self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
        self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]
        self.next_rews = torch.zeros_like(self.rews)
        for episode in range(self.n_episode):
            self.next_rews[episode*self.horizon:(episode+1)*self.horizon-1] = self.rews[episode*self.horizon+1:(episode+1)*self.horizon]
        
        self.next_acts = torch.tensor(dataset['next_acts'], dtype=torch.long)[self.included_idx]
        self.init_acts = torch.tensor(dataset['init_acts'], dtype=torch.long)
        if self.policy_net is not None:
            raise NotImplementedError
        else:
            self.pi_current = torch.tensor(dataset['target_prob_obs'],dtype=dtype)[self.included_idx]
            self.pi_next = torch.tensor(dataset['target_prob_next_obs'], dtype=dtype)[self.included_idx]
            self.pi_init = torch.tensor(dataset['target_prob_init_obs'], dtype=dtype)
            self.pi_term = torch.tensor(dataset['target_prob_term_obs'], dtype=dtype)
            self.mu_prob_act = torch.tensor(dataset['behavior_act_prob'],dtype=dtype)[self.included_idx]
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

        self.w_network = W_network(input_dim = self.obs_dim, hidden_layers=hidden_layers,activation=activation)
        # self.mu_hat_network = Mu_network(input_dim=self.obs_dim, output_dim=self.act_dim, hidden_layers=hidden_layers, activation=activation)
        self.r_network = R_network(input_dim=self.obs_dim, output_dim=self.act_dim, hidden_layers=hidden_layers, activation=activation)
        # for param in self.w_network.parameters():
        #     torch.nn.init.normal_(param.data, mean=0, std = 0.001)
        self.debug = debug
        # self.ratio_org = torch.tensor(dataset['ratio'], dtype = dtype)[self.included_idx]
        self.ratio = torch.tensor(dataset['ratio'], dtype = dtype)[self.included_idx]

    def train(self, num_iter = 2000, lr = 1.0e-3, batch_size = 500, tail_average=10, reg = 1e-3,\
        eps = 1.0e-4):
        if not batch_size:
            batch_size = self.n_samples #* use the whole batch if no batchsize declared
        feature_dim =  self.hidden_layers[-1]
        optimizer = AdaBound(self.w_network.parameters(), lr = lr, final_lr=1.0,\
            gamma=1.0e-3, eps=eps)
        # optimizer = torch.optim.SGD(self.w_network.parameters(), lr=1e-2, momentum=0.9)
        # optimizer = torch.optim.Adam(self.w_network.parameters(), lr=1e-2)
        discount = torch.tensor(np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1]), dtype=dtype)
        horizon_normalization = (1-self.gamma**(self.horizon+1)) / (1-self.gamma)
        value_est_list = []
        loss_list = []
        # log_gamma = np.log(self.gamma)
        # log_ratio = torch.log(self.ratio)
        one_hot_acts = torch.tensor(torch.squeeze(F.one_hot(self.data_acts, num_classes = self.act_dim)), dtype=dtype)
        # optimizer_mu = torch.optim.Adam(self.mu_hat_network.parameters(), lr = 1.0e-3, weight_decay=1e-2)
        #optimizer_r = torch.optim.Adam(self.r_network.parameters(), lr = 1.0e-3)
        optimizer_r = AdaBound(self.r_network.parameters(), lr = 1.0e-3, final_lr=0.1,\
            gamma=1.0e-3, eps=eps)
        # weights = torch.zeros(self.act_dim)
        # for a in range(self.act_dim):
        #     # weights[a] = (self.data_acts[:,0] == a).sum() / self.data_acts.shape[0]
        #     # weights[a] = 1/(self.data_acts[:,0] == a).sum()
        #     weights[a] = (self.data_acts[:,0] == a).sum()
        train_r = False
        if train_r:
            for i in range(200):
                bs = 500
                perm = torch.randperm(self.n_samples)
                num_batches = self.n_samples // bs
                # loss_func = torch.nn.CrossEntropyLoss(weight=weights)
                current_loss = 0
                for j in range(num_batches):
                    idx = perm[j*bs:(j+1)*bs]
                    obs = self.obs[idx]
                    r = self.rews[idx]
                    # acts = one_hot_acts[idx]
                    # acts = torch.squeeze(self.data_acts[idx])
                    acts = self.data_acts[idx]
                    # predicted_acts = self.mu_hat_network(obs)
                    predicted_r = self.r_network(obs).gather(1, acts)
                    # pdb.set_trace()
                    # loss_func = torch.nn.CrossEntropyLoss()
                    # optimizer_mu.zero_grad()
                    optimizer_r.zero_grad()
                    # loss_mu = loss_func(predicted_acts, acts)
                    # loss_mu = F.kl_div(predicted_acts, acts)
                    loss_r = F.mse_loss(predicted_r, r)
                    # loss = torch.nn.CrossEntropyLoss(predicted_acts, acts)
                    # pdb.set_trace()
                    # loss = F.cross_entropy(predicted_acts, acts)
                    
                    # loss_mu.backward()
                    # optimizer_mu.step()
                    # current_loss += loss_mu.clone().detach()
                    loss_r.backward()
                    optimizer_r.step()
                    current_loss += loss_r.clone().detach()

                    # print(loss)
                current_loss /= num_batches
                if (i+1)%10 == 0:
                    print('r train iter: ', i)
                    print('r train loss:', current_loss)
                # pdb.set_trace()
            # self.mu_hat = self.mu_hat_network(self.obs).gather(1, self.data_acts).detach()
            # self.ratio = self.pi_current.gather(1,self.data_acts) / self.mu_hat
            self.r_hat = self.r_network(self.obs).gather(1, self.data_acts).detach()
            self.r_pi = (self.r_network(self.obs).detach()*self.pi_current).sum(dim=1, keepdims= True)
            # pdb.set_trace()
        for i in range(num_iter):
            perm = torch.randperm(self.n_samples)
            num_batches = self.n_samples // batch_size
            current_loss = 0
            # normalized_w_factor = torch.mean(self.w_network(self.obs)).detach()
            for j in range(num_batches):
                idx = perm[j*batch_size:(j+1)*batch_size]
                obs = self.obs[idx]
                next_obs = self.next_obs[idx]
                ratio = self.ratio[idx]
                discount_factor = discount[idx]
                # log_ratio_batch = log_ratio[idx]
                # print(self.w_network(obs).detach().max())
                # w_init = self.w_network(self.init_obs, avg_w)
                # log_w = self.w_network(obs)
                # log_w_next = self.w_network(obs)
                w = self.w_network(obs)
                w_next = self.w_network(next_obs)
                w_init = self.w_network(self.init_obs)
                # w_mean = torch.sum(torch.exp(log_w)*discount_factor)/self.n_episode/horizon_normalization
                w_mean = torch.sum(w*discount_factor) / torch.sum(discount_factor)
                # w_mean = torch.sum(w)/self.n_episode/horizon_normalization
                # w_mean = torch.mean(w)
                # normalized_w = w / w_mean
                # normalized_w_next = w_next / w_mean
                # log_w = self.w_network(obs)
                # log_w_next = self.w_network(next_obs)
                # expected_w_next = (1-self.gamma) + self.gamma*ratio*w.detach()
                expected_w_next = 1/(i+1)*w_next.detach()+ (1-1/(i+1))*ratio*w.detach()
                # expected_w_next = ratio*w.detach()
                # expected_w_next = ratio*w
                # expected_w_next = self.gamma*ratio*normalized_w.detach()
                # expected_w_next = self.gamma*ratio*normalized_w.detach()
                # expected_w_next = self.gamma*ratio*w.detach()
                # expected_log_w_next = log_gamma + log_ratio_batch + log_w.detach()
                # expected_log_w_next = log_ratio_batch + log_w
                # loss = F.smooth_l1_loss(w_next, expected_w_next.detach(),beta=1.0e-2) + (torch.mean(w)-1)**2 + F.mse_loss(w_init, torch.ones(self.n_episode,1))
                # loss = F.smooth_l1_loss(w_next, expected_w_next) #+ 10*(torch.mean(w)-1)**2
                # loss = F.smooth_l1_loss(normalized_w_next, expected_w_next, beta=0.01)
                # loss = F.mse_loss(normalized_w_next, expected_w_next)
                # loss = torch.sum(discount_factor*(normalized_w_next - expected_w_next)**2) /batch_size
                # loss = torch.sum(discount_factor*(normalized_w_next - expected_w_next)**2) /batch_size
                # loss = torch.sum(discount_factor*(normalized_w_next - expected_w_next)**2) / self.n_episode/horizon_normalization
                
                # loss = torch.sum((log_w_next - expected_log_w_next)**2) / batch_size + (w_mean-1)**2
                # loss = torch.sum(discount_factor*(w_next - expected_w_next)**2) / torch.sum(discount_factor) + (w_mean-1)**2
                # loss = torch.sum((w_next - expected_w_next)**2) / batch_size + (w_mean-1)**2
                # loss = torch.sum((w_next - expected_w_next)**2) / batch_size+ (w_mean-1)**2
                # loss = F.smooth_l1_loss(w_next, expected_w_next) + F.mse_loss(w_init, torch.ones(self.n_episode,1)) + (w_mean-1)**2
                loss = F.smooth_l1_loss(w_next, expected_w_next) + F.mse_loss(w_init, torch.ones(self.n_episode,1))#+ (w_mean-1/self.gamma)**2
                # loss = F.smooth_l1_loss(w_next, expected_w_next, beta=0.1) + (w_mean-1)**2
                # loss = torch.sum((torch.log(normalized_w_next) -torch.log(expected_w_next))**2) / batch_size + (w_mean-1)**2
                # loss = torch.sum((normalized_w_next - expected_w_next)**2) / batch_size
                # n = torch.abs(normalized_w_next - expected_w_next)*discount_factor
                # beta = 0.1
                # cond = n < beta
                # loss = (torch.where(cond, (0.5*n**2/beta), (n-0.5*beta))).sum() / batch_size
                # loss = F.smooth_l1_loss(log_w_next, expected_log_w_next.detach()) + 100*(torch.mean(torch.exp(log_w))-1)**2
                # loss = F.mse_loss(log_w_next, expected_log_w_next.detach()) + 100*(torch.mean(torch.exp(log_w))-1)**2
                optimizer.zero_grad()
                loss.backward()
                for param in self.w_network.parameters():
                    param.grad.data.clamp_(-1,1)

                optimizer.step()
                current_loss += loss.clone().detach()
                # print(loss.clone().detach())
            current_loss /= num_batches
            if i%5 == 0:
                if self.debug:
                    for params in self.w_network.parameters(): print(params.data.max())
                    print('iter {} current loss: '.format(i), current_loss.clone().detach().numpy())
                with torch.no_grad():
                    self.w_network.eval()
                    w = self.w_network(self.obs)
                    # normalize_w = torch.sum(w*discount) / self.n_episode/self.horizon
                    # w = w/normalize_w
                    # w_mean = torch.sum(w*discount)/self.n_episode/horizon_normalization
                    # w_mean = torch.sum(w)/self.n_episode/horizon_normalization
                    # w_mean = torch.mean(w)
                    # w = w/ w_mean
                    # w = w / torch.mean(w)
                    # value_est = torch.sum(w * self.ratio * discount * self.rews)/self.n_episode
                    # value_est = torch.sum(w * discount * self.r_pi)/self.n_episode/torch.sum(w*discount)*horizon_normalization
                    # value_est = torch.sum(w * self.ratio * discount * self.rews)/torch.sum(w*discount)*horizon_normalization
                    value_est = torch.sum(w * self.ratio * discount * self.rews)/torch.sum(w*self.ratio*discount)*horizon_normalization
                    # value_est = torch.sum(w * self.ratio * discount * self.rews)/torch.sum(w*self.ratio*discount)*horizon_normalization
                    # value_est = torch.sum(w * self.ratio * self.rews)/self.n_episode/horizon_normalization
                    # value_est = torch.sum(w * self.ratio * self.rews)/self.n_episode
                    # value_est = torch.sum(w * self.ratio * discount * self.rews)
                    print('max w', w.max())
                    print('latest estimate: ', value_est)
                    print('\n')
                    value_est_list.append(np.asscalar(value_est.numpy()))
                    loss_list.append(np.asscalar(current_loss.numpy()))
                self.w_network.train()
                # if i % 1000 == 0 and i>0:
                #     pdb.set_trace()
        pdb.set_trace()
        return np.mean(value_est_list[-int(tail_average/100 * len(value_est_list)):])


