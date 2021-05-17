import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_nexus.utils.ope_utils import choose_estimate_from_sequence, select_most_likely_value
from rl_nexus.components.models.Simple_MLP.Simple_MLP import Simple_MLP
from rl_nexus.utils.optimizer import AdaBound
from torch.autograd import Variable
import pdb

dtype = torch.float

class Feature_Extractor(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation):
        super(Feature_Extractor, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.norm_input = nn.LayerNorm(hidden_layers[0])
        self.hidden = []
        self.norm = []
        for i in range(len(hidden_layers)-1):
            self.hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.norm.append(nn.LayerNorm(hidden_layers[i+1]))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.norm_input(x)
        x = torch.tanh(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.norm[i](x)
            x = torch.tanh(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class Feature_Learning():
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

        # self.dynamics = Dynamics(input_dim= self.obs_dim, num_actions = self.act_dim, hidden_layers=hidden_layers,\
        #     activation=activation)
        self.feature_extractor = Feature_Extractor(input_dim = self.obs_dim, hidden_layers=hidden_layers,activation=activation)
        for param in self.feature_extractor.parameters():
            torch.nn.init.normal_(param.data, mean=0, std = 0.001)
        self.debug = debug
        self.ratio = torch.tensor(dataset['ratio'], dtype = dtype)[self.included_idx]

    def train(self, num_iter = 2000, lr_feature = 1.0e-3, lr_model = 1.0e-3, batch_size = 500, tail_average=10, reg = 1e-3,\
        eps = 1.0e-4):
        # optimizer_feature = AdaBound(self.feature_extractor.parameters(), lr = lr_model, final_lr=1.0, gamma=1.0e-4, eps=eps)
        if not batch_size:
            batch_size = self.n_samples #* use the whole batch if no batchsize declared
        feature_dim =  self.hidden_layers[-1]

        P = Variable(torch.randn(feature_dim*self.act_dim, feature_dim*self.act_dim).type(dtype)/1000, requires_grad=True)
        w = Variable(torch.randn(feature_dim*self.act_dim, 1).type(dtype)/1000, requires_grad=True)
        theta = Variable(torch.randn(feature_dim*self.act_dim, 1).type(dtype)/1000, requires_grad=True)
        # P_mu = Variable(torch.randn(feature_dim, feature_dim).type(dtype)/1000, requires_grad=True)
        # w_mu = Variable(torch.randn(feature_dim, 1).type(dtype)/1000, requires_grad=True)
        # optimizer_model = AdaBound([P,w,P_mu,w_mu], lr = lr_model, final_lr=1.0, gamma=1.0e-4, eps=eps)
        # optimizer = AdaBound(list(self.feature_extractor.parameters())+ [P,w,P_mu,w_mu], lr = lr_model, final_lr=1.0,\
        #     gamma=1.0e-4, eps=eps)
        # optimizer = AdaBound(list(self.feature_extractor.parameters())+ [P,w], lr = lr_model, final_lr=1.0,\
        #     gamma=1.0e-4, eps=eps)
        optimizer = AdaBound(list(self.feature_extractor.parameters())+ [P,w,theta], lr = lr_model, final_lr=1.0,\
            gamma=1.0e-4, eps=eps)

        # optimizer = torch.optim.Adam(list(self.feature_extractor.parameters())+ [P,w,P_mu,w_mu], lr = lr_model)

        one_hot_acts = torch.squeeze(F.one_hot(self.data_acts, num_classes = self.act_dim))
        one_hot_next_acts = torch.squeeze(F.one_hot(self.next_acts, num_classes = self.act_dim))
        one_hot_init_acts = torch.squeeze(F.one_hot(self.init_acts, num_classes = self.act_dim))
        I_sa = torch.eye(feature_dim*self.act_dim)
        I_s = torch.eye(feature_dim)
        # all_next_acts = torch.zeros(batch_size, self.act_dim**2, dtype=torch.long)
        # for a in range(self.act_dim):
        #     all_next_acts[:, a*self.act_dim:(a+1)*self.act_dim] = torch.squeeze(F.one_hot(a*torch.ones(batch_size,1,dtype=torch.long), num_classes = self.act_dim))
        

        value_est_list = []
        # value_est_list_mu = []
        value_est_list_q = []
        old_init_feature = torch.zeros(self.n_episode, feature_dim, dtype=dtype)
        old_transition = torch.zeros(feature_dim*self.act_dim, feature_dim*self.act_dim, dtype=dtype)
        old_P = torch.zeros(feature_dim*self.act_dim, feature_dim*self.act_dim, dtype=dtype)
        old_w = torch.zeros(feature_dim*self.act_dim, 1, dtype=dtype)
        old_feature = torch.zeros(self.n_samples, feature_dim, dtype=dtype)
        discount = torch.tensor(np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1]), dtype=dtype)
        horizon_normalization = (1-self.gamma**(self.horizon+1)) / (1-self.gamma)
        V_mu = (self.rews * discount).view(-1, self.horizon).sum(dim=1, keepdims=True) / horizon_normalization
        # batch_size_list = [100,200,500,1000,2000,5000,10000,20000,50000,100000]
        for i in range(num_iter):
            # batch_size = batch_size_list[i//500]
            perm = torch.randperm(self.n_samples)
            # perm = torch.arange(self.n_samples)
            num_batches = self.n_samples // batch_size
            current_loss = 0
            for j in range(num_batches):
                idx = perm[j*batch_size:(j+1)*batch_size]
                obs = self.obs[idx]
                acts = one_hot_acts[idx]
                next_acts = one_hot_next_acts[idx]
                next_obs = self.next_obs[idx]
                pi_next = self.pi_next[idx]
                pi_current = self.pi_current[idx]
                rews = self.rews[idx]
                ratio = self.ratio[idx]
                next_rews = self.next_rews[idx]
                non_terminal = self.non_terminal[idx]
                # mu_prob_act = self.mu_prob_act[idx]
                # pi_prob_act = pi_current.gather(1, self.data_acts[idx])

                Z = self.feature_extractor(obs)
                Z_prime = self.feature_extractor(next_obs).detach()
                Z_expanded = torch.einsum('ab,ac->abc', Z, acts).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
                phi_prime = torch.einsum('ab,ac->abc', Z_prime, pi_next).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
                next_feature = Z_expanded @ P
                reward = Z_expanded @ w
                # next_r = next_feature @ w
                # next_r_target = (phi_prime @ w).detach()
                # model_loss = torch.sum((Z_expanded @ P - phi_prime.detach())**2) / batch_size
                # reward_loss = torch.sum((Z_expanded @ w - rews)**2) / batch_size
                # feature_loss = torch.sum((Z_expanded @ P.detach() @ w.detach() - phi_prime @ w.detach())**2) /batch_size
                # q_sa = (Z @ theta).gather(1, self.data_acts[idx])
                q_sa = Z_expanded @ theta
                # q_prime_pi = next_feature @ theta
                # q_target = (reward + self.gamma*q_prime_pi).detach()
                q_prime_pi = phi_prime @ theta.detach()
                q_target = rews+ self.gamma*q_prime_pi
                # Z_prime_expanded = torch.einsum('ab,ac->abc', Z_prime, all_next_acts)
                # q_prime_pi = Z_expanded @ P.detach() @ theta
                # q_target = Z_expanded @ w.detach() + self.gamma*q_prime_pi
                # next_feature_mu = Z @ P_mu
                # reward_mu = Z @ w_mu
                # pdb.set_trace()
                # q_sa = q.gather(1, self.data_acts[idx])
                
                # model_loss = torch.sum((next_feature - phi_prime)**2) / batch_size
                # reward_loss = torch.sum((reward - rews)**2) / batch_size
                # # reg_loss = torch.sum((q_sa - reward - self.gamma*q_prime_pi)**2) / batch_size
                # reg_loss = torch.sum((q_sa - q_target)**2) / batch_size
                model_loss = F.mse_loss(next_feature, phi_prime)
                reward_loss = F.mse_loss(reward, rews)
                reg_loss = F.smooth_l1_loss(q_sa, q_target)
                # reg_loss = torch.sum((next_r - next_r_target)**2) /batch_size
                # reg_loss = torch.sum((q_sa - q_target)**2) / batch_size
                # model_loss = torch.sum((next_feature - phi_prime)**2*ratio) / batch_size
                # reward_loss = torch.sum((reward - rews)**2*ratio) / batch_size
                # model_loss_mu = torch.sum( (next_feature_mu - Z_prime)**2) / batch_size
                # reward_loss_mu = torch.sum((reward_mu-rews)**2) / batch_size
                # model_loss = F.smooth_l1_loss(next_feature, phi_prime, beta=1e-2, reduction='sum')/batch_size
                # reward_loss = F.smooth_l1_loss(reward, rews, beta=1e-2, reduction='sum')/batch_size 
                # model_loss_mu = F.smooth_l1_loss(next_feature_mu, Z_prime, beta=1e-2, reduction='sum')/batch_size
                # reward_loss_mu = F.smooth_l1_loss(reward_mu, rews, beta=1e-2, reduction='sum')/batch_size 

                # loss = model_loss_mu +  reward_loss_mu
                # reg_loss = torch.norm(P) + torch.norm(w) + torch.norm(P_mu) + torch.norm(w_mu)
                # reg_loss = torch.sum(P.T @ P) + torch.sum(w.T @ w) + torch.sum(P_mu.T @ P_mu) + torch.sum(w_mu.T @ w_mu)
                # reg_loss = torch.sum(P.T @ P) + torch.sum(w.T @ w)
                # loss = model_loss + reward_loss + model_loss_mu +  reward_loss_mu + reg*reg_loss
                # loss = model_loss + reward_loss + model_loss_mu +  reward_loss_mu
                # loss = model_loss + reward_loss + model_loss_mu +  reward_loss_mu
                # loss = model_loss + reward_loss + reg*reg_loss
                loss = model_loss + reward_loss + reg_loss
                # loss = model_loss + reward_loss
                # loss = reg_loss 
                # loss = model_loss + reward_loss
                # loss = model_loss + reward_loss + feature_loss

                # optimizer_feature.zero_grad()
                # optimizer_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                # for param in self.feature_extractor.parameters():
                #     param.grad.data.clamp_(-1,1)
                # P.grad.data.clamp_(-1,1)
                # w.grad.data.clamp_(-1,1)
                # theta.grad.data.clamp_(-1,1)
                # P_mu.grad.data.clamp_(-1,1)
                # w_mu.grad.data.clamp_(-1,1)

                # optimizer_feature.step()
                # optimizer_model.step()
                optimizer.step()
                current_loss += loss.clone().detach()
            current_loss /= num_batches
            # #* update theta?
            # with torch.no_grad():
            #     Z = self.feature_extractor(self.obs)
            #     Z_prime = self.feature_extractor(self.next_obs)
            #     phi = torch.einsum('ab,ac->abc', Z, one_hot_acts).permute(0,2,1).contiguous().view(self.n_samples, feature_dim*self.act_dim)
            #     phi_prime = torch.einsum('ab,ac->abc', Z_prime, self.pi_next).permute(0,2,1).contiguous().view(self.n_samples, feature_dim*self.act_dim)
            #     feature_difference = phi - self.gamma*phi_prime
            #     theta = torch.inverse(feature_difference.T @ feature_difference + 1e-8*self.n_samples*I_sa) @ feature_difference.T @ self.rews
            if i%5 == 0:
                # pdb.set_trace()
                if self.debug:
                    # print('max weight model:')
                    # print(P.data.max())
                    # print(w.data.max())
                    # print(P_mu.data.max())
                    # print(w_mu.data.max())
                    # print('max weight feature extractor:')
                    # for param in self.feature_extractor.parameters(): print(torch.norm(param.data.max()))
                    # print('gradient norm feature extractor:')
                    # for param in self.feature_extractor.parameters(): print(torch.norm(param.grad.data))
                    # print('gradient norm model:')
                    # print(torch.norm(P.grad.data))
                    # print(torch.norm(w.grad.data))
                    # print('\n')                
                    print('iter {} current loss: '.format(i), current_loss.clone().detach().numpy())
                with torch.no_grad():
                    self.feature_extractor.eval()
                    Z_init_eval = self.feature_extractor(self.init_obs)
                    Z_eval = self.feature_extractor(self.obs)
                    phi_init_eval = torch.einsum('ab,ac->abc', Z_init_eval , self.pi_init).permute(0,2,1).contiguous().view(self.n_episode, feature_dim*self.act_dim)
                    finite_horizon_correction_eval = I_sa - torch.matrix_power(self.gamma*P, self.horizon)
                    transposed_transition_inverse_eval = torch.inverse(I_sa - self.gamma*P)
                    # accumulated_feature_eval = phi_init_eval @ finite_horizon_correction_eval @ transposed_transition_inverse_eval
                    accumulated_feature_eval = phi_init_eval @ transposed_transition_inverse_eval
                    V_eval = accumulated_feature_eval @ w
                    value_est = torch.mean(V_eval).numpy()
                    q_pi = phi_init_eval @ theta
                    value_est_q = torch.mean(q_pi).numpy()
                    value_est_list_q.append(value_est_q)
                    if not np.isnan(value_est_q) and np.abs(value_est_q)<1e6 and i>=100:
                        value_est_list.append(value_est_q)

                    # if not np.isnan(value_est) and np.abs(value_est)<1e6 and i>=100:
                    #     value_est_list.append(value_est)
                    # if self.debug:
                    #     # #* evaluate mu
                    #     # finite_horizon_correction_eval_mu = I_s - torch.matrix_power(self.gamma*P_mu, self.horizon)
                    #     # transposed_transition_inverse_eval_mu = torch.inverse(I_s - self.gamma*P_mu)
                    #     # accumulated_feature_eval_mu = Z_init_eval @ finite_horizon_correction_eval_mu @ transposed_transition_inverse_eval_mu
                    #     # # accumulated_feature_eval_mu = Z_init_eval @ transposed_transition_inverse_eval_mu
                    #     # V_eval_mu = accumulated_feature_eval_mu @ w_mu
                    #     # value_est_mu = torch.mean(V_eval_mu).numpy()
                    #     # value_est_list_mu.append(value_est_mu)
                    #     #* evaluate via q
                    #     q_pi = phi_init_eval @ theta
                    #     value_est_q = torch.mean(q_pi).numpy()
                    #     value_est_list_q.append(value_est_q)

                    if self.debug:
                        try:
                            operator_norm_I_P = torch.svd(torch.inverse(I_sa - self.gamma*P))[1][0]
                            print('matrix norm for I minus P matrix: ', operator_norm_I_P)
                            print('spectral norm of P', torch.svd(P)[1][0].numpy())
                        except:
                            pdb.set_trace()
                        # print('change in initial features: ', F.l1_loss(Z_init_eval, old_init_feature).numpy()) 
                        # print('change in transition inverse: ', F.l1_loss(transposed_transition_inverse_eval, old_transition).numpy()) 
                        # print('change in P: ', F.l1_loss(P, old_P).numpy())
                        # print('change in r: ', F.l1_loss(w, old_w).numpy())
                        # print('change in feature: ', F.l1_loss(Z_eval, old_feature).numpy())
                        print('latest estimate: ', value_est)                        
                        print('tail average: ', np.mean(value_est_list[-int(tail_average/100 * len(value_est_list)):]))
                        if len(value_est_list) >=10:
                        #     # print('average over most frequent bin: ', choose_estimate_from_sequence(value_est_list))
                            print('most likely estimate: ', select_most_likely_value(value_est_list))
                            print('most likely estimate - q: ', select_most_likely_value(value_est_list_q))
                        #     # print('average over most frequent bin - mu: ', choose_estimate_from_sequence(value_est_list_mu))
                        #     print('most likely estimate - mu: ', select_most_likely_value(value_est_list_mu))
                        # print('latest behavior policy estimate vs MC : ', value_est_mu, torch.mean(V_mu).numpy()*horizon_normalization)
                        print('latest estimate - q', value_est_q)
                        # print('reg factor used', reg)
                        # print('batch size', batch_size)
                        print('\n')
                        old_init_feature = Z_init_eval.clone()
                        old_transition = transposed_transition_inverse_eval.clone()
                        old_P = P.clone()
                        old_w = w.clone()
                        old_feature = Z_eval.clone()
                    self.feature_extractor.train()
                # if i % 200 == 0 and i>0:
                #     pdb.set_trace()
        # pdb.set_trace()
        return np.mean(select_most_likely_value(value_est_list))
        # return np.mean(select_most_likely_value(value_est_list_q))
        








# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# import pdb
# from rl_nexus.utils.optimizer import AdaBound

# dtype = torch.float

# class Feature_Extractor(nn.Module):
#     def __init__(self, input_dim, hidden_nodes, activation):
#         super(Feature_Extractor, self).__init__()
#         self.hidden_layer = nn.Linear(input_dim, hidden_nodes)
#     def forward(self,x):
#         x = self.hidden_layer(x)
#         # x = torch.relu(x)
#         x = torch.tanh(x)
#         # x = F.normalize(x, p=2, dim=1)
#         return x

# class Feature_Learning():
#     def __init__(self, dataset, obs_dim, act_dim, gamma, horizon, 
#                 hidden_nodes, activation, norm = 'std', debug = True):
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.gamma = gamma
#         self.horizon = horizon
#         self.norm = norm
#         self.hidden_nodes = hidden_nodes
#         self.activation = activation
#         self.n_episode =  dataset['init_obs'].shape[0]
#         self.included_idx = torch.arange(dataset['obs'].shape[0])
#         self.end_idx = np.arange(self.horizon-1, dataset['obs'].shape[0], self.horizon)
#         # self.absorbing_idx = np.where(dataset['info'][:,0] == True)[0]
#         self.n_samples = self.included_idx.shape[0]
#         # self.non_absorbing_mask = torch.ones(self.n_samples, dtype=torch.bool)
#         # self.non_absorbing_mask[self.absorbing_idx] = False
#         # self.non_terminal = torch.ones(self.n_samples,1)
#         # self.non_terminal[self.end_idx,0] = 0
#         self.data_acts = torch.tensor(dataset['acts'], dtype=torch.long)[self.included_idx]
#         self.rews = torch.tensor(dataset['rews'], dtype=dtype)[self.included_idx]
#         self.next_acts = torch.tensor(dataset['next_acts'], dtype=torch.long)[self.included_idx]
#         self.init_acts = torch.tensor(dataset['init_acts'], dtype=torch.long)

#         self.pi_current = torch.tensor(dataset['target_prob_obs'],dtype=dtype)[self.included_idx]
#         self.pi_next = torch.tensor(dataset['target_prob_next_obs'], dtype=dtype)[self.included_idx]
#         self.pi_init = torch.tensor(dataset['target_prob_init_obs'], dtype=dtype)
#         self.pi_term = torch.tensor(dataset['target_prob_term_obs'], dtype=dtype)
#         self.mu_prob_act = torch.tensor(dataset['behavior_act_prob'],dtype=dtype)[self.included_idx]

#         self.obs = torch.tensor(dataset['obs'], dtype = dtype)[self.included_idx]
#         self.next_obs = torch.tensor(dataset['next_obs'], dtype=dtype)[self.included_idx]
#         self.init_obs = torch.tensor(dataset['init_obs'], dtype=dtype)
#         self.term_obs = torch.tensor(dataset['term_obs'], dtype=dtype)

#         #* whiten the included observation data
#         obs_mean = torch.mean(self.obs, dim=0, keepdims= True)
#         obs_std = torch.std(self.obs, dim=0, keepdims= True)
#         self.obs = (self.obs - obs_mean) / obs_std
#         self.next_obs = (self.next_obs - obs_mean) / obs_std
#         self.init_obs = (self.init_obs - obs_mean) / obs_std
#         self.term_obs = (self.term_obs - obs_mean) / obs_std

#         self.feature_extractor = Feature_Extractor(input_dim = self.obs_dim, hidden_nodes=hidden_nodes,activation=activation)
#         for param in self.feature_extractor.parameters():
#             torch.nn.init.normal_(param.data, mean=0, std = 1/hidden_nodes)
#         self.debug = debug
#         self.ratio = torch.tensor(dataset['ratio'], dtype = dtype)[self.included_idx]
        
#     def train(self, num_iter = 2000, lr_feature = 1.0e-3, lr_weight = 1.0e-3,batch_size = 500, tail_average=10, reg = 1e-3):
#         if not batch_size:
#             batch_size = self.n_samples #* use the whole batch if no batchsize declared
#         feature_dim =  self.hidden_nodes
#         w = Variable(torch.randn(feature_dim*self.act_dim, 1).type(dtype), requires_grad=True)
#         optimizer = torch.optim.SGD(self.feature_extractor.parameters(), lr = lr_feature)

#         torch.nn.init.normal_(w, mean = 0.0, std = 1/self.hidden_nodes**0.5)
#         # optimizer_w = torch.optim.SGD([w], lr = lr_weight)
#         # optimizer_w = AdaBound([w], lr = lr_weight, final_lr=1.0, gamma=1.0e-4, eps=1.0e-8)
#         optimizer_w = torch.optim.Adam([w], lr = lr_weight)

#         one_hot_acts = torch.squeeze(F.one_hot(self.data_acts, num_classes = self.act_dim))
#         one_hot_init_acts = torch.squeeze(F.one_hot(self.init_acts, num_classes = self.act_dim))
#         I_sa = torch.eye(feature_dim*self.act_dim)
#         I_s = torch.eye(feature_dim)
#         value_est_list = []
#         discount = torch.tensor(np.array([self.gamma ** (i % self.horizon) for i in range(self.n_samples)]).reshape([-1, 1]), dtype=dtype)
#         horizon_normalization = (1-self.gamma**(self.horizon+1)) / (1-self.gamma)

#         Z = self.feature_extractor(self.obs).detach()
#         Z_prime = self.feature_extractor(self.next_obs).detach()
#         # Z_expanded = torch.einsum('ab,ac->abc', Z, acts).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
#         # Phi = torch.einsum('ab,ac->abc', Z, one_hot_acts).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
#         # Phi_prime = torch.einsum('ab,ac->abc', Z_prime, self.pi_next).permute(0,2,1).contiguous().view(batch_size, feature_dim*self.act_dim)
#         Phi = torch.einsum('ab,ac->abc', Z, one_hot_acts).permute(0,2,1).contiguous().view(self.n_samples, feature_dim*self.act_dim)
#         Phi_prime = torch.einsum('ab,ac->abc', Z_prime, self.pi_next).permute(0,2,1).contiguous().view(self.n_samples, feature_dim*self.act_dim)


#         #* First, let's look at the performance of fixed features
#         #* compare the performance of BRM and Fixed Point solution

#         #* fixed point solution:
#         w_ls = torch.inverse(Phi.T @ Phi- self.gamma*Phi.T@Phi_prime + reg *self.n_samples* I_sa) @ Phi.T @self.rews
#         # pdb.set_trace()
#         Z_init_eval = self.feature_extractor(self.init_obs).detach()
#         Phi_init_eval = torch.einsum('ab,ac->abc', Z_init_eval , self.pi_init).permute(0,2,1).contiguous().view(self.n_episode, feature_dim*self.act_dim)
#         V_eval = Phi_init_eval @ w_ls
#         value_est = torch.mean(V_eval.detach()).numpy()
#         print('random feature with least squared fixed point solution: ', value_est)

#         #* BRM solution
#         A = Phi - self.gamma * Phi_prime
#         w_brm = torch.inverse(A.T @ A + reg*self.n_samples* I_sa) @ A.T @ self.rews
#         V_eval = Phi_init_eval @ w_brm
#         value_est = torch.mean(V_eval.detach()).numpy()
#         print('random feature with BRM solution: ', value_est)

#         pdb.set_trace()

#         for i in range(num_iter):
#             perm = torch.randperm(self.n_samples)
#             num_batches = self.n_samples // batch_size
#             current_loss = 0
#             for j in range(num_batches):
#                 idx = perm[j*batch_size:(j+1)*batch_size]
#                 phi = Phi[idx]
#                 phi_prime = Phi_prime[idx]
#                 rews = self.rews[idx]
#                 # target = (rews + phi_prime @ w).detach()
#                 # loss = 1/batch_size * torch.norm(phi @ w - target)**2 + 1.0e-5 * torch.norm(w)**2
#                 # loss = 1/batch_size * torch.norm((phi-self.gamma*phi_prime) @ w - rews)**2
#                 target = phi.T @ rews
#                 A = phi.T @ (phi - self.gamma * phi_prime)
#                 loss = 1/batch_size * torch.norm(A @ w - target)**2
#                 optimizer_w.zero_grad()
#                 loss.backward()
#                 # pdb.set_trace()
#                 optimizer_w.step()
#                 current_loss += loss.clone().detach()
#             current_loss /= num_batches
#             if i % 10 == 0:
#                 print('iteration ', i)
#                 print('current loss value: ', current_loss.numpy())
#                 V_eval = Phi_init_eval @ w
#                 value_est = torch.mean(V_eval.detach()).numpy()
#                 print('max weight: ', torch.max(w))
#                 print('random feature with SGD: ', value_est)
#                 print('\n')

#         return value_est

