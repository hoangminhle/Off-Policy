import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import pdb

class LSTDQ_Kernel():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon,
                value_reg, 
                default_length_scale = 0.2,
                random_feature_per_obs_dim = 250,
                norm = None,
                scale_length_adjustment = 'median', 
                dtype = np.float32,
                policy_net = None,
                separate_action_indexing = False,
                action_encoding_scheme = 'continuous'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.value_reg = value_reg
        self.dtype = dtype
        self.separate_action_indexing = separate_action_indexing
        self.action_encoding_scheme = action_encoding_scheme

        self.n_samples = dataset['obs'].shape[0]
        self.n_episode =  dataset['init_obs'].shape[0]
        

        self.non_terminal_idx = (dataset['info']==False)[:,0]
        self.n_samples_non_terminal = self.non_terminal_idx.sum()
        self.data_acts = dataset['acts'][self.non_terminal_idx]

        if self.policy_net is not None:
            self.pi_current = self.policy_net.get_probabilities(dataset['obs'])
            self.pi_next = self.policy_net.get_probabilities(dataset['next_obs'])
            self.pi_init = self.policy_net.get_probabilities(dataset['init_obs'])
            self.pi_term = self.policy_net.get_probabilities(dataset['term_obs'])
        else:
            self.pi_current = dataset['target_prob_obs'][self.non_terminal_idx]
            self.pi_next = dataset['target_prob_next_obs'][self.non_terminal_idx]
            self.pi_init = dataset['target_prob_init_obs']
            self.pi_term = dataset['target_prob_term_obs']
        if self.norm is None:
            self.obs = dataset['obs'][self.non_terminal_idx]
            self.next_obs = dataset['next_obs'][self.non_terminal_idx]
            self.init_obs = dataset['init_obs']
            self.term_obs = dataset['term_obs']
        elif self.norm == 'std':
            self.obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            self.obs_std = np.std(dataset['obs'], axis=0, keepdims=True)
            self.obs = (dataset['obs'] - self.obs_mean) / self.obs_std
            self.next_obs = (dataset['next_obs'] - self.obs_mean) / self.obs_std
            self.init_obs = (dataset['init_obs'] - self.obs_mean) / self.obs_std
            self.term_obs = (dataset['term_obs'] - self.obs_mean) / self.obs_std
        else:
            raise NotImplementedError
        # pdb.set_trace()
        #* what if we only whiten over the non-terminal tuples
        non_terminal_idx = (dataset['info']==False)[:,0]
        obs_mean = np.mean(dataset['obs'][non_terminal_idx], axis=0, keepdims= True)
        obs_std = np.std(dataset['obs'][non_terminal_idx], axis=0, keepdims= True)
        # #* re-whiten the observations:
        self.obs = (self.obs - obs_mean) / obs_std
        self.next_obs = (self.next_obs - obs_mean) / obs_std
        self.init_obs = (self.init_obs - obs_mean) / obs_std
        self.term_obs = (self.term_obs - obs_mean) / obs_std

        #* if not separate action indexing, we are concatenating (s,a) as input
        if not self.separate_action_indexing:
            if self.action_encoding_scheme == 'continuous':
                encoded_actions = np.linspace(-1,1, self.act_dim)
                # mean_action = np.mean(encoded_actions[self.data_acts[non_terminal_idx]])
                # std_action = np.std(encoded_actions[self.data_acts[non_terminal_idx]])
                mean_action = np.mean(encoded_actions[self.data_acts])
                std_action = np.std(encoded_actions[self.data_acts])

                self.encoded_actions = (encoded_actions - mean_action)/ std_action
                
                # self.act = (self.data_acts / (self.act_dim-1)) * 2 -1
                # self.act = (self.act - np.mean(self.act, axis=0, keepdims=True))/np.std(self.act, axis=0, keepdims=True)
                self.act = self.encoded_actions[self.data_acts]

                self.input = np.concatenate((self.obs, self.act), axis=1)
                self.input_dim = self.input.shape[1]
            else:
                raise NotImplementedError
        else: 
            self.input = self.obs
            self.input_dim = self.obs.shape[1]

        if scale_length_adjustment == 'median':
            sample_num = 5000
            # idx1 = np.random.choice(self.n_samples, sample_num); idx2 = np.random.choice(self.n_samples, sample_num)
            # idx1 = np.random.choice(np.arange(self.n_samples)[non_terminal_idx], sample_num); idx2 = np.random.choice(np.arange(self.n_samples)[non_terminal_idx], sample_num)
            idx1 = np.random.choice(self.n_samples_non_terminal, sample_num); idx2 = np.random.choice(self.n_samples_non_terminal, sample_num)
            # med_dist = np.median(np.square(self.obs[None, idx1, :] - self.obs[idx2, None, :]), axis = (0,1))
            med_dist = np.median(np.square(self.input[None, idx1, :] - self.input[idx2, None, :]), axis = (0,1))
            med_dist[med_dist<0.01] = 0.01 # enforce a upperbound on the scale-length of the action component
            self.scale_length_vector = 1.0/med_dist
        else:
            # scale_length_vector = np.ones(self.obs_dim)
            self.scale_length_vector = np.ones(self.input_dim)

        # self.scale_length_vector = np.linspace(1,2,5)
        self.scale_length_vector = np.ones(self.input_dim)
        self.z_dim = random_feature_per_obs_dim * self.input_dim
        self.rff = RBFSampler(n_components = self.z_dim, gamma = default_length_scale)
        self.rff.fit([self.input[0]])
        # #* set the fourier feature
        # transformer_list = []
        # # self.z_dim = random_feature_per_obs_dim * self.obs_dim
        # self.z_dim = random_feature_per_obs_dim * self.input_dim
        # models = [RBFSampler(n_components = random_feature_per_obs_dim, gamma = default_length_scale*dist) for dist in self.scale_length_vector]
        # for model in models:
        #     # model.fit([self.obs[0]])
        #     model.fit([self.input[0]])
        #     transformer_list.append((str(model), model))
        # self.rff = FeatureUnion(transformer_list)

        # models = [RBFSampler(n_components = random_feature_per_obs_dim, gamma = default_length_scale*dist) for dist in self.scale_length_vector]
        # for model in models:
        #     # model.fit([self.obs[0]])
        #     model.fit([self.input[0]])
        #     transformer_list.append((str(model), model))
        # self.rff = [RBFSampler(n_components = random_feature_per_obs_dim, gamma = default_length_scale)]
        # self.rff.fit([self.input[0]])

        #* Some commonly used variables
        # self.I_sa = np.eye(self.act_dim*self.z_dim)
        self.rews = dataset['rews'][self.non_terminal_idx]
        # self.init_idx = np.arange(0, self.n_samples, self.horizon)
        # self.end_idx = np.arange(self.horizon-1, self.n_samples, self.horizon)

        self.rho = dataset['ratio'][self.non_terminal_idx] #* make sure that the importance weights are already calculated
        # pdb.set_trace()

    def estimate(self):
        if self.separate_action_indexing:
            value_est = self.estimate_LSTDQ_separate_action_indexing()
        else:
            value_est = self.estimate_LSTDQ_concat_sa_input()
        return value_est

    def estimate_LSTDQ_concat_sa_input(self):
        # transformed_action = np.linspace(-1,1, self.act_dim)
        # n_samples = self.non_terminal_idx.sum()
        a_prime = np.tile(self.encoded_actions, self.n_samples_non_terminal)[:,np.newaxis]
        # a_prime = np.tile(self.encoded_actions, self.n_samples)[:,np.newaxis]
        x_prime = np.concatenate((np.repeat(self.next_obs, self.act_dim, axis=0), a_prime), axis=1)
        # a0_expanded = np.tile(transformed_action,self.n_episode)[:,np.newaxis]
        a0_expanded = np.tile(self.encoded_actions,self.n_episode)[:,np.newaxis]
        x0 = np.concatenate((np.repeat(self.init_obs, self.act_dim, axis=0), a0_expanded), axis=1)
        # aterm_expanded = np.tile(transformed_action, self.n_episode)[:,np.newaxis]
        aterm_expanded = np.tile(self.encoded_actions, self.n_episode)[:,np.newaxis]
        xterm = np.concatenate((np.repeat(self.term_obs, self.act_dim, axis=0), aterm_expanded), axis=1)

        Z = self.rff.transform(self.input).astype(self.dtype)
        Z_prime = self.rff.transform(x_prime).astype(self.dtype)
        aprime_probs = self.pi_next.flatten()[:,np.newaxis]
        Z_prime = Z_prime * aprime_probs
        Z_prime = Z_prime.reshape((self.n_samples_non_terminal, self.act_dim, self.z_dim)).sum(axis=1)

        reg = self.value_reg

        regularized_inverse = np.linalg.inv( np.matmul(Z.T, Z - self.gamma*Z_prime) + reg*np.eye(self.z_dim))
        featurized_reward = np.matmul(Z.T, self.rews)
        value_coef = np.matmul(regularized_inverse, featurized_reward)

        Z0 = self.rff.transform(x0)
        Q0 = np.matmul(Z0, value_coef)

        Z_term = self.rff.transform(xterm)
        Q_term = np.matmul(Z_term, value_coef)

        V_init = (Q0 * self.pi_init.flatten()[:,np.newaxis]).reshape((self.n_episode, self.act_dim)).sum(axis=1)
        V_term = (Q_term * self.pi_term.flatten()[:, np.newaxis]).reshape((self.n_episode, self.act_dim)).sum(axis=1)
        V_traj = V_init - V_term*self.gamma**self.horizon
        value_est = np.mean(V_traj)
        # pdb.set_trace()
        return value_est

    def estimate_LSTDQ_separate_action_indexing(self):
        #* separate action set indexing
        act_idx = []
        for i in range(self.act_dim):
            act_idx.append(np.where(self.data_acts==i)[0])
        #* apply transformation
        Z = self.rff.transform(self.obs).astype(self.dtype); Z_prime = self.rff.transform(self.next_obs).astype(self.dtype)
        Z_init = self.rff.transform(self.init_obs).astype(self.dtype); Z_term = self.rff.transform(self.term_obs).astype(self.dtype)
        # import pdb; pdb.set_trace()
        assert self.z_dim == Z.shape[1]
        Phi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim), dtype=self.dtype)
        Phi_pi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim),dtype=self.dtype)
        Phi_prime_pi = np.zeros((Z_prime.shape[0], Z_prime.shape[1]* self.act_dim),dtype=self.dtype)
        Phi_init_pi = np.zeros((Z_init.shape[0], Z_init.shape[1]*self.act_dim), dtype=self.dtype)
        Phi_term_pi = np.zeros((Z_term.shape[0], Z_term.shape[1]*self.act_dim),dtype=self.dtype)
        for i in range(self.act_dim):
            Phi[act_idx[i], i*self.z_dim:(i+1)*self.z_dim] = Z[act_idx[i]]
            Phi_pi[:, i*self.z_dim:(i+1)*self.z_dim] = self.pi_current[:,i][:,None] * Z        
            Phi_prime_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_next[:,i][:,None] * Z_prime
            Phi_init_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_init[:,i][:,None]*Z_init
            Phi_term_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_term[:,i][:,None]*Z_term
        
        I_sa = np.eye(self.act_dim*self.z_dim)

        regularized_inverse = np.linalg.inv( np.matmul(Phi.T, Phi-self.gamma*Phi_prime_pi) + self.value_reg*I_sa)
        featurized_reward = np.matmul(Phi.T, self.rews)
        reward_coef = np.matmul(regularized_inverse, featurized_reward)
        V_init = Phi_init_pi @ reward_coef
        V_term = Phi_term_pi @ reward_coef
        V_traj = V_init - V_term*self.gamma**self.horizon
        value_est = np.mean(V_traj)
        # import pdb; pdb.set_trace()
        return value_est
