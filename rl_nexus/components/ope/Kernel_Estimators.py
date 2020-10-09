import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

class Kernel_Estimators(object):
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon,
                model_reg, reward_reg,value_reg,
                default_length_scale = 0.1,
                random_feature_per_obs_dim = 250,
                norm = None,
                scale_length_adjustment = 'median', 
                dtype = np.float64,
                policy_net = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.model_reg = model_reg
        self.reward_reg = reward_reg
        self.value_reg = value_reg
        self.dtype = dtype
        
        self.n_samples = dataset['obs'].shape[0]
        self.n_episode =  dataset['init_obs'].shape[0]
        self.data_acts = dataset['acts']
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
        if self.norm is None:
            self.obs = dataset['obs']
            self.next_obs = dataset['next_obs']
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

        # #* separate action set indexing
        # act_idx = []
        # for i in range(self.act_dim):
        #     act_idx.append(np.where(dataset['acts']==i)[0])
        # #* apply transformation
        # Z = self.rff.transform(self.obs).astype(self.dtype); Z_prime = self.rff.transform(self.next_obs).astype(self.dtype)
        # Z_init = self.rff.transform(self.init_obs).astype(self.dtype); Z_term = self.rff.transform(self.term_obs).astype(self.dtype)
        # assert self.z_dim == Z.shape[1]
        # self.Phi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim), dtype=self.dtype)
        # self.Phi_pi = np.zeros((Z.shape[0], Z.shape[1]* self.act_dim),dtype=self.dtype)
        # self.Phi_prime_pi = np.zeros((Z_prime.shape[0], Z_prime.shape[1]* self.act_dim),dtype=self.dtype)
        # self.Phi_init_pi = np.zeros((Z_init.shape[0], Z_init.shape[1]*self.act_dim), dtype=self.dtype)
        # self.Phi_term_pi = np.zeros((Z_term.shape[0], Z_term.shape[1]*self.act_dim),dtype=self.dtype)
        # for i in range(self.act_dim):
        #     self.Phi[act_idx[i], i*self.z_dim:(i+1)*self.z_dim] = Z[act_idx[i]]
        #     self.Phi_pi[:, i*self.z_dim:(i+1)*self.z_dim] = self.pi_current[:,i][:,None] * Z        
        #     self.Phi_prime_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_next[:,i][:,None] * Z_prime
        #     self.Phi_init_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_init[:,i][:,None]*Z_init
        #     self.Phi_term_pi[:,i*self.z_dim:(i+1)*self.z_dim] = self.pi_term[:,i][:,None]*Z_term


        #* Some commonly used variables
        # self.I_sa = np.eye(self.act_dim*self.z_dim)
        self.rews = dataset['rews']
        self.init_idx = np.arange(0, self.n_samples, self.horizon)
        self.end_idx = np.arange(self.horizon-1, self.n_samples, self.horizon)

        self.rho = dataset['ratio'] #* make sure that the importance weights are already calculated


    def estimate_model_based(self):
        #* separate action set indexing
        act_idx = []
        for i in range(self.act_dim):
            act_idx.append(np.where(self.data_acts==i)[0])
        #* apply transformation
        Z = self.rff.transform(self.obs).astype(self.dtype); Z_prime = self.rff.transform(self.next_obs).astype(self.dtype)
        Z_init = self.rff.transform(self.init_obs).astype(self.dtype); Z_term = self.rff.transform(self.term_obs).astype(self.dtype)
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
        #* uncentered /center covariance identity:
        H = np.eye(self.n_samples)
        # H = np.eye(self.n_samples) - 1.0/self.n_samples*np.ones((self.n_samples, self.n_samples))

        #* estimate reward function
        r_sa = np.linalg.inv(Phi.T @ Phi + self.reward_reg*I_sa) @ Phi.T @ self.rews
        Sigma_yx = 1/self.n_samples*Phi_prime_pi.T @ H @ Phi 
        Sigma_xx = 1/self.n_samples*Phi.T @ H @ Phi
        P = np.matmul(Sigma_yx, np.linalg.inv(Sigma_xx+ self.model_reg*I_sa))
        #* Now that we have the transition operator, we have that:
        #* E_{s'|s}[\phi(s')|s] = P \phi(s)
        #* This gives a clean mechanism to roll the model forward 
        #* in particular, the next feature matrix will be 
        #* Phi' = Phi P.T, where Phi = [phi_1, ..., phi_n].T \in R^{n\times p}
        finite_horizon_correction = I_sa - np.linalg.matrix_power(self.gamma*P.T, self.horizon)
        transposed_transition_inverse = np.linalg.inv(I_sa - self.gamma*P.T)
        accumulated_feature = Phi_pi @ finite_horizon_correction @ transposed_transition_inverse
        
        V = accumulated_feature @ r_sa
        value_est = np.mean(V[self.init_idx])
        return value_est

    def estimate_LSTD(self):
        reg = self.value_reg
        Z = self.rff.transform(self.obs); Z_prime = self.rff.transform(self.next_obs)
        R = self.rho * self.rews
        regularized_inverse = np.linalg.inv( np.matmul(Z.T, Z-self.gamma*self.rho*Z_prime) + reg*np.eye(self.z_dim))
        featurized_reward = np.matmul(Z.T, R)
        reward_coef = np.matmul(regularized_inverse, featurized_reward)
        V_init = Z[self.init_idx] @ reward_coef
        V_term = Z[self.end_idx] @ reward_coef
        V_traj = V_init - V_term*self.gamma**self.horizon
        value_est = np.mean(V_traj)
        return value_est

    def estimate_LSTDQ(self):
        #* separate action set indexing
        act_idx = []
        for i in range(self.act_dim):
            act_idx.append(np.where(self.data_acts==i)[0])
        #* apply transformation
        Z = self.rff.transform(self.obs).astype(self.dtype); Z_prime = self.rff.transform(self.next_obs).astype(self.dtype)
        Z_init = self.rff.transform(self.init_obs).astype(self.dtype); Z_term = self.rff.transform(self.term_obs).astype(self.dtype)
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
        return value_est

    def estimate_LSTD_dual(self):
        import kernel_util as ku
        sample_num = 5000
        idx1 = np.random.choice(self.n_samples, sample_num); idx2 = np.random.choice(self.n_samples, sample_num)
        med_dist = np.median(np.square(self.obs[None, idx1, :] - self.obs[idx2, None, :]), axis = (0,1))
        med_dist[med_dist<0.01] = 0.01 # enforce a upperbound on the scale-length of the action component
        w = 1.0/med_dist

        default_gamma = 0.1
        reg = 1e-2
        
        ratio_vector = self.rho.copy().astype(np.float32)

        K = ku.weighted_rbf_kernel(self.obs, w=w, gamma = default_gamma).astype(np.float32)
        K_prime = ku.weighted_rbf_kernel(self.next_obs, self.obs, w=w, gamma = default_gamma).astype(np.float32)
        K_prime = self.gamma * (K_prime * ratio_vector.repeat(self.n_samples, axis=1))
        R = (ratio_vector*self.rews).astype(np.float32)
        beta = np.linalg.inv(K-K_prime + reg*np.eye(self.n_samples)).dot(R)
        
        K0 = ku.weighted_rbf_kernel(self.obs, self.init_obs, w=w, gamma=default_gamma)
        K_terminal = ku.weighted_rbf_kernel(self.obs, self.term_obs, w=w, gamma=default_gamma)
        
        # V_init = np.matmul(beta.T, K0)        
        # V_term = np.matmul(beta.T, K_terminal)
        V_init = K0.T @ beta 
        V_term = K_terminal.T @ beta
        V_traj = V_init - V_term*self.gamma**self.horizon
        value_est = np.mean(V_traj)
        import pdb; pdb.set_trace()
        return value_est







            