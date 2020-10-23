import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import pdb

class Model_Based_Kernel():
    def __init__(self, dataset, obs_dim, act_dim, gamma, horizon,
                model_reg, reward_reg, 
                default_length_scale = 0.2,
                random_feature_per_obs_dim = 250,
                norm = None,
                scale_length_adjustment = 'median', 
                dtype = np.float32,
                policy_net = None,
                separate_action_indexing = True,
                action_encoding_scheme = 'continuous'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.horizon = horizon
        self.norm = norm
        self.policy_net = policy_net
        self.model_reg = model_reg
        self.reward_reg = reward_reg
        self.dtype = dtype
        self.separate_action_indexing = separate_action_indexing
        self.action_encoding_scheme = action_encoding_scheme

        self.n_samples = dataset['obs'].shape[0]
        self.n_episode =  dataset['init_obs'].shape[0]
        

        # self.non_terminal_idx = (dataset['info']==False)[:,0]
        self.non_terminal_idx = np.arange(dataset['obs'].shape[0])
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

        self.default_length_scale = default_length_scale
        self.z_dim = random_feature_per_obs_dim * self.obs_dim

        self.rff = RBFSampler(n_components = self.z_dim, gamma = default_length_scale)
        self.rff.fit([self.obs[0]])

        self.rews = dataset['rews'][self.non_terminal_idx]

        self.rho = dataset['ratio'][self.non_terminal_idx] #* make sure that the importance weights are already calculated

    def estimate(self):
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
        # H = np.eye(self.n_samples)
        # H = np.eye(self.n_samples) - 1.0/self.n_samples*np.ones((self.n_samples, self.n_samples))
        #! Important note: we will ignore H here to avoid memory error

        #* estimate reward function
        r_sa = np.linalg.inv(Phi.T @ Phi + self.reward_reg*I_sa) @ Phi.T @ self.rews
        # r_sa = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ self.rews
        # Sigma_yx = 1/self.n_samples*Phi_prime_pi.T @ H @ Phi 
        # Sigma_xx = 1/self.n_samples*Phi.T @ H @ Phi
        Sigma_yx = 1/self.n_samples*Phi_prime_pi.T @ Phi 
        Sigma_xx = 1/self.n_samples*Phi.T @ Phi
        P = np.matmul(Sigma_yx, np.linalg.inv(Sigma_xx+ self.model_reg*I_sa))
        #* Now that we have the transition operator, we have that:
        #* E_{s'|s}[\phi(s')|s] = P \phi(s)
        #* This gives a clean mechanism to roll the model forward 
        #* in particular, the next feature matrix will be 
        #* Phi' = Phi P.T, where Phi = [phi_1, ..., phi_n].T \in R^{n\times p}
        finite_horizon_correction = I_sa - np.linalg.matrix_power(self.gamma*P.T, self.horizon)
        transposed_transition_inverse = np.linalg.inv(I_sa - self.gamma*P.T)
        # accumulated_feature = Phi_pi @ finite_horizon_correction @ transposed_transition_inverse
        accumulated_feature = Phi_init_pi @ finite_horizon_correction @ transposed_transition_inverse
        V = accumulated_feature @ r_sa
        # pdb.set_trace()
        value_est = np.mean(V)
        return value_est


