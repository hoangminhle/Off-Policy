import numpy as np
from rl_nexus.utils.ope_utils import set_seed, read_batch_experience, evaluate_on_policy,\
    sample_data, est_med_dist, convert_torch_model_weights_to_list, summarize_data
from rl_nexus.utils.utils import ensure_dir_exists
from rl_nexus.utils.ope_logger import logger
from rl_nexus.utils.data_size_check import total_size
import pdb
import random
import pickle
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from rl_nexus.utils.metric import Metric
import ray

class Off_Policy_Evaluation():
    def __init__(self, spec_tree, device):
        self.spec_tree = spec_tree
        
        self.seed = spec_tree['algo_seed']

        self.dataset_seed = spec_tree['dataset_seed']
        self.use_ray = spec_tree['use_ray']

        self.ope = spec_tree['offline_estimators']
        assert isinstance(self.ope, list), 'the estimators should be described as a list'
        # self.metrics = {}
        # for estimator in self.ope:
        #     metric = Metric(short_name=estimator, long_name=estimator,formatting_string='{:5.2f}', higher_is_better=False)
        #     self.metrics[estimator] = metric
        self.horizon = spec_tree['horizon']
        self.num_episodes = spec_tree['num_episodes']
        self.gamma = spec_tree['gamma']
        self.normalization = spec_tree['normalization']
        self.target_temp = spec_tree['target_policy_temperature']

        # reset the environment seed
        spec_tree['environment']['seed'] = self.seed
        assert self.horizon == spec_tree['environment']['max_ep_len'], 'horizon and max_ep_len of environment do not match'
        assert not spec_tree['environment']['fixed_length_episode'], 'should turn fixed length episode to false for on policy evaluation purpose'
        self.environment = spec_tree.create_component('environment')
        obs_space = self.environment.observation_space
        action_space = self.environment.action_space
        
        self.obs_dim = obs_space.shape[0]
        self.act_dim = action_space.n

       #* prepare the result path
        self.behavior_policy_type = spec_tree['behavior_policy_type']
        if self.behavior_policy_type == 'random':
            behavior_type_string = '/behavior_uniform_random'
        elif self.behavior_policy_type == 'random_network':
            behavior_type_string = '/behavior_random_network'
        elif self.behavior_policy_type == 'epsilon_greedy':
            self.behavior_min_id = spec_tree['behavior_policy_range']['min']
            self.behavior_max_id = spec_tree['behavior_policy_range']['max']
            behavior_type_string = 'behavior_epsilon_greedy_'+str(self.behavior_min_id)+\
                '_'+str(self.behavior_max_id)
        else:
            self.behavior_min_id = spec_tree['behavior_policy_range']['min']
            self.behavior_max_id = spec_tree['behavior_policy_range']['max']
            behavior_type_string = '/behavior_'+str(self.behavior_min_id)+\
                '_'+str(self.behavior_max_id)
        
        self.debug_mode = spec_tree['debug_mode']
        self.result_path_prefix = spec_tree['save_results_to'] + behavior_type_string + '/target_{}_temp{}/n_eps{}/horizon{}/seed'.format(\
            spec_tree['target_policy_id'], self.target_temp, self.num_episodes, self.horizon)
        
        ensure_dir_exists(file=self.result_path_prefix)

        #* prepare the data path (if reading from an external data set)
        self.read_data_from_file = spec_tree['read_data_from_file']
        self.dataset_path_prefix = None

        if self.read_data_from_file:
            self.dataset_path_prefix = spec_tree['load_data_from'] + behavior_type_string +'/n_eps{}/horizon{}/seed'.format(\
                self.num_episodes, self.horizon)

        target_policy_net_path = spec_tree['load_model_from']+ '/policy_' +\
             str(spec_tree['target_policy_id'])+'.pt'        
        # self.tf_policy_net = convert_policy_network(spec_tree,obs_space,action_space, temperature = self.target_temp, path = target_policy_net_path)
        self.target_policy_net = spec_tree.create_component('model', obs_space, action_space)
        self.target_policy_net.temperature = self.target_temp
        self.target_policy_net.load_model(target_policy_net_path)
        # self.target_model_weights_list = extract_model_weights(self.target_policy_net.model, self.obs_dim, self.act_dim)
        self.target_model_weights_list = convert_torch_model_weights_to_list(self.target_policy_net.model)
        
        on_policy_num_eps = spec_tree['on_policy_eval_num_episodes']
        self.value_true = evaluate_on_policy(self.environment, self.target_policy_net, num_episodes = on_policy_num_eps, gamma = self.gamma)
        
        self.hidden_layers_net = self.spec_tree['model']['fcnet_hiddens']
        self.activation_net = self.spec_tree['model']['fcnet_activation']

        self.data = None
    # # def prepare_data(self, dataset_seed):
    # def run_estimators(self):
    #     # get the data
    #     data
    #     # run sequential experiments

    #     pass
    def load_data(self, data=None):
        if not self.read_data_from_file:
            assert data is not None
            assert data['metadata']['dataset_seed'] == self.dataset_seed
            self.data = data
            self.data['factor'] = self.gamma**data['time_step']
            # pdb.set_trace()
            self.data['target_act_prob'] = self.target_policy_net.get_prob_with_act(data['obs'], data['acts'])
            self.data['target_prob_obs'] = self.target_policy_net.get_probabilities(data['obs'])
            self.data['target_prob_next_obs'] = self.target_policy_net.get_probabilities(data['next_obs'])
            self.data['ratio'] = self.data['target_act_prob'] / self.data['behavior_act_prob']
            self.data['target_prob_init_obs'] = self.data['target_prob_obs'][::self.horizon]
            self.data['target_prob_term_obs'] = self.data['target_prob_next_obs'][self.horizon-1::self.horizon]
        else:
            assert self.dataset_path_prefix is not None
            dataset_path = self.dataset_path_prefix + str(self.dataset_seed) + '.pickle'
            self.data = read_batch_experience(dataset_path, self.target_policy_net, self.num_episodes, self.target_temp, self.horizon, self.gamma)
        if self.normalization == 'std_norm':
            # #* whiten data
            # obs_mean = np.mean(self.data['obs'], axis=0, keepdims = True)
            # obs_std = np.std(self.data['obs'], axis = 0, keepdims = True)
            #* whiten data over non-terminal indices
            non_terminal_idx = (self.data['info'] == False)[:,0]
            obs_mean = np.mean(self.data['obs'][non_terminal_idx], axis=0, keepdims = True)
            obs_std = np.std(self.data['obs'][non_terminal_idx], axis = 0, keepdims = True)
            self.data['obs'] = (self.data['obs'] - obs_mean) / obs_std
            self.data['next_obs'] = (self.data['next_obs'] - obs_mean) / obs_std
            self.data['init_obs'] = (self.data['init_obs'] - obs_mean) / obs_std
            self.data['term_obs'] = (self.data['term_obs'] - obs_mean) / obs_std

            self.norm_performed = {'type': self.normalization, 'shift': obs_mean, 'scale': obs_std}
        else:
            self.norm_performed = {'type': None, 'shift': None, 'scale': None}
        data_size = total_size(self.data) / 1024**2
        summarize_data(self.data, self.result_path_prefix, save_fig = False)
        print('Processed data size: {:2f} MB'.format(data_size))

        discount_factor = np.array([self.gamma ** (i % self.horizon) for i in range(self.data['obs'].shape[0])]).reshape([-1, 1])
        self.behavior_value_est = (self.data['rews'] * discount_factor).sum()/self.num_episodes
        # pdb.set_trace()

    def prepare_data(self, dataset_seed):
        # locate off-line dataset
        dataset_path = self.dataset_path_prefix + str(dataset_seed) + '.h5'
        data = read_batch_experience(dataset_path, self.target_policy_net, self.num_episodes, self.target_temp, self.horizon, self.gamma)
        if self.normalization == 'std_norm':
            #* whiten data
            obs_mean = np.mean(data['obs'], axis=0, keepdims = True)
            obs_std = np.std(data['obs'], axis = 0, keepdims = True)
            data['obs'] = (data['obs'] - obs_mean) / obs_std
            data['next_obs'] = (data['next_obs'] - obs_mean) / obs_std
            data['init_obs'] = (data['init_obs'] - obs_mean) / obs_std
            data['term_obs'] = (data['term_obs'] - obs_mean) / obs_std

            self.norm_performed = {'type': self.normalization, 'shift': obs_mean, 'scale': obs_std}
        else:
            self.norm_performed = {'type': None, 'shift': None, 'scale': None}
        # data['init_obs'] = data['obs'][::self.horizon]
        # data['init_acts'] = data['acts'][::self.horizon]
        # data['term_obs'] = data['next_obs'][self.horizon-1::self.horizon]
        data['target_prob_init_obs'] = data['target_prob_obs'][::self.horizon]
        data['target_prob_term_obs'] = data['target_prob_next_obs'][self.horizon-1::self.horizon]
        return data
    
    #* duplicate another version for ray remote below
    def get_estimate(self, estimator):
        if estimator == 'MWL':
            value_est = self.run_MWL()
        elif estimator == 'MSWL':
            value_est = self.run_MSWL()
        elif estimator == 'MQL':
            value_est = self.run_MQL()
        elif estimator == 'DualDICE':
            value_est = self.run_DualDICE()
        elif estimator == 'TDREG-K':
            value_est = self.run_TDREG_Kernel()
        elif estimator == 'TDREG-N':
            value_est = self.run_TDREG_Neural()
        elif estimator == 'FQE':
            value_est = self.run_FQE()
        elif estimator == 'MB-K':
            value_est = self.run_MB_Kernel()
        elif estimator == 'MB-N':
            value_est = self.run_MB_Neural()
        elif estimator == 'LSTD':
            value_est = self.run_LSTD()
        elif estimator == 'LSTDQ':
            value_est = self.run_LSTDQ()
        elif estimator == 'PDIS':
            value_est = self.run_PDIS()
        elif estimator == 'WPDIS':
            value_est = self.run_WPDIS()
        elif estimator == 'FQE':
            value_est = self.run_FQE()
        else:
            raise NotImplementedError
        return value_est

    @ray.remote
    def get_estimate_ray(self, estimator):
        if estimator == 'MWL':
            value_est = self.run_MWL()
        elif estimator == 'MSWL':
            value_est = self.run_MSWL()
        elif estimator == 'MQL':
            value_est = self.run_MQL()
        elif estimator == 'DualDICE':
            value_est = self.run_DualDICE()
        elif estimator == 'TDREG-K':
            value_est = self.run_TDREG_Kernel()
        elif estimator == 'TDREG-N':
            value_est = self.run_TDREG_Neural()
        elif estimator == 'FQE':
            value_est = self.run_FQE()
        elif estimator == 'MB-K':
            value_est = self.run_MB_Kernel()
        elif estimator == 'MB-N':
            value_est = self.run_MB_Neural()
        elif estimator == 'LSTD':
            value_est = self.run_LSTD()
        elif estimator == 'LSTDQ':
            value_est = self.run_LSTDQ()
        elif estimator == 'PDIS':
            value_est = self.run_PDIS()
        elif estimator == 'WPDIS':
            value_est = self.run_WPDIS()
        elif estimator == 'FQE':
            value_est = self.run_FQE()
        else:
            raise NotImplementedError
        return value_est

    def run_estimators(self):
        # self.data = self.prepare_data(dataset_seed)
        assert self.data is not None
        result = {}
        result['On_Policy'] = self.value_true
        result['Behavior'] = self.behavior_value_est
        #* Do we run kernel-based methods?
        # if 'MB' in self.ope or 'LSTD' in self.ope or 'LSTDQ' in self.ope:
        if 'LSTD' in self.ope: # handle lstdq separately for now for tuning
            self.initialize_kernel_estimator()
        
        if self.spec_tree['use_ray']:
            ray.init()
            result_ids = []
            for estimator in self.ope:
                # value_est = self.get_estimate(estimator, data)
                # value_est = Off_Policy_Evaluation.get_estimate.remote(self, estimator, data)
                result_ids.append(Off_Policy_Evaluation.get_estimate_ray.remote(self, estimator))
                # result[estimator] = value_est
                # logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(dataset_seed, \
                #     self.value_true, estimator, value_est))
            
            unordered_result = ray.get(result_ids)
            ray.shutdown()

            for item in unordered_result:
                estimator = item[0]
                value_est = item[1]
                result[estimator] = value_est
                logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(self.dataset_seed, \
                    self.value_true, estimator, value_est))
        else:
            for estimator in self.ope:
                result_tuple = self.get_estimate(estimator)
                value_est = result_tuple[1] #* the value est occupies the 2nd position
                result[estimator] = value_est
                logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(self.dataset_seed, \
                    self.value_true, estimator, value_est))
                # logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}, {:.2f}'.format(self.dataset_seed, \
                #     self.value_true, estimator, value_est[0], value_est[1]))

        error_metrics = {}
        for estimator in self.ope:
            error_metrics[estimator] = abs((result[estimator] - self.value_true)/ self.value_true)
        logger.write_ope_metrics(self.dataset_seed, error_metrics, result)

        return result

    def execute(self):
        set_seed(self.seed)

        # dataset_seed = self.dataset_seed
        result = self.run_estimators()
        #* save the result
        if not self.debug_mode:
            result_path = self.result_path_prefix + '{}.pickle'.format(self.dataset_seed)
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
        # logger.write_and_condense_metrics()
        return result

    def run_MSWL_torch(self):
        from rl_nexus.components.ope.MSWL_Kernel import MSWL
        k_tau = self.spec_tree['MSWL']['k_tau']
        lr = self.spec_tree['MSWL']['lr']
        reg = self.spec_tree['MSWL']['reg']
        num_iter = self.spec_tree['MSWL']['num_iter']
        batch_size = self.spec_tree['MSWL']['batch_size']
        hidden_layers = self.spec_tree['MSWL']['hidden_layers']
        eval_interval = self.spec_tree['MSWL']['eval_interval']
        tail_average = self.spec_tree['MSWL']['tail_average']

        mswl = MSWL(self.data, self.obs_dim, self.act_dim, k_tau, norm = None, hidden_layers=hidden_layers,\
            lr = lr, reg_factor=reg, gamma = self.gamma)
        value_est = mswl.train(num_iter = num_iter, batch_size=batch_size, lr=lr, w_reg = reg, eval_interval = eval_interval)
        return ('MSWL', value_est)
        
    def run_MSWL(self):
        from rl_nexus.components.ope.MSWL_Kernel_Tf import MSWL
        k_tau = self.spec_tree['MSWL']['k_tau']
        lr = self.spec_tree['MSWL']['lr']
        reg = self.spec_tree['MSWL']['reg']
        num_iter = self.spec_tree['MSWL']['num_iter']
        batch_size = self.spec_tree['MSWL']['batch_size']
        hidden_layers = self.spec_tree['MSWL']['hidden_layers']
        eval_interval = self.spec_tree['MSWL']['eval_interval']
        tail_average = self.spec_tree['MSWL']['tail_average']

        med_dist = est_med_dist(self.data, estimator = 'MSWL') / k_tau

        mswl = MSWL(self.obs_dim, self.act_dim, hidden_layers=hidden_layers,\
            reg_factor = reg, gamma = self.gamma, med_dist = med_dist)
        value_est_list = []
        for iter in range(num_iter):
            batch = sample_data(self.data, batch_size, estimator = 'MSWL')
            debug, loss = mswl.train(batch)
            if iter % eval_interval == 0:
                value_est = mswl.evaluation(self.data['obs'], self.data['acts'], self.data['factor'], self.data['rews'])
                value_est_list.append(value_est)
                if iter % 1000 == 0 and self.debug_mode:
                    print('Iter: {}. True: {:.2f}. MSWL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
        
        mswl.close()

        return ('MSWL', np.mean(value_est_list[-tail_average:]))

    def run_MWL(self):
        from rl_nexus.components.ope.MWL_Kernel_Tf import MWL
        k_tau = self.spec_tree['MWL']['k_tau']
        lr = self.spec_tree['MWL']['lr']
        reg = self.spec_tree['MWL']['reg']
        num_iter = self.spec_tree['MWL']['num_iter']
        batch_size = self.spec_tree['MWL']['batch_size']
        hidden_layers_w = self.spec_tree['MWL']['hidden_layers']
        eval_interval = self.spec_tree['MWL']['eval_interval']
        tail_average = self.spec_tree['MWL']['tail_average']

        
        med_dist = est_med_dist(self.data, estimator = 'MWL') / k_tau

        mwl = MWL(self.obs_dim, self.act_dim, model_weights = self.target_model_weights_list, target_temp=self.target_temp,\
            seed = self.seed, hidden_layers_q = self.hidden_layers_net, activation_q = self.activation_net,\
            hidden_layers_w = hidden_layers_w,lr = lr, med_dist = med_dist, reg_factor = reg, gamma = self.gamma, norm = self.norm_performed)
        
        value_est_list = []
        for iter in range(num_iter):
            batch = sample_data(self.data, batch_size, estimator='MWL')
            debug, loss = mwl.train(batch)
            if iter % eval_interval == 0:
                value_est = mwl.evaluation(self.data['obs'], self.data['acts'], self.data['factor'], self.data['rews'])
                value_est_list.append(value_est)
                if iter % 1000 == 0 and self.debug_mode:
                    print('Iter: {}. True: {:.2f}. MWL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
                    # pdb.set_trace()
        mwl.close()
        
        return ('MWL', np.mean(value_est_list[-tail_average:]))

    def run_MQL_torch(self):
        from rl_nexus.components.ope.MQL_Kernel import MQL
        k_tau = self.spec_tree['MQL']['k_tau']
        lr = self.spec_tree['MQL']['lr']
        reg = self.spec_tree['MQL']['reg']
        num_iter = self.spec_tree['MQL']['num_iter']
        batch_size = self.spec_tree['MQL']['batch_size']
        hidden_layers = self.spec_tree['MQL']['hidden_layers']
        eval_interval = self.spec_tree['MQL']['eval_interval']
        tail_average = self.spec_tree['MQL']['tail_average']

        k_tau = 1
        num_iter = 20000
        reg = 1e-3
        hidden_layers = [64,64]
        lr = 5e-3

        mql = MQL(self.data, self.obs_dim, self.act_dim, k_tau, norm = None, hidden_layers=hidden_layers,\
            seed = self.seed, gamma = self.gamma, debug= self.debug_mode)
        value_est = mql.train(num_iter = num_iter, batch_size=batch_size, lr=lr, q_reg = reg, eval_interval = eval_interval)
        return ('MQL', value_est)

    def run_MQL(self):
        from rl_nexus.components.ope.MQL_Kernel_Tf import MQL
        k_tau = self.spec_tree['MQL']['k_tau']
        lr = self.spec_tree['MQL']['lr']
        reg = self.spec_tree['MQL']['reg']
        num_iter = self.spec_tree['MQL']['num_iter']
        batch_size = self.spec_tree['MQL']['batch_size']
        hidden_layers = self.spec_tree['MQL']['hidden_layers']
        eval_interval = self.spec_tree['MQL']['eval_interval']
        tail_average = self.spec_tree['MQL']['tail_average']

        k_tau = 15
        num_iter = 30000
        lr = 5e-3
        reg = 2e-3
        batch_size = 500
        hidden_layers = [64,64]
        eval_interval = 100

        med_dist = est_med_dist(self.data, estimator = 'MQL') / k_tau

        mql = MQL(self.obs_dim, self.act_dim, model_weights = self.target_model_weights_list, target_temp=self.target_temp,\
            seed = self.seed, hidden_layers_p = self.hidden_layers_net, activation_p = self.activation_net, hidden_layers = hidden_layers,\
            lr = lr, med_dist = med_dist, reg_factor = reg, gamma = self.gamma, norm = self.norm_performed)
        
        value_est_list = []
        for iter in range(num_iter):
            batch = sample_data(self.data, batch_size, estimator='MQL')
            debug, loss = mql.train(batch)
            if iter % eval_interval == 0:
                value_est = mql.evaluation(self.data['init_obs'])
                value_est_list.append(value_est)
                if iter % 1000 == 0 and self.debug_mode:
                    print('Iter: {}. True: {:.2f}. MQL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
        
        mql.close()
        
        return ('MQL', np.mean(value_est_list[-tail_average:]))

    def run_DualDICE(self):
        import rl_nexus.components.ope.DualDICE_Neural_Tf as neural_dual_dice
        data = self.data.copy()
        nu_learning_rate = self.spec_tree['DualDICE']['nu_learning_rate']
        zeta_learning_rate = self.spec_tree['DualDICE']['zeta_learning_rate']
        batch_size = self.spec_tree['DualDICE']['batch_size']
        num_iter = self.spec_tree['DualDICE']['num_iter']
        log_every = self.spec_tree['DualDICE']['log_every']
        tail_average = self.spec_tree['DualDICE']['tail_average']
        hidden_dim = self.spec_tree['DualDICE']['hidden_dim']
        hidden_layers = self.spec_tree['DualDICE']['hidden_layers']
        activation = self.spec_tree['DualDICE']['activation']
        function_exponent = self.spec_tree['DualDICE']['function_exponent']
        

        data['acts'] = np.squeeze(data['acts'])
        data['next_acts'] = np.squeeze(data['next_acts'])
        data['rews'] = np.squeeze(data['rews'])
        data['time_step'] = np.squeeze(data['time_step'])

        # Get solver.
        neural_solver_params = neural_dual_dice.NeuralSolverParameters(
            self.obs_dim,self.act_dim,self.gamma,
            hidden_dim=hidden_layers, hidden_layers=hidden_layers,
            discrete_actions=True, deterministic_env=False,
            nu_learning_rate=nu_learning_rate,
            zeta_learning_rate=zeta_learning_rate,
            batch_size=batch_size,
            num_steps=num_iter,
            log_every=log_every,
            smooth_over=tail_average)
        target_policy_config ={
            'model_weights': self.target_model_weights_list,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'hidden_layers_p': self.hidden_layers_net,
            'activation_p':self.activation_net,
            'target_temp': self.target_temp,
            'seed': self.seed,
        }

        density_estimator = neural_dual_dice.NeuralDualDice(
            parameters=neural_solver_params,
            target_policy_config = target_policy_config,
            solve_for_state_action_ratio=True,
            function_exponent=function_exponent)

        value_est = density_estimator.solve(data, self.norm_performed, logger=None)
        density_estimator.close()
        
        return ('DualDICE', value_est)
    
    def run_PDIS(self):
        # from rl_nexus.utils.ope_utils import plot_histogram_is
        value_est = 0.0
        num_samples = self.data['obs'].shape[0]
        tol = 1e-20
        pi = self.data['target_act_prob'].copy()[:,0]
        mu = self.data['behavior_act_prob'].copy()[:,0]
        pi[pi<tol] = tol; mu[mu<tol] = tol
        reward = self.data['rews'][:,0]
        time_step = self.data['time_step']
        gamma = self.gamma
        num_episodes = self.num_episodes
        importance_weights = np.empty(num_samples)

        for i in range(num_samples):
            if i % self.horizon == 0:
                step_log_pr = 0.0
                est_reward = 0.0
                discounted_t = 1.0
            step_log_pr += np.log(pi[i]) - np.log(mu[i])
            if np.isinf(step_log_pr):
                logger.write_line(" Problem with PDIS ratio being infinite")
                step_log_pr = 1e10 #* replace with something less ridiculous
            est_reward += np.exp(step_log_pr)* reward[i]*discounted_t
            importance_weights[i] = np.exp(step_log_pr)
            discounted_t *= gamma
            if (i+1)% self.horizon == 0: 
                value_est += est_reward
        value_est /= num_episodes
        # plot_histogram_is(importance_weights, self.data['info'][:,0], self.dataset_seed, self.result_path_prefix, name = 'pdis_ratio', save_fig=True)
        # import pdb; pdb.set_trace()
        return ('PDIS', value_est)
    
    def run_WPDIS(self):
        # from rl_nexus.utils.ope_utils import plot_histogram_is
        Log_policy_ratio = []
        REW = []
        tol = 1e-20
        num_samples = self.data['obs'].shape[0]
        pi = self.data['target_act_prob'].copy()[:,0]
        mu = self.data['behavior_act_prob'].copy()[:,0]
        pi[pi<tol] = tol; mu[mu<tol] = tol
        reward = self.data['rews'][:,0]
        time_step = self.data['time_step']
        gamma = self.gamma
        num_episodes = self.num_episodes
        importance_weights = np.empty(num_samples)

        for i in range(num_samples):
            if i % self.horizon == 0:
                # reset for new episode
                log_policy_ratio = []
                rew = []
                discounted_t = 1.0
            log_pr = np.log(pi[i]) - np.log(mu[i])
            if np.isinf(log_pr):
                logger.write_line(" Problem with WPDIS ratio being infinite")
                log_pr = 1e10 #* replace with something less ridiculous
            if log_policy_ratio:
                log_policy_ratio.append(log_pr + log_policy_ratio[-1])
            else:
                log_policy_ratio.append(log_pr)
            rew.append(reward[i]*discounted_t)
            discounted_t *= gamma
            if (i+1) % self.horizon == 0:
                Log_policy_ratio.append(log_policy_ratio)
                REW.append(rew)
        value_est = 0.0
        rho = np.exp(Log_policy_ratio)
        REW = np.array(REW)

        for i in range(REW.shape[0]):
            value_est += np.sum(np.nan_to_num(rho[i]/np.mean(rho, axis = 0)) * REW[i])
            ratio = np.nan_to_num(rho[i]/np.mean(rho, axis = 0))
            ratio[ratio<tol] = tol
            importance_weights[i*REW.shape[1]:(i+1)*REW.shape[1]] = np.log(ratio)
        
        value_est /= REW.shape[0]
        # plot_histogram_is(importance_weights, self.data['info'][:,0], self.dataset_seed, self.result_path_prefix, name = 'wpdis_ratio', save_fig=True)
        # import pdb; pdb.set_trace()
        return ('WPDIS', value_est)

    def initialize_kernel_estimator(self):
        from rl_nexus.components.ope.Kernel_Estimators import Kernel_Estimators
        
        num_random_feature_per_obs_dim = self.spec_tree['Kernel']['num_random_feature_per_obs_dim']
        default_length_scale = self.spec_tree['Kernel']['default_length_scale']
        scale_length_adjustment = self.spec_tree['Kernel']['scale_length_adjustment']
        
        model_reg = self.spec_tree['Kernel']['model_reg']
        reward_reg = self.spec_tree['Kernel']['reward_reg']
        value_reg = self.spec_tree['Kernel']['value_reg']

        self.kernel_estimator = Kernel_Estimators(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon,\
            model_reg = model_reg, reward_reg = reward_reg, value_reg = value_reg,\
                default_length_scale=default_length_scale, random_feature_per_obs_dim=num_random_feature_per_obs_dim,\
                    scale_length_adjustment= scale_length_adjustment, norm = None, policy_net = None)

    # def run_MB(self):
    #     # value_est = self.run_MB_Kernel()
    #     value_est = self.run_MB_Neural()
    #     return ('MB', value_est)

    def run_MB_Neural(self):
        from rl_nexus.components.ope.Model_Based import Model_Based
        hidden_layers = self.spec_tree['MB-N']['hidden_layers']
        activation = self.spec_tree['MB-N']['activation']
        num_iter = self.spec_tree['MB-N']['num_iter']
        reg = self.spec_tree['MB-N']['reg']
        lr = self.spec_tree['MB-N']['lr']
        batch_size = self.spec_tree['MB-N']['batch_size']
        tail_average = self.spec_tree['MB-N']['tail_average']

        mb = Model_Based(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon,\
            hidden_layers=hidden_layers, activation=activation, norm=None, debug=self.debug_mode)
        value_est = mb.train(num_iter=num_iter, lr=lr, batch_size=batch_size, tail_average=tail_average)
        return ('MB-N', value_est)

    def run_MB_Kernel(self):
        from rl_nexus.components.ope.Model_Based_Kernel import Model_Based_Kernel
        num_random_feature_per_obs_dim = self.spec_tree['MB-K']['num_random_feature_per_obs_dim']
        default_length_scale = self.spec_tree['MB-K']['default_length_scale']
        scale_length_adjustment = self.spec_tree['MB-K']['scale_length_adjustment']
        model_reg = self.spec_tree['MB-K']['model_reg']
        reward_reg = self.spec_tree['MB-K']['reward_reg']

        mb = Model_Based_Kernel(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon,\
            model_reg=model_reg, reward_reg=reward_reg, default_length_scale=default_length_scale,\
                random_feature_per_obs_dim=num_random_feature_per_obs_dim, scale_length_adjustment=scale_length_adjustment,\
                    norm=None, policy_net=None)

        value_est = mb.estimate()
        return ('MB-K', value_est)
            
    def run_LSTD(self):
        value_est = self.kernel_estimator.estimate_LSTD()
        return ('LSTD', value_est)

    def run_LSTDQ(self):
        from rl_nexus.components.ope.LSTDQ_Kernel import LSTDQ_Kernel
        num_random_feature_per_obs_dim = self.spec_tree['LSTDQ']['num_random_feature_per_obs_dim']
        default_length_scale = self.spec_tree['LSTDQ']['default_length_scale']
        scale_length_adjustment = self.spec_tree['LSTDQ']['scale_length_adjustment']
        value_reg = self.spec_tree['LSTDQ']['value_reg']
        separate_action_indexing = self.spec_tree['LSTDQ']['separate_action_indexing']

        lstdq = LSTDQ_Kernel(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon,\
            value_reg = value_reg, default_length_scale=default_length_scale, random_feature_per_obs_dim=num_random_feature_per_obs_dim,\
                scale_length_adjustment=scale_length_adjustment, separate_action_indexing = separate_action_indexing, norm=None, policy_net=None)
        
        value_est = lstdq.estimate()
        # value_est_seperate_action = self.kernel_estimator.estimate_LSTDQ_separate_action_indexing()
        return ('LSTDQ', value_est)

    def run_FQE(self):
        from rl_nexus.components.ope.FQE import FQE
        hidden_layers = self.spec_tree['FQE']['hidden_layers']
        activation = self.spec_tree['FQE']['activation']

        num_iter = self.spec_tree['FQE']['num_iter']
        lr = self.spec_tree['FQE']['lr']
        batch_size = self.spec_tree['FQE']['batch_size']
        tail_average = self.spec_tree['FQE']['tail_average']
        use_delayed_target = self.spec_tree['FQE']['use_delayed_target']
        reg = self.spec_tree['FQE']['reg']

        fqe = FQE(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon, policy_net = None,\
            hidden_layers = hidden_layers, activation = activation, norm = None, use_delayed_target= use_delayed_target, debug = self.debug_mode)
        
        value_est = fqe.train(num_iter = num_iter, lr=lr, batch_size = batch_size, tail_average=tail_average, reg = reg)
        return ('FQE', value_est)

    def run_TDREG_Neural(self):
        from rl_nexus.components.ope.TDREG_Neural import TDREG_Neural
        hidden_layers = self.spec_tree['TDREG-N']['hidden_layers']
        activation = self.spec_tree['TDREG-N']['activation']

        num_iter = self.spec_tree['TDREG-N']['num_iter']
        normalize_w = self.spec_tree['TDREG-N']['normalize_w']
        lr = self.spec_tree['TDREG-N']['lr']
        batch_size = self.spec_tree['TDREG-N']['batch_size']
        td_reg = self.spec_tree['TDREG-N']['td_reg']

        tdreg = TDREG_Neural(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon, policy_net = None,\
            seed = self.seed, hidden_layers = hidden_layers, activation = activation, output_transform = 'logexp',\
                norm = None)
        
        value_est = tdreg.train(num_iter = num_iter, lr=lr, td_reg = td_reg, batch_size = batch_size, normalize_w=normalize_w)
        
        return('TDREG-N', value_est)

    def run_TDREG_Kernel(self):
        from rl_nexus.components.ope.TDREG_Kernel import TDREG_Kernel

        value_reg = self.spec_tree['TDREG-K']['value_reg']
        td_ball_epsilon = self.spec_tree['TDREG-K']['td_ball_epsilon']
        w_reg = self.spec_tree['TDREG-K']['w_reg']
        hidden_layers = self.spec_tree['TDREG-K']['hidden_layers']
        activation = self.spec_tree['TDREG-K']['activation']

        input_mode = self.spec_tree['TDREG-K']['input_mode']
        num_iter = self.spec_tree['TDREG-K']['num_iter']
        normalize_w = self.spec_tree['TDREG-K']['normalize_w']
        lr = self.spec_tree['TDREG-K']['lr']
        batch_size = self.spec_tree['TDREG-K']['batch_size']
        use_var_in_loss = self.spec_tree['TDREG-K']['use_var_in_loss']
        tail_average = self.spec_tree['TDREG-K']['tail_average']

        num_random_feature_per_obs_dim = self.spec_tree['TDREG-K']['num_random_feature_per_obs_dim']
        default_length_scale = self.spec_tree['TDREG-K']['default_length_scale']
        scale_length_adjustment = self.spec_tree['TDREG-K']['scale_length_adjustment']

        tdreg = TDREG_Kernel(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon, policy_net = None,\
            value_reg = value_reg, input_mode = input_mode, seed = self.seed, default_length_scale=default_length_scale,\
                hidden_layers = hidden_layers, activation = activation, output_transform = 'logexp',
                random_feature_per_obs_dim=num_random_feature_per_obs_dim, scale_length_adjustment=scale_length_adjustment,norm = None)
        
        value_est = tdreg.train(num_iter = num_iter, batch_size = batch_size, lr = lr, td_ball_epsilon=td_ball_epsilon,\
            w_reg = w_reg, normalize_w = normalize_w, use_var_in_loss = use_var_in_loss, tail_average = tail_average)

        return('TDREG-K', value_est)














