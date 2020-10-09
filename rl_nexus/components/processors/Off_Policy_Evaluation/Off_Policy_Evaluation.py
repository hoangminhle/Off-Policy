import numpy as np
from rl_nexus.utils.ope_utils import set_seed, read_batch_experience, evaluate_on_policy,\
    sample_data, est_med_dist, convert_torch_model_weights_to_list
from rl_nexus.utils.utils import ensure_dir_exists
from rl_nexus.utils.ope_logger import logger
import pdb
import random
import pickle
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from rl_nexus.utils.metric import Metric
# import ray

class Off_Policy_Evaluation():
    def __init__(self, spec_tree, device):
        self.spec_tree = spec_tree
        
        # self.seed = eval('random.'+spec_tree['seed'])
        # self.seed = random.randint(spec_tree['algo_seed'])
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

        # locate off-line dataset
        # dataset_path = spec_tree['load_data_from'] + '/' + self.environment.name+'/behavior_'+str(behavior_policy_min_id)+'_'+\
        #     str(behavior_policy_max_id)+'/n_eps'+str(self.num_episodes)+'/horizon'+str(self.horizon)+'/seed'+str(self.dataset_seed)+'.h5'

        # dataset_path = spec_tree['load_data_from'] + '/behavior_' + str(self.behavior_min_id) + '_' +\
        #     str(self.behavior_max_id)+'/n_eps'+str(self.num_episodes)+'/horizon'+str(self.horizon)+'/seed'+str(self.dataset_seed)+'.h5'

        target_policy_net_path = spec_tree['load_model_from']+ '/policy_' +\
             str(spec_tree['target_policy_id'])+'.pt'       

        #* prepare the data and result path
        self.behavior_min_id = spec_tree['behavior_policy_range']['min']
        self.behavior_max_id = spec_tree['behavior_policy_range']['max']

        # self.dataset_path_prefix = spec_tree['load_data_from'] + '/behavior_' + str(self.behavior_min_id) + '_' +\
        #     str(self.behavior_max_id)+'/n_eps'+str(self.num_episodes)+'/horizon'+str(self.horizon)+'/seed'
        
        self.dataset_path_prefix = spec_tree['load_data_from'] + '/behavior_{}_{}/n_eps{}/horizon{}/seed'.format(self.behavior_min_id,\
            self.behavior_max_id, self.num_episodes, self.horizon)

        # self.result_path_prefix = spec_tree['save_results_to'] + '/behavior_' + str(self.behavior_min_id) + '_' +\
        #     str(self.behavior_max_id)+'/target_'+str(spec_tree['target_policy_id'])+'_temp'+str(self.target_temp)+'/n_eps'+str(self.num_episodes)+'/horizon'+str(self.horizon)+'/seed'
        self.result_path_prefix = spec_tree['save_results_to'] + '/behavior_{}_{}/target_{}_temp{}/n_eps{}/horizon{}/seed'.format(\
            self.behavior_min_id, self.behavior_max_id, spec_tree['target_policy_id'], self.target_temp, self.num_episodes, self.horizon)
        
        ensure_dir_exists(file=self.result_path_prefix)

        # sess = make_session()
        # sess.__enter__()

        # self.tf_policy_net = convert_policy_network(spec_tree,obs_space,action_space, temperature = self.target_temp, path = target_policy_net_path)
        self.target_policy_net = spec_tree.create_component('model', obs_space, action_space)
        self.target_policy_net.temperature = self.target_temp
        self.target_policy_net.load_model(target_policy_net_path)
        # self.target_model_weights_list = extract_model_weights(self.target_policy_net.model, self.obs_dim, self.act_dim)
        self.target_model_weights_list = convert_torch_model_weights_to_list(self.target_policy_net.model)
        # diff = 0
        # pdb.set_trace()
        # for i in range(len(weights_list)):
        #     if len(weights_list[i])>0:
        #         diff += ((weights_list[i][0] - self.target_model_weights_list[i][0])**2).sum()
        #         diff += ((weights_list[i][1] - self.target_model_weights_list[i][1])**2).sum()
        # pdb.set_trace()
        # data = read_batch_experience(dataset_path, self.target_policy_net, self.num_episodes, self.target_temp, self.horizon, self.gamma)
        
        on_policy_num_eps = spec_tree['on_policy_eval_num_episodes']
        self.value_true = evaluate_on_policy(self.environment, self.target_policy_net, num_episodes = on_policy_num_eps, gamma = self.gamma)
        
        # if self.normalization == 'std_norm':
        #     #* whiten data
        #     obs_mean = np.mean(data['obs'], axis=0, keepdims = True)
        #     obs_std = np.std(data['obs'], axis = 0, keepdims = True)
        #     data['obs'] = (data['obs'] - obs_mean) / obs_std
        #     data['next_obs'] = (data['next_obs'] - obs_mean) / obs_std
        #     data['init_obs'] = (data['init_obs'] - obs_mean) / obs_std
        #     data['term_obs'] = (data['term_obs'] - obs_mean) / obs_std

        #     self.norm_performed = {'type': self.normalization, 'shift': obs_mean, 'scale': obs_std}
        # else:
        #     self.norm_performed = {'type': None, 'shift': None, 'scale': None}
        # # data['init_obs'] = data['obs'][::self.horizon]
        # # data['init_acts'] = data['acts'][::self.horizon]
        # # data['term_obs'] = data['next_obs'][self.horizon-1::self.horizon]
        # data['target_prob_init_obs'] = data['target_prob_obs'][::self.horizon]
        # data['target_prob_term_obs'] = data['target_prob_next_obs'][self.horizon-1::self.horizon]

        # data['next_acts'] = data['acts'].copy()
        # for i in range(self.num_episodes):
        #     data['next_acts'][i*self.horizon:(i+1)*self.horizon-1] = data['acts'][i*self.horizon+1:(i+1)*self.horizon].copy()
        # self.data = data
        # import pdb; pdb.set_trace()
        self.hidden_layers_net = self.spec_tree['model']['fcnet_hiddens']
        self.activation_net = self.spec_tree['model']['fcnet_activation']

    # # def prepare_data(self, dataset_seed):
    # def run_estimators(self):
    #     # get the data
    #     data
    #     # run sequential experiments

    #     pass
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
    
    # @ray.remote
    def get_estimate(self, estimator):
        if estimator == 'MWL':
            value_est = self.run_MWL()
        elif estimator == 'MSWL':
            value_est = self.run_MSWL()
        elif estimator == 'MQL':
            value_est = self.run_MQL()
        elif estimator == 'DualDICE':
            value_est = self.run_DualDICE()
        elif estimator == 'TDREG':
            value_est = self.run_TDREG()
        elif estimator == 'MB':
            value_est = self.run_MB()
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

    def run_estimators(self, dataset_seed):
        self.data = self.prepare_data(dataset_seed)
        result = {}
        result['On_Policy'] = self.value_true
        
        #* Do we run kernel-based methods?
        if 'MB' in self.ope or 'LSTD' in self.ope or 'LSTDQ' in self.ope:
            self.initialize_kernel_estimator()
        
        if self.spec_tree['use_ray']:
            ray.init()
            result_ids = []
            for estimator in self.ope:
                # value_est = self.get_estimate(estimator, data)
                # value_est = Off_Policy_Evaluation.get_estimate.remote(self, estimator, data)
                result_ids.append(Off_Policy_Evaluation.get_estimate.remote(self, estimator))
                # result[estimator] = value_est
                # logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(dataset_seed, \
                #     self.value_true, estimator, value_est))
            
            unordered_result = ray.get(result_ids)
            ray.shutdown()

            for item in unordered_result:
                estimator = item[0]
                value_est = item[1]
                result[estimator] = value_est
                logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(dataset_seed, \
                    self.value_true, estimator, value_est))
        else:
            for estimator in self.ope:
                result_tuple = self.get_estimate(estimator)
                value_est = result_tuple[1] #* the value est occupies the 2nd position
                result[estimator] = value_est
                logger.write_line('Dataset Seed: {}, True Value: {:.2f}, {} Estimator: {:.2f}'.format(dataset_seed, \
                    self.value_true, estimator, value_est))

        error_metrics = {}
        for estimator in self.ope:
            error_metrics[estimator] = abs((result[estimator] - self.value_true)/ self.value_true)
        logger.write_ope_metrics(dataset_seed, error_metrics, result)

        return result

    def execute(self):
        set_seed(self.seed)

        dataset_seed = self.dataset_seed
        result = self.run_estimators(dataset_seed)
        #* save the result
        result_path = self.result_path_prefix + '{}.pickle'.format(dataset_seed)
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        # logger.write_and_condense_metrics()

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
                # if iter % 1000 == 0:
                #     print('Iter: {}. True: {:.2f}. MSWL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
        
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
                # if iter % 1000 == 0:
                #     print('Iter: {}. True: {:.2f}. MWL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
        
        mwl.close()
        
        return ('MWL', np.mean(value_est_list[-tail_average:]))

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
                # if iter % 1000 == 0:
                #     print('Iter: {}. True: {:.2f}. MQL Estimate: {:.2f}'.format(iter,self.value_true, np.mean(value_est_list[-tail_average:])))
        
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
            discounted_t *= gamma
            if (i+1)% self.horizon == 0: 
                value_est += est_reward
        value_est /= num_episodes

        return ('PDIS', value_est)
    
    def run_WPDIS(self):
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
        
        value_est /= REW.shape[0]
        
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


    def run_MB(self):
        value_est = self.kernel_estimator.estimate_model_based()
        return ('MB', value_est)
            
    def run_LSTD(self):
        value_est = self.kernel_estimator.estimate_LSTD()
        return ('LSTD', value_est)

    def run_LSTDQ(self):
        value_est = self.kernel_estimator.estimate_LSTDQ()
        return ('LSTDQ', value_est)

    def run_TDREG(self):
        from rl_nexus.components.ope.TDREG_Kernel import TDREG_Kernel

        value_reg = self.spec_tree['TDREG']['value_reg']
        td_ball_epsilon = self.spec_tree['TDREG']['td_ball_epsilon']
        w_reg = self.spec_tree['TDREG']['w_reg']
        hidden_layers = self.spec_tree['TDREG']['hidden_layers']
        activation = self.spec_tree['TDREG']['activation']

        input_mode = self.spec_tree['TDREG']['input_mode']
        num_iter = self.spec_tree['TDREG']['num_iter']
        normalize_w = self.spec_tree['TDREG']['normalize_w']
        lr = self.spec_tree['TDREG']['lr']
        batch_size = self.spec_tree['TDREG']['batch_size']
        use_var_in_loss = self.spec_tree['TDREG']['use_var_in_loss']
        tail_average = self.spec_tree['TDREG']['tail_average']

        num_random_feature_per_obs_dim = self.spec_tree['Kernel']['num_random_feature_per_obs_dim']
        default_length_scale = self.spec_tree['Kernel']['default_length_scale']
        scale_length_adjustment = self.spec_tree['Kernel']['scale_length_adjustment']

        tdreg = TDREG_Kernel(self.data, self.obs_dim, self.act_dim, self.gamma, self.horizon, policy_net = None,\
            value_reg = value_reg, input_mode = input_mode, seed = self.seed, default_length_scale=default_length_scale,\
                hidden_layers = hidden_layers, activation = activation, output_transform = 'logexp',
                random_feature_per_obs_dim=num_random_feature_per_obs_dim, scale_length_adjustment=scale_length_adjustment,norm = None)
        
        value_est = tdreg.train(num_iter = num_iter, batch_size = batch_size, lr = lr, td_ball_epsilon=td_ball_epsilon,\
            w_reg = w_reg, normalize_w = normalize_w, use_var_in_loss = use_var_in_loss, tail_average = tail_average)

        return('TDREG', value_est)














