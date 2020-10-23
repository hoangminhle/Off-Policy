import numpy as np
import torch
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from pytorch2keras import pytorch_to_keras
import pdb
import h5py
from rl_nexus.utils.data_size_check import total_size
import torch.nn.functional as F
import json
import pickle
import math

### OPE additions
def set_seed(seed):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# data = read_batch_experience(dataset_path, tf_policy_net, self.num_episodes, self.target_temp, self.horizon, self.gamma)
def read_batch_experience(dataset_path, target_net, num_episodes, target_temp, horizon, gamma):
    with open(dataset_path,'rb') as f:
        data = pickle.load(f)
    num_samples = data['obs'].shape[0]
    data['target_act_prob'] = target_net.get_prob_with_act(data['obs'], data['acts'])
    data['ratio'] = data['target_act_prob'] / data['behavior_act_prob']
    data['factor'] = gamma**data['time_step']
    data['target_prob_obs'] = target_net.get_probabilities(data['obs'])
    data['target_prob_next_obs'] = target_net.get_probabilities(data['next_obs'])
    data['target_prob_init_obs'] = data['target_prob_obs'][::horizon]
    data['target_prob_term_obs'] = data['target_prob_next_obs'][horizon-1::horizon]
    return data

def plot_histogram_is(ratio, info, dataset_seed, result_path, name, save_fig= False):
    if save_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        #* plot ratio histogram
        n_bins= 10
        bin_list = [0]
        mark = 1.0/2
        while mark < ratio.max():
            bin_list.append(mark)
            mark = 2*mark
        # bin_list = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, ratio.max()]
        
        non_terminal_ratio = ratio[info == False]
        fig, axs = plt.subplots(2)
        # axs[0].hist(non_terminal_ratio, weights = np.ones(len(non_terminal_ratio))/len(non_terminal_ratio), bins = n_bins)
        axs[0].hist(non_terminal_ratio, weights = np.ones(len(non_terminal_ratio))/len(non_terminal_ratio), bins = n_bins)
        axs[0].set_title('Log scale - Trajectory-level importance ratio over non-terminal steps', fontsize = 9)
        # axs[1].hist(ratio, weights = np.ones(len(ratio))/len(ratio), bins = n_bins)
        axs[1].hist(ratio, weights = np.ones(len(ratio))/len(ratio), bins = n_bins)
        axs[1].set_title('Log scale - Trajectory-level importance ratio over all steps', fontsize = 9)
        axs[0].yaxis.set_major_formatter(PercentFormatter(1))
        axs[1].yaxis.set_major_formatter(PercentFormatter(1))
        # plt.savefig(result_path+'{}_pdis_ratio_dist.png'.format(dataset_seed))
        # plt.savefig('test/'+'{}_pdis_ratio_dist.png'.format(dataset_seed))
        plt.savefig('test/'+'{}_dist.png'.format(name))
        pdb.set_trace()
def summarize_data(data, result_path, save_fig = False):
    if save_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        #* plot ratio histogram
        n_bins = 10
        non_terminal_data = data['ratio'][data['info']==False]
        plt.hist(non_terminal_data, weights = np.ones(len(non_terminal_data))/len(non_terminal_data), bins = n_bins)
        plt.title('Density ratio histogram over non-Terminal States', fontsize = 15)
        # pdb.set_trace()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(result_path+'{}_ratio_dist.png'.format(data['metadata']['dataset_seed']))

def choose_estimate_from_sequence(value_est_list):
        low = math.floor(min(value_est_list))
        high = math.ceil(max(value_est_list))
        bins = range(low, high+1)
        hist = np.histogram(np.array(value_est_list), bins)
        largest_bin = hist[0].argmax()
        lower = hist[1][largest_bin]
        upper = hist[1][largest_bin+1]
        #* get the entries in between lower and upper
        selected_values = [val for val in value_est_list if (lower <= val) and (val <= upper)]
        assert len(selected_values) == hist[0][largest_bin]
        return np.mean(selected_values)

# def read_batch_experience(dataset_path, target_net, num_episodes, target_temp, horizon, gamma):
#     data = {}
#     hf = h5py.File(dataset_path,'r')
#     n_samples = hf['obs'].shape[0]
#     data['obs'] = hf.get('obs')[:]
#     data['next_obs'] = hf.get('next_obs')[:]
#     data['acts'] = hf.get('acts')[:].reshape(n_samples,-1)
#     data['next_acts'] = hf.get('next_acts')[:].reshape(n_samples,-1)
#     data['rews'] = hf.get('rews')[:].reshape(n_samples, -1)
#     data['done'] = hf.get('done')[:].reshape(n_samples, -1)
#     # data['info'] = hf.get('info')[:].reshape(n_samples, -1)
#     data['behavior_act_prob'] = hf.get('behavior_act_prob')[:].reshape(n_samples,-1)
#     data['target_act_prob'] = target_net.get_prob_with_act(data['obs'], data['acts'])
#     data['ratio'] = data['target_act_prob'] / data['behavior_act_prob']
#     data['time_step'] = hf.get('time_step')[:].reshape(n_samples, -1)
#     data['factor'] = gamma**data['time_step']
#     # data['init_acts'] = data['acts'][::ep_len]
#     # data['init_obs'] = data['obs'][::ep_len]
#     data['target_prob_obs'] = target_net.get_probabilities(data['obs'])
#     data['target_prob_next_obs'] = target_net.get_probabilities(data['next_obs'])
#     data['init_obs'] = hf.get('init_obs')[:].reshape(num_episodes,-1)
#     data['term_obs'] = hf.get('term_obs')[:].reshape(num_episodes,-1)
#     data['init_acts'] = hf.get('init_acts')[:].reshape(num_episodes,-1)
    
#     if total_size(data) / 1024**2 > 10:
#         import warnings
#         warnings.warn("The data size is rather large")
#     # print(total_size(data, verbose= True))
#     return data

# value_true = evaluate_on_policy(self.environment, tf_policy_net, num_episodes = 20, gamma = self.gamma)

def evaluate_on_policy(env, policy_net, num_episodes=200, gamma = 0.99):
    accum_rew = 0.0
    rew_list = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        factor = 1.0
        while not done:
            act = policy_net.sample_action([obs])
            obs, rew, done, _ = env.step(act)
            rew *= factor
            factor *= gamma
            accum_rew += rew
            # rew_list.append(rew)
        # if i % 10 == 0:
        #     print('Traj ', i, ' Current Average ', accum_rew / (i + 1))
    print('\nOn policy estimate of given policy with {} trajectories: {:.2f}\n'.format(num_episodes, accum_rew / num_episodes))
    return accum_rew / num_episodes

def sample_data(dataset, sample_num, estimator):
    data_size = dataset['obs'].shape[0]
    init_size = dataset['init_obs'].shape[0]
    
    index_1 = np.random.choice(data_size, sample_num)
    index_2 = np.random.choice(data_size, sample_num + 10)

    init_index_1 = np.random.choice(init_size, sample_num)
    init_index_2 = np.random.choice(init_size, sample_num + 10)
    if estimator == 'MWL':
        return {
            'obs_1': dataset['obs'][index_1],
            'obs_2': dataset['obs'][index_2],
            'next_obs_1': dataset['next_obs'][index_1],
            'next_obs_2': dataset['next_obs'][index_2],
            'next_acts_1': dataset['next_acts'][index_1],
            'next_acts_2': dataset['next_acts'][index_2],
            'acts_1': dataset['acts'][index_1],
            'acts_2': dataset['acts'][index_2],
            'init_obs_1': dataset['init_obs'][init_index_1],
            'init_obs_2': dataset['init_obs'][init_index_2],
            'init_acts_1': dataset['init_acts'][init_index_1],
            'init_acts_2': dataset['init_acts'][init_index_2],
            'factor_1': dataset['factor'][index_1],
            'factor_2': dataset['factor'][index_2],
            'rew': dataset['rews'][index_1],
            'done': dataset['done'][index_1],
        }
    elif estimator == 'MSWL':
        ratio_1 = dataset['ratio'][index_1]
        ratio_2 = dataset['ratio'][index_2]
        return {
            'obs_1': dataset['obs'][index_1],
            'obs_2': dataset['obs'][index_2],
            'next_obs_1': dataset['next_obs'][index_1],
            'next_obs_2': dataset['next_obs'][index_2],
            'ratio_1': ratio_1,
            'ratio_2': ratio_2,
            'init_obs_1': dataset['init_obs'][init_index_1],
            'init_obs_2': dataset['init_obs'][init_index_2],
            'factor_1': dataset['factor'][index_1],
            'factor_2': dataset['factor'][index_2],
            'rew': dataset['rews'][index_1],
            'done': dataset['done'][index_1],
        }
    elif estimator == 'MQL':
        return {
            'obs_1': dataset['obs'][index_1],
            'obs_2': dataset['obs'][index_2],
            'next_obs_1': dataset['next_obs'][index_1],
            'next_obs_2': dataset['next_obs'][index_2],
            'act_1': dataset['acts'][index_1],
            'act_2': dataset['acts'][index_2],
            'rew_1': dataset['rews'][index_1],
            'rew_2': dataset['rews'][index_2],
        }
    else:
        raise NotImplementedError


def est_med_dist(dataset, estimator):
    data = sample_data(dataset, 5000, estimator)
    if estimator == 'MWL':
        obs_1 = data['obs_1']
        obs_2 = data['obs_2']
        act_1 = data['acts_1']
        act_2 = data['acts_2']

        obs_act_1 = np.concatenate([obs_1, act_1], axis=1)
        obs_act_2 = np.concatenate([obs_2, act_2], axis=1)
        
        med = np.median(np.sqrt(np.sum(np.square(obs_act_1[None, :, :] - obs_act_2[:, None, :]), axis = -1)))

        return np.array([med] * 36)
    
    elif estimator == 'MSWL':
        obs_1 = data['obs_1']
        obs_2 = data['obs_2']

        m0 = np.median(np.sqrt(np.sum(np.square(obs_1[None, :, :] - obs_2[:, None, :]), axis = -1)))
        
        return np.array([m0] * 4)
    elif estimator == 'MQL':
        obs_1 = data['obs_1']
        obs_2 = data['obs_2']
        act_1 = data['act_1']
        act_2 = data['act_2']
        
        obs_act_1 = np.concatenate([obs_1, act_1], axis=1)
        obs_act_2 = np.concatenate([obs_2, act_2], axis=1)
        
        return np.median(np.sqrt(np.sum(np.square(obs_act_1[None, :, :] - obs_act_2[:, None, :]), axis = -1)))
    else:
        raise NotImplementedError

def convert_torch_model_weights_to_list(torch_model):
    state_dict = torch_model.state_dict()
    keys = list(state_dict.copy().keys())
    assert len(keys) % 2 == 0 # make sure weights and biases come in pair
    weights_list = []
    #* Here's the assumption we will make: the weights and biases come from odd indexed layers
    #* meaning we view the simple model as input, layer, activation, layer, activation, etc, layer, output
    # import pdb; pdb.set_trace()
    for i in range(len(keys)):
        if i % 2 == 0:
            weights_list.append([])
        else:
            #* Transpose the weights to accommodate Keras and tf
            weight = state_dict[keys[i-1]].numpy().T
            bias = state_dict[keys[i]].numpy()
            weights_list.append([weight, bias])
    return weights_list


def extract_model_weights(torch_model, obs_dim, act_dim):
    # assert path is not None
    # obs_dim = obs_space.shape[0]
    # act_dim = action_space.n
    # policy_net_torch = spec_tree.create_component('model', obs_space, action_space)
    # policy_net_torch.load_model(path)
    print('Number of torch model parameters: ', sum(p.numel() for p in torch_model.parameters()))

    dummy_state = np.random.rand(obs_dim)
    dummy_input = torch.from_numpy(dummy_state.reshape(1, -1)).float()
    keras_model = pytorch_to_keras(torch_model, dummy_input)

    weights_list = []
    for layer in keras_model.layers:
        weights_list.append(layer.get_weights())
    return weights_list

class Q_Model_Tf():
    def __init__(self, obs_dim, act_dim, seed, hidden_layers = [64,64], activation = 'tanh', temperature = 1.0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = 'converted_model/policy_net',
        self.seed = seed
        self.temperature = temperature
        if activation == 'tanh':
            activation_fn = tf.keras.activations.tanh
        elif activation == 'relu':
            activation_fn = tf.keras.activations.relu
        else: raise NotImplementedError

        inputs = tf.keras.Input(shape = (self.obs_dim,))
        x = tf.keras.layers.Dense(hidden_layers[0])(inputs)
        x = tf.keras.layers.Activation(activation_fn)(x)
        for hidden in hidden_layers[1:]:
            x = tf.keras.layers.Dense(hidden)(x)
            x = tf.keras.layers.Activation(activation_fn)(x)
        outputs = tf.keras.layers.Dense(self.act_dim)(x)
        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)

        dummy_state = tf.random.normal(shape = (3,self.obs_dim))
        dummy_output = self.model(dummy_state)
        
        # for l in self.model.layers:
        #     l.trainable = False
        self.tau_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])

    def build_random_policy(self, obs_ph, reuse = True):
        logit_value = self.model(obs_ph)        
        logits = logit_value / self.tau_ph
        random_action = tf.random.categorical(logits, 1, seed=self.seed)
        return random_action

    def build_prob(self, obs_ph, reuse=True, split=True):
        assert reuse is True
        logit_value = self.model(obs_ph)
        logits = logit_value / self.tau_ph
        prob = tf.stop_gradient(tf.nn.softmax(logits, axis=1))
        if split:
            # return tf.split(prob, 2, axis=1)
            return tf.split(prob, self.act_dim, axis=1)
        else:
            return prob
    
    # load model weights from external list of weights
    def load_weight(self, model_weights):
        # pdb.set_trace()
        assert len(self.model.layers) == len(model_weights), 'the number of layers do not match, check source model'
        for idx, layer in enumerate(self.model.layers):
            self.model.layers[idx].set_weights(model_weights[idx])
        for l in self.model.layers:
            l.trainable = False
        # build action selection model
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        q_value = self.model(inputs)
        logits = q_value / self.temperature
        sampled_action = tf.random.categorical(logits, 1, seed=self.seed)
        self.action_selection_model = tf.keras.Model(inputs = inputs, outputs = sampled_action)
        # self.model.summary()

    def sample_action(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        # q_value = self.model.predict(org_obs)
        # logits = q_value / self.temperature
        # # sampled_action = tf.compat.v1.squeeze(tf.multinomial(logits, 1, seed=self.seed))
        # sampled_action = tf.random.categorical(logits,1)
        # import pdb; pdb.set_trace()
        return np.squeeze(self.action_selection_model.predict(np.array(org_obs)))
        
        # return sampled_action
        # return np.squeeze(self.action_selection_model.predict(np.array(org_obs)))

    # def get_prob_with_act(self, obs, act, tau):
    #     q_values = self.model(obs)
    #     logits = q_values / tau
    #     # action_probabilities = F.softmax(torch.tensor(logits), dim=1)
    #     action_probabilities = tf.nn.softmax(logits, dim=1)
    #     one_hot_action = torch.squeeze(F.one_hot(torch.tensor(act), num_classes = self.act_dim))
    #     act_prob_given = torch.sum(action_probabilities*one_hot_action, dim=1).numpy()[:,np.newaxis]
    #     return act_prob_given        

    def get_probabilities(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        q_values = self.model.predict(org_obs)
        
        logits = q_values / self.temperature
        action_probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        # action_probabilities = tf.nn.softmax(logits, axis=1)
        # import pdb; pdb.set_trace()
        return action_probabilities

class convert_policy_network():
    def __init__(self, spec_tree, obs_space, action_space, temperature=1.0, path=None):
        assert path is not None
        self.obs_dim = obs_space.shape[0]
        self.act_dim = action_space.n
        self.temperature = temperature
        self.seed = spec_tree['seed']
        policy_net_torch = spec_tree.create_component('model', obs_space, action_space)
        policy_net_torch.load_model(path)
        print('Number of torch model parameters: ', sum(p.numel() for p in policy_net_torch.model.parameters()))
        
        dummy_state = np.random.rand(self.obs_dim)
        dummy_input = torch.from_numpy(dummy_state.reshape(1, -1)).float()
        self.k_model = pytorch_to_keras(policy_net_torch.model, dummy_input)

        # build tf-keras model
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        logit_value = self.k_model(inputs)
        logits = logit_value / self.temperature
        sampled_action = tf.random.categorical(logits, 1, seed=self.seed)
        self.action_selection_model = tf.keras.Model(inputs = inputs, outputs = sampled_action)

        for l in self.k_model.layers:
            l.trainable = False
        for l in self.action_selection_model.layers:
            l.trainable = False

        self.action_selection_model.summary()
        self.tau_ph = self.temperature
        self.tau_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])

    def build_random_policy(self, obs_ph, reuse = True):
        # pdb.set_trace()
        logit_value = self.k_model(obs_ph)        
        logits = logit_value / self.tau_ph
        random_action = tf.random.categorical(logits, 1, seed=self.seed)

        return random_action

    def build_prob(self, obs_ph, reuse=True, split=True):
        assert reuse is True
        logit_value = self.k_model(obs_ph)
        logits = logit_value / self.tau_ph
        prob = tf.stop_gradient(tf.nn.softmax(logits, axis=1))
        if split:
            return tf.split(prob, 2, axis=1)
        else:
            return prob
    def sample_action(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        return np.squeeze(self.action_selection_model.predict(np.array(org_obs)))

    def get_prob_with_act(self, obs, act, tau):
        logit_values = self.k_model.predict(obs)
        logits = logit_values / tau
        action_probabilities = F.softmax(torch.tensor(logits), dim=1)
        one_hot_action = torch.squeeze(F.one_hot(torch.tensor(act), num_classes = self.act_dim))
        act_prob_given = torch.sum(action_probabilities*one_hot_action, dim=1).numpy()[:,np.newaxis]
        return act_prob_given        

    def get_probabilities(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        logit_values = self.k_model.predict(org_obs)
        logits = logit_values / self.temperature
        action_probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        return action_probabilities



