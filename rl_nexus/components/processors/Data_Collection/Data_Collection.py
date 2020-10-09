from rl_nexus.utils.utils import resolve_path, ensure_dir_exists
import yaml
import h5py
import numpy as np
import pickle
import random
# import ray
from copy import deepcopy
import torch

class Data_Collection():
    def __init__(self, spec_tree, device):
        self.spec_tree = spec_tree
        self.device = device

        # get spec values
        self.enabled = spec_tree['enabled']
        behavior_policy_range = spec_tree['behavior_policy_range']
        self.behavior_policy_range = range(behavior_policy_range['min'], behavior_policy_range['max']+1, behavior_policy_range['step'])
        self.temperature = spec_tree['temperature']
        self.num_episodes = spec_tree['num_episodes']
        self.save_data_to = resolve_path(spec_tree['save_data_to'])
        self.dataset_seed = spec_tree['dataset_seed']
        # self.dataset_seed = range(dataset_seed['start'], dataset_seed['stop'], dataset_seed['step'])

        self.behavior_policy_collection = []
        dummy_env = self.spec_tree.create_component('environment')
        self.obs_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        for behavior_policy_id in self.behavior_policy_range:
            behavior_policy = self.spec_tree.create_component('model', self.obs_space, self.action_space)
            model_path = self.spec_tree['load_model_from']+'policy_'+str(behavior_policy_id)+'.pt'
            behavior_policy.load_model(model_path)
            self.behavior_policy_collection.append(behavior_policy)
        
        data_path = spec_tree['save_data_to']+'/behavior_'+str(min(self.behavior_policy_range))+\
            '_'+str(max(self.behavior_policy_range))+'/n_eps'+str(self.num_episodes)+'/horizon'+str(dummy_env.max_ep_len)+'/'
        ensure_dir_exists(file=data_path)

    # @ray.remote    
    def generate_data(self,dataset_seed):
        spec_tree = deepcopy(self.spec_tree)

        spec_tree['environment']['seed'] = dataset_seed
        environment = spec_tree.create_component('environment')

        torch.manual_seed(dataset_seed)

        obs_space = environment.observation_space
        action_space = environment.action_space
        max_ep_len = environment.max_ep_len
        assert environment.fixed_length_episode, 'episode length should be fixed to ensure even-sized trajectories'
        
        #* generating and saving data to hdf5
        data_file_name = spec_tree['save_data_to']+'/behavior_'+str(min(self.behavior_policy_range))+\
            '_'+str(max(self.behavior_policy_range))+'/n_eps'+str(self.num_episodes)+'/horizon'+str(max_ep_len)+\
                '/seed'+str(dataset_seed)+'.h5'
        # ensure_dir_exists(file=data_file_name)
        env_metadata = {}; env_metadata['obs_space'] = environment.observation_space; env_metadata['action_space'] = environment.action_space
        metadata = np.void(pickle.dumps(env_metadata))
        data_file = h5py.File(data_file_name, mode = 'w')
        
        data_file.create_dataset("env_metadata", data = metadata)
        max_size = self.num_episodes * max_ep_len
        data_file.create_dataset('obs', (max_size,)+obs_space.shape, obs_space.dtype, maxshape = (None,)+obs_space.shape, compression='lzf')
        data_file.create_dataset('acts', (max_size,)+action_space.shape, action_space.dtype, maxshape = (None,)+action_space.shape)
        data_file.create_dataset('time_step', (max_size,), dtype='i8')
        data_file.create_dataset('behavior_act_prob', (max_size,), dtype = 'f')
        data_file.create_dataset('rews', (max_size,), dtype = 'f')
        data_file.create_dataset('done', (max_size,), dtype = '?')
        data_file.create_dataset('info', (max_size,), dtype = '?')
        data_file.create_dataset('next_obs', (max_size,)+obs_space.shape, obs_space.dtype, maxshape = (None,)+obs_space.shape, compression='lzf')
        data_file.create_dataset('eps_id', (max_size,), dtype='i4')
        data_file.create_dataset('policy_id', (max_size,), dtype='i4')
        row_idx = 0
        data_file.create_dataset('init_obs', (self.num_episodes,)+obs_space.shape, dtype=obs_space.dtype)
        data_file.create_dataset('term_obs', (self.num_episodes,)+obs_space.shape, dtype=obs_space.dtype)
        data_file.create_dataset('next_acts',(max_size,)+action_space.shape, action_space.dtype, maxshape = (None,)+action_space.shape)
        data_file.create_dataset('init_acts', (self.num_episodes,)+action_space.shape, action_space.dtype, maxshape = (None,)+action_space.shape)

        for eps_id in range(self.num_episodes):
            idx = random.choice(range(len(self.behavior_policy_range)))
            policy = self.behavior_policy_collection[idx]
            obs = environment.reset()
            done = False
            t = 0
            data_file['init_obs'][eps_id] = obs
            is_init_step = True
            while True:
                action = policy.sample_action(obs)
                action, action_prob = policy.sample_action_with_prob(obs)
                next_obs, rew, done, info = environment.step(action)
                if is_init_step:
                    is_init_step = False
                    data_file['init_acts'][eps_id] = action
                else:
                    data_file['next_acts'][row_idx-1] = action
                # write data
                data_file['obs'][row_idx] = obs
                data_file['next_obs'][row_idx] = next_obs
                data_file['acts'][row_idx] = action
                data_file['time_step'][row_idx] = t
                data_file['rews'][row_idx] = rew
                data_file['behavior_act_prob'][row_idx] = action_prob
                data_file['done'][row_idx] = done
                data_file['info'][row_idx] = info['terminal']
                data_file['eps_id'][row_idx] = eps_id
                data_file['policy_id'][row_idx] = self.behavior_policy_range[idx]
                
                row_idx += 1
                t += 1
                if done:
                    action = policy.sample_action(next_obs)
                    data_file['next_acts'][row_idx-1] = action
                    break
                obs = next_obs
            data_file['term_obs'][eps_id] = next_obs
            # print('Done with episode {}'.format(eps_id+1))
            # import pdb; pdb.set_trace()
        data_file.close()
        print('Done with dataseed', dataset_seed)
        return None


    def execute(self):
        print("running data collection")
        # ray.init()
        # dataset_seed = self.dataset_seed[0]
        # collector = Collector.remote(self.spec_tree, self.behavior_policy_collection, dataset_seed)
        # collector.generate_data.remote()
        # ray.shutdown()
        # ray.init()
        self.generate_data(self.dataset_seed)
        # result_ids = []
        # dataset_seed = self.dataset_seed
        # for dataset_seed in self.dataset_seed:
        #     result_ids.append(Data_Collection.generate_data.remote(self,dataset_seed))
        # results = ray.get(result_ids)
        # #     # self.generate_data.remote(dataset_seed)
        # ray.shutdown()
        # behavior_policy_id = self.behavior_policy_range[0]
        
        
        # self.spec_tree['environment']['seed'] = dataset_seed
        # environment = self.spec_tree.create_component('environment')
        # obs_space = environment.observation_space
        # action_space = environment.action_space
        # max_ep_len = environment.max_ep_len
        # assert environment.fixed_length_episode, 'episode length should be fixed to ensure even-sized trajectories'

        # behavior_policy_collection = []
        # for behavior_policy_id in self.behavior_policy_range:
        #     behavior_policy = self.spec_tree.create_component('model', obs_space, action_space)
        #     model_path = self.spec_tree['model_path']+'/policy_'+str(behavior_policy_id)+'.pt'
        #     behavior_policy.load_model(model_path)
        #     behavior_policy_collection.append(behavior_policy)

        # #* generating and saving data to hdf5
        # data_file_name = self.spec_tree['save_data_to']+'/behavior_'+str(min(self.behavior_policy_range))+\
        #     '_'+str(max(self.behavior_policy_range))+'/n_eps'+str(self.num_episodes)+'/horizon'+str(max_ep_len)+\
        #         '/seed'+str(dataset_seed)+'.h5'
        # ensure_dir_exists(file=data_file_name)
        # env_metadata = {}; env_metadata['obs_space'] = environment.observation_space; env_metadata['action_space'] = environment.action_space
        # metadata = np.void(pickle.dumps(env_metadata))
        # data_file = h5py.File(data_file_name, mode = 'w')
        
        # data_file.create_dataset("env_metadata", data = metadata)
        # max_size = self.num_episodes * max_ep_len
        # data_file.create_dataset('obs', (max_size,)+obs_space.shape, obs_space.dtype, maxshape = (None,)+obs_space.shape, compression='lzf')
        # data_file.create_dataset('acts', (max_size,)+action_space.shape, action_space.dtype, maxshape = (None,)+action_space.shape)
        # data_file.create_dataset('time_step', (max_size,), dtype='i8')
        # data_file.create_dataset('behavior_act_prob', (max_size,), dtype = 'f')
        # data_file.create_dataset('rews', (max_size,), dtype = 'f')
        # data_file.create_dataset('done', (max_size,), dtype = '?')
        # data_file.create_dataset('info', (max_size,), dtype = '?')
        # data_file.create_dataset('next_obs', (max_size,)+obs_space.shape, obs_space.dtype, maxshape = (None,)+obs_space.shape, compression='lzf')
        # data_file.create_dataset('eps_id', (max_size,), dtype='i4')
        # data_file.create_dataset('policy_id', (max_size,), dtype='i4')
        # row_idx = 0
        # for eps_id in range(self.num_episodes):
        #     idx = random.choice(range(len(self.behavior_policy_range)))
        #     policy = behavior_policy_collection[idx]
        #     obs = environment.reset()
        #     # obs = np.vstack((environment.reset(), environment.reset()))
        #     # import pdb; pdb.set_trace()
        #     done = False
        #     t = 0
        #     while not done:
        #         action = policy.sample_action(obs)
        #         action, action_prob = policy.sample_action_with_prob(obs)
        #         next_obs, rew, done, info = environment.step(action)
        #         # write data
        #         data_file['obs'][row_idx] = obs
        #         data_file['next_obs'][row_idx] = next_obs
        #         data_file['acts'][row_idx] = action
        #         data_file['time_step'][row_idx] = t
        #         data_file['rews'][row_idx] = rew
        #         data_file['behavior_act_prob'][row_idx] = action_prob
        #         data_file['done'][row_idx] = done
        #         # import pdb; pdb.set_trace()
        #         data_file['info'][row_idx] = info['terminal']
        #         data_file['eps_id'][row_idx] = eps_id
        #         data_file['policy_id'][row_idx] = self.behavior_policy_range[idx]
                
        #         row_idx += 1
        #         obs = next_obs
        #         t += 1
        #     print('Done with episode {}'.format(eps_id+1))
        #     # import pdb; pdb.set_trace()
        # data_file.close()
        # print('Done with dataseed', dataset_seed)
        # import pdb; pdb.set_trace()