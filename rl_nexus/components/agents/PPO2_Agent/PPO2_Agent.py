import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from rl_nexus.components.agents.common.agent import *

from ray.rllib.utils.annotations import override

class PPO2_Agent(Agent):
    def __init__(self, spec_tree, obs_space, act_space, device):
        super().__init__(spec_tree, obs_space, act_space, device)

        # Get spec values.
        self.local_mode = spec_tree.get_value('local_mode')

        # Values to add to rllib_config
        rllib_config_value_names = ('num_workers', 
                                    'num_envs_per_worker', 
                                    'batch_mode',
                                    'num_gpus',
                                    'optimizer',
                                    'num_cpus_per_worker',
                                    'num_gpus_per_worker',
                                    'gamma',
                                    'horizon',
                                    'soft_horizon',
                                    'no_done_at_end',
                                    'normalize_actions',
                                    'clip_rewards',
                                    'clip_actions',
                                    'preprocessor_pref',
                                    'use_critic',
                                    'use_gae',
                                    'lambda',
                                    'kl_coeff',
                                    'rollout_fragment_length',
                                    'train_batch_size',
                                    'sgd_minibatch_size',
                                    'shuffle_sequences',
                                    'num_sgd_iter',
                                    'lr',
                                    'lr_schedule',
                                    'vf_loss_coeff',
                                    'entropy_coeff',
                                    'entropy_coeff_schedule',
                                    'clip_param',
                                    'vf_clip_param',
                                    'grad_clip',
                                    'kl_target',
                                    'observation_filter',
                                    'simple_optimizer',
                                    'metrics_smoothing_episodes',
                                    'seed',
                                    'vf_share_layers')
        self.rllib_config = {}
        for p in rllib_config_value_names:
            self.rllib_config[p] = spec_tree.get_value(p)

        # Model only created so that it's written to expanded_spec.yaml
        # Should make this component inactive as models that are used by rllib
        # are created during the instantiation of RLLib_Model wrapper instances.
        # import pdb; pdb.set_trace()
        model = spec_tree.create_component('model', obs_space, act_space)
        # import pdb; pdb.set_trace()
        del model

    @override(Agent)
    def loop_type(self):
        return 'ray_loop'

    def name(self):
        return "PPO"
   
    def adapt(self, reward, done, next_observation):
        pass
       
    def adapt_on_end_of_episode(self):
        pass
    
    def adapt_on_end_of_sequence(self, next_observation):
        pass
    
    def loss_function(self, next_value, values, logps, actions, rewards):
        pass
    
    def train(self, next_state_value):
        pass

    def reset_adaptation_state(self):
        pass

    def reset_state(self):
        pass
        
    def step(self, observation):
        pass
