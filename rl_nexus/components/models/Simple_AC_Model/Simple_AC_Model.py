import logging
import numpy as np
# from base.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from rl_nexus.utils.utils import resolve_path, ensure_dir_exists

torch,nn = try_import_torch()

logger = logging.getLogger(__name__)

class Simple_AC_Model(TorchModelV2, nn.Module):
    def __init__(self, spec_tree, obs_space, action_space):
        fcnet_hiddens = spec_tree['fcnet_hiddens']
        fcnet_activation = spec_tree['fcnet_activation']
        vf_share_layers = spec_tree['vf_share_layers']
        model_config = {'fcnet_hiddens': fcnet_hiddens,
                        'fcnet_activation': fcnet_activation,
                        'vf_share_layers': vf_share_layers}
        name = 'MLP'
        num_outputs = action_space.n
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        # import pdb; pdb.set_trace()
        assert model_config['vf_share_layers'] is False, 'Separate the value and policy branches for simplicity'
        layers = []
        input_layer = nn.Linear(int(np.product(obs_space.shape)), fcnet_hiddens[0])
        if fcnet_activation == 'tanh':
            activation_layer = nn.Tanh()
        elif fcnet_activation == 'relu':
            activation_layer = nn.ReLU()
        
        layers.append(input_layer)
        layers.append(activation_layer)

        for i in range(len(fcnet_hiddens)-1):
            hidden_layer = nn.Linear(fcnet_hiddens[i], fcnet_hiddens[i+1])
            layers.append(hidden_layer)
            layers.append(activation_layer)
        output_layer = nn.Linear(fcnet_hiddens[-1], num_outputs)
        layers.append(output_layer)
        
        self.policy_network = nn.Sequential(*layers)

        value_layers = []
        input_layer = nn.Linear(int(np.product(obs_space.shape)), fcnet_hiddens[0])
        value_layers.append(input_layer)
        value_layers.append(activation_layer)
        for i in range(len(fcnet_hiddens)-1):
            hidden_layer = nn.Linear(fcnet_hiddens[i], fcnet_hiddens[i+1])
            value_layers.append(hidden_layer)
            value_layers.append(activation_layer)
        output_layer = nn.Linear(fcnet_hiddens[-1], 1)
        value_layers.append(output_layer)

        self.value_network = nn.Sequential(*value_layers)

        # self.policy_network = nn.Sequential(
        #                         nn.Linear(int(np.product(obs_space.shape)), layer_size),
        #                         nn.Tanh(),
        #                         nn.Linear(layer_size,layer_size),
        #                         nn.Tanh(),
        #                         nn.Linear(layer_size, num_outputs)
        #                         )
        # self.value_network = nn.Sequential(
        #                         nn.Linear(int(np.product(obs_space.shape)), 64),
        #                         nn.Tanh(),
        #                         nn.Linear(64,64),
        #                         nn.Tanh(),
        #                         nn.Linear(64, 1)
        #                         )
        self._last_flat_in = None                                

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        logits = self.policy_network(self._last_flat_in)
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        return self.value_network(self._last_flat_in).squeeze(1)

    def save(self, save_paths_dict, verbose = False):
        # self.policy_network.save(save_paths_dict, verbose)
        # import pdb; pdb.set_trace()
        torch.save(self.policy_network.state_dict(), save_paths_dict['save_model_to'])
    
    def save_model_in_progress(self, save_paths_dict, policy_tag):
        save_path = save_paths_dict['save_model_to']+'_'+str(policy_tag)+".pt"
        ensure_dir_exists(file=save_path)
        torch.save(self.policy_network.state_dict(), save_path)
    
