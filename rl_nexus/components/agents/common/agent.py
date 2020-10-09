from abc import ABC, abstractmethod
import torch
import torch.nn as nn


"""
Abstract trainable agent class. In addition to the basic Agent class functionality,
trainable agents can learn or adapt the policy model and save it, for which they
have several additional class methods.
"""
class Agent(ABC):
    # Override the constructor to specify how exactly the model is initialized
    @abstractmethod
    def __init__(self, spec_tree, obs_space, act_space, device):
        super().__init__()
        # This is a gym.spaces.Tuple, gym.spaces.Discrete, or gym.spaces.Box object
        self.obs_space = obs_space
        # For now we assume that this is a gym.spaces.Discrete object
        self.act_space = act_space
        self.model: nn.Module = None

    @staticmethod
    def count_parameters(network):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)

    def loop_type(self):
        return 'rl_loop'

    def save_model(self, save_paths_dict, verbose):
        self.model.save(save_paths_dict, verbose)

    def load_model(self, load_paths_dict, verbose):
        self.model.load(load_paths_dict, verbose)

    @abstractmethod
    def adapt(self, observation, reward, done, history):
        pass

    @abstractmethod
    def adapt_on_end_of_episode(self):
        pass

    @abstractmethod
    def adapt_on_end_of_sequence(self, observation, history):
        pass

    @abstractmethod
    def loss_function(self, next_value, values, logps, actions, rewards):
        pass

    @abstractmethod
    def train(self, next_state_value):
        pass

    @abstractmethod
    def reset_adaptation_state(self):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def step(self, observation, history):
        pass


"""
The abstract static agent class describes the functionality of an agent that
can only execute, not learn or otherwise modify, its policy. Thus, all its methods
intended for train a policy or saving it don't do anything. The reset() and step()
methods are still abstract.
"""
class StaticAgent(Agent):
    def __init__(self, spec_tree, obs_space, act_space, device):
        super().__init__(spec_tree, obs_space, act_space, device)

    def save_model(self, save_paths_dict, flag):
        pass

    def adapt(self, observation, reward, done, history):
        pass

    def adapt_on_end_of_episode(self):
        pass

    def adapt_on_end_of_sequence(self, observation, history):
        pass

    def loss_function(self, next_value, values, logps, actions, rewards):
        pass

    def train(self, next_state_value):
        pass

    def reset_adaptation_state(self):
        pass

