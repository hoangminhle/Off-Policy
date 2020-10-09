import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl_nexus.utils.utils import resolve_path, ensure_dir_exists
from rl_nexus.utils.nexus_logger import logger

class Train_Policy():
    def __init__(self, spec_tree, device):
        self.spec_tree = spec_tree
        self.device    = device

        # Get spec values
        self.enabled              = spec_tree['enabled']
        self.save_model_to        = resolve_path(spec_tree['save_model_to'])
        self.save_logs_to         = resolve_path(spec_tree['save_logs_to'])
        self.max_iterations       = spec_tree['max_iterations']
        self.iters_per_report     = spec_tree['iters_per_report']
        self.get_action_from_env  = spec_tree['get_action_from_env']
        self.train                = spec_tree['train']
        self.render               = spec_tree['render']
        self.model_load_paths_dict = {
            'load_model_from':      resolve_path(spec_tree['load_model_from']),
            'load_backbone_from':   resolve_path(spec_tree['load_backbone_from']),
            'load_core_from':       resolve_path(spec_tree['load_core_from']),
            'load_embedding_from':  resolve_path(spec_tree['load_embedding_from']),
            'load_head_from':       resolve_path(spec_tree['load_head_from'])
        }
        self.model_save_paths_dict = {
            'save_model_to':        resolve_path(spec_tree['save_model_to']),
            'save_backbone_to':     resolve_path(spec_tree['save_backbone_to']),
            'save_core_to':         None,
            'save_embedding_to':    None,
            'save_head_to':         resolve_path(spec_tree['save_head_to'])
        }

        # Environment component
        self.environment = spec_tree.create_component('environment')

        # Agent component
        self.agent = spec_tree.create_component('agent',
                                                self.environment.observation_space,
                                                self.environment.action_space,
                                                device)
        # import pdb; pdb.set_trace()
        # XT related
        self.xt_run_name = os.getenv("XT_RUN_NAME", None)
        self.xt_run = None
        if self.xt_run_name:
            from xtlib.run import Run as XTRun
            self.xt_run = XTRun()
            # log hyperparameter values to XT. (in progress)
            #hd = cf.get_hparam_dict()
            #self.xt_run.log_hparams( hd )

        if self.agent.loop_type() is not 'ray_loop':
            evaluation_num_episodes = spec_tree['evaluation_num_episodes']
            assert evaluation_num_episodes == 0, 'Only rllib\'s algorithm implementations support intra-stage evaluation.' 
            self.agent.load_model(self.model_load_paths_dict, True)

        if self.save_model_to:
            ensure_dir_exists(file=self.save_model_to)
            logger.write_line("Saving models to {}".format(self.save_model_to))

        # Switch the agent into eval mode if requested
        if not self.train and not spec_tree['disable_eval']:
            if self.agent.model is not None:
                self.agent.model.eval()

        self.metric_data_list = []

    def init_episode(self):
        """ Reset the environment and return the initial observation. """
        self.agent.reset_state()
        observation = self.environment.reset()
        return np.copy(observation)

    def _extract_history(self, observation):
        """ Deconstructs a complex Dictionary-based observation into the current_observation
            and the history.
        """
        history = ()

        # Oddly Dictionary observations are returned as ndarrays when env.reset() is called,
        # but dict when env.step() is called. Here we check for both cases...
        if isinstance(observation, dict) or \
           (observation.shape == () and 'observation' in observation.item()):
            d = observation if isinstance(observation, dict) else observation.item()
            observation_tt = torch.tensor([d['observation']], device=self.device)
            if 'prev_actions' in d and 'prev_rewards' in d and 'observations' in d:
                # Add the batch dimension to prev_observations/actions/rewards.
                observations = torch.tensor([d['observations']], device=self.device)
                prev_actions = torch.tensor([d['prev_actions']], device=self.device)
                prev_rewards = torch.tensor([d['prev_rewards']], device=self.device)
                history = (observations, prev_actions, prev_rewards)
        else:
            observation_tt = torch.tensor([observation], device=self.device)
            return observation_tt, ()

        return observation_tt, history

    def execute(self):
        loop_type = self.agent.loop_type()
        if loop_type == 'ray_loop':
            """ Run the rllib loop """
            from rl_nexus.components.processors.common.ray_loop import Ray_Loop
            rl_loop = Ray_Loop(processor=self)
            return rl_loop.execute()
        else:
            raise NotImplementedError

        # """ Run the main RL Loop """
        # self.tf_writer = SummaryWriter(self.save_logs_to) if self.save_logs_to else None
        # start_time = time.time()
        # observation = self.init_episode()
        # observation_tt, history = self._extract_history(observation)
        # metrics = self.environment.metrics
        # for iteration in range(1, self.max_iterations + 1):
        #     # Get an action.
        #     if self.get_action_from_env == True:
        #         action = self.environment.sample_expert_action()
        #     else:
        #         import pdb; pdb.set_trace()
        #         action = self.agent.step(observation_tt, history)

        #     if self.render:
        #         self.environment.render()

        #     # Apply the action to the environment, which may say the episode is done.
        #     observation, reward, done, info = self.environment.step(action)
        #     observation_tt, history = self._extract_history(observation)

        #     # Let the agent adapt to the effects of the action before a new episode can be initialized.
        #     if self.train:
        #         self.agent.adapt(observation_tt, reward, done, history)

        #     # After an episode ends, start a new one.
        #     if done:
        #         observation = self.init_episode()
        #         observation_tt, history = self._extract_history(observation)

        #     # Report results periodically.
        #     if iteration % self.iters_per_report == 0:
        #         total_seconds = time.time() - start_time

        #         # Consider whether to save a model.
        #         saved = False
        #         if self.save_model_to and self.train and metrics[0].currently_optimal:
        #             self.agent.save_model(self.model_save_paths_dict, False)
        #             saved = True

        #         # Write the metrics for this reporting period.
        #         logger.write_and_condense_metrics(total_seconds, 'iters', iteration, saved,
        #                                           metrics, self.tf_writer)

        # # Consider whether to save a model.
        # if self.save_model_to and self.train and metrics[0].currently_optimal:
        #     self.agent.save_model(self.model_save_paths_dict, False)

        # # Get a summary metric for the entire stage, based on the environment's first metric.
        # summary_metric = logger.summarize_stage(metrics[0])

        # self.environment.close()

        # return summary_metric
