import os
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gym

from rl_nexus.utils.nexus_logger import logger

import ray
from ray import tune
from ray.tune.logger import UnifiedLogger
#from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env

from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib import _register_all

from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.utils import try_import_torch
_, nn = try_import_torch()



class RLLib_Model(TorchModelV2, nn.Module):
    """RLLib custom pytorch model wrapper"""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, proc_spec_tree, model_load_paths_dict, train_mode, original_obs_space):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Note: For complex Dictionary spaces as created by Env_Formatter,
        # we pass the original_obs_space separately since RLLib mangles the obs_space
        # by applying a DictFlatteningPreprocessor.
        if isinstance(original_obs_space, gym.spaces.dict.Dict):
            obs_space = original_obs_space["observation"]

        (_, _, agent_spec_tree) = proc_spec_tree.get_component_info('agent')
        self.model = agent_spec_tree.create_component('model', obs_space, action_space)
        self.model_load_paths_dict = model_load_paths_dict
        self.model.load(model_load_paths_dict, True)

        self._cur_value = None

        if train_mode:
            self.train()
        else:
            self.eval()

        print("{:11,d} trainable parameters".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        import pdb; pdb.set_trace()

    @override(TorchModelV2)
    def get_initial_state(self):
        init_state = self.model.init_state()
        if init_state is None:
            return []
        return init_state

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        if isinstance(obs, dict):
            observations = obs['observations']
            prev_actions = obs['prev_actions']
            prev_rewards = obs['prev_rewards']
            history = (observations, prev_actions, prev_rewards)
            obs = obs["observation"]
        else:
            history = ()

        obs = obs.float()

        if len(state) == 0:
            state = None
        elif len(state) == 1:
            state = state[0] # Unpack from list

        output, new_state = self.model(obs, state, history, seq_lens)

        if len(output) == 2:
            self._cur_value = output[0].squeeze(1)
            logits = output[1]
        else:
            self._cur_value = output.squeeze(1)
            logits = output

        if new_state is None:
            new_state = []
        else:
            new_state = [new_state]

        return logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save(self, save_paths_dict, verbose = False):
        self.model.save(save_paths_dict, verbose)


def rllib_logger_creator(config):
    """Creates a Unified logger with a provided logdir prefix
    """
    logdir = os.path.join(config["evaluation_config"]["save_logs_to"], "rllib")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return UnifiedLogger(config, logdir, loggers=None)


class RLlib_Loop(object):
    def __init__(self, processor):
        self.pr = processor

        # TEMPORARY CHANGE: setting itertion values to be time step values to preserve time step functionality
        self.pr.total_steps = self.pr.max_iterations
        self.pr.reporting_interval = self.pr.iters_per_report

        if not self.pr.train:
            # Disable saving the model when not training and use a single worker/environment
            self.pr.save_model_to = None
            self.pr.agent.rllib_config["num_workers"] = 0
            self.pr.agent.rllib_config["num_envs_per_worker"] = 1
            self.pr.agent.rllib_config["rollout_fragment_length"] = self.pr.reporting_interval
            self.pr.agent.rllib_config["batch_mode"] = "complete_episodes"
            #self.pr.agent.rllib_config["num_gpus"] = 0
            #self.pr.agent.rllib_config["explore"] = False

        if self.pr.save_model_to is None:
            # Clear model_save_paths_dict as indicator no model is saved
            self.pr.model_save_paths_dict = None

        proc_spec_tree = self.pr.spec_tree
        model_load_paths_dict = self.pr.model_load_paths_dict
        train_mode = self.pr.train or self.pr.spec_tree['disable_eval']
        evaluation_num_episodes = self.pr.spec_tree['evaluation_num_episodes']
        evaluation_use_exploration = self.pr.spec_tree['evaluation_use_exploration']
        evaluation_environment = self.pr.spec_tree['evaluation_environment']
        assert evaluation_num_episodes == 0 or evaluation_environment is not None, 'If intra-stage evaluation is on, you must specify an evaluation environment.'
        timesteps_per_iteration = self.pr.reporting_interval
        save_logs_to = self.pr.save_logs_to

        # Register a custom environment creation function
        register_env("nexus_env", lambda config: \
            proc_spec_tree.create_component('environment') if config["eval_mode"] == False else proc_spec_tree.create_component('evaluation_environment'))

        self.rllib_config={
            "env": "nexus_env",
            "env_config": {
                "eval_mode": False
            },
            "model": {
                "custom_model": "rllib_model",
                "custom_model_config": {
                    "proc_spec_tree": proc_spec_tree,
                    "model_load_paths_dict": model_load_paths_dict,
                    "train_mode": train_mode,
                    "original_obs_space": self.pr.environment.observation_space,
                },
            },
            "framework": "torch",
            "min_iter_time_s": 0,
            "timesteps_per_iteration": timesteps_per_iteration,
            "evaluation_num_workers": 1,
            # If evaluation_num_episodes is 0, set evaluation_interval to 0 as well to avoid getting invalid evaluation results.
            "evaluation_interval": 1 if evaluation_num_episodes > 0 else 0,
            "evaluation_num_episodes": evaluation_num_episodes,
            "evaluation_config": {
                "save_logs_to": save_logs_to,
                "explore": evaluation_use_exploration,
                "env_config": {
                    "eval_mode": True
                }
            },
        }

        self.rllib_config.update(self.pr.agent.rllib_config)

        # Ensure that the previous stage's seed will not be reused for selecting redis ports.
        random.seed()

        ray.init(local_mode=self.pr.agent.local_mode, include_dashboard=(logger.xt_run is None))
        ModelCatalog.register_custom_model("rllib_model", RLLib_Model)

        # Environment instance not used by rllib agent as environments that are used by rllib
        # are instantiated by rllib.
        del self.pr.environment


    def execute(self):
        timesteps = 0
        best_period_value = None

        if self.pr.agent.name() == "A2C":
            trainer = A2CTrainer(config=self.rllib_config, logger_creator=rllib_logger_creator)
        elif self.pr.agent.name() == "PPO":
            trainer = PPOTrainer(config=self.rllib_config, logger_creator=rllib_logger_creator)
        else:
            raise ValueError('There is no rllib trainer with name ' + self.pr.agent.name())

        tf_writer = SummaryWriter(self.pr.save_logs_to) if self.pr.save_logs_to else None

        if self.pr.train:
            start_time = time.time()

            while timesteps < self.pr.total_steps:
                result = trainer.train()
                timesteps = result["timesteps_total"]

                # Get a metric list from each environment.
                if hasattr(trainer, "evaluation_workers"):
                    metric_lists = sum(trainer.evaluation_workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.metrics)), [])
                else:
                    metric_lists = sum(trainer.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.metrics)), [])

                metrics = metric_lists[0]

                # Aggregate metrics from all other environments.
                for metric_list in metric_lists[1:]:
                    for i, metric in enumerate(metric_list):
                        metrics[i]._values.extend(metric._values)

                save_logs_to = self.pr.save_logs_to
                model_save_paths_dict = self.pr.model_save_paths_dict
                # Consider whether to save a model.
                saved = False
                if model_save_paths_dict is not None and metrics[0].currently_optimal:
                    trainer.get_policy().model.save(model_save_paths_dict)
                    saved = True

                # Write the metrics for this reporting period.
                total_seconds = time.time() - start_time
                logger.write_and_condense_metrics(total_seconds, 'iters', timesteps, saved, metrics, tf_writer)

                # Clear the metrics, both those maintained by the training workers and by the evaluation ones.
                condense_fn = lambda environment: [m.condense_values() for m in environment.metrics]
                trainer.workers.foreach_worker(lambda w: w.foreach_env(condense_fn))
                if hasattr(trainer, "evaluation_workers"):
                    trainer.evaluation_workers.foreach_worker(lambda w: w.foreach_env(condense_fn))
        else:
            start_time = time.time()
            env = trainer.workers.local_worker().env
            metrics = env.metrics
            worker = trainer.workers.local_worker()
            steps = steps_since_report = 0

            while True:
                batch = worker.sample()
                current_steps = len(batch["obs"])
                steps += current_steps
                steps_since_report += current_steps

                if steps_since_report >= self.pr.reporting_interval:
                    total_seconds = time.time() - start_time

                    # Write the metrics for this reporting period.
                    logger.write_and_condense_metrics(total_seconds, 'iters', steps, False,
                                                      metrics, tf_writer)

                    steps_since_report = 0
                    if steps >= self.pr.total_steps:
                        break

            env.close()

        # Get a summary metric for the entire stage, based on the environment's first metric.
        summary_metric = logger.summarize_stage(metrics[0])

        # Temporary workaround for https://github.com/ray-project/ray/issues/8205
        ray.shutdown()
        _register_all()

        return summary_metric
