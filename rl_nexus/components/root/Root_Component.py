import os
import torch
import gc
from os.path import join as pjoin
from shutil import copyfile
from rl_nexus.utils.spec_tree import SpecTree
from rl_nexus.utils import utils
# from rl_nexus.utils.nexus_logger import logger
from rl_nexus.utils.ope_logger import logger


class Root_Component:
    def __init__(self, spec_tree, run_spec_path, hp_handler):
        self.spec_tree = spec_tree
        self.hp_handler = hp_handler

        # Get spec values.
        self.cuda               = spec_tree['cuda']
        self.log_to_tensorboard = spec_tree['log_to_tensorboard']
        self.experiment_path    = spec_tree['experiment_path']

        # Begin writing two copies of the console output.
        logger.add_output_file('console.txt')
        logger.add_output_file(os.path.join(self.experiment_path, 'console.txt'))

        # Check the experiment_path.
        if not self.experiment_path.startswith('../results'):
            logger.write_line("WARNING: experiment_path \'{}\' (found in runspec) does not begin with '../results'. "
                              "Job results will not be mirrored to Azure Storage.".format(self.experiment_path))

        # Copy the launched runspec to results folder
        dest = pjoin(self.experiment_path, os.path.basename(run_spec_path))
        if run_spec_path != dest:
            copyfile(run_spec_path, dest)

        # Is this app running as part of a launched job?
        in_job = os.getenv("XT_RUN_NAME")
        if in_job:
            # Yes. Don't create another job launcher.
            self.job_launcher = None
        else:
            # No. Try to instantiate a job launcher.
            self.job_launcher = spec_tree.create_component('job_launcher')
            if self.job_launcher and self.job_launcher.hp_tuning:
                self.hp_handler.write_hp_config_file()


        # Write the top portion of the repro spec tree to two files,
        # one in the rl_nexus dir, and the other in the experiment_path dir.
        local_repro_spec_path = 'repro_spec.yaml'
        exper_repro_spec_path = os.path.join(self.experiment_path, 'repro_spec.yaml')
        utils.ensure_dir_exists(file=exper_repro_spec_path)
        self.repro_spec_paths = (local_repro_spec_path, exper_repro_spec_path)
        self.write_to_repro_spec(self.spec_tree, '', 'w')
        self.write_to_repro_spec('\nprocessing_stages:\n', '', 'a')

    def launch_job(self, run_spec_path):
        # Copy the runspec to rl_nexus so that remote instances of rl_nexus can use it,
        # and so that users can inspect it after job launch.
        job_spec_name = 'launched_job_spec.yaml'
        copyfile(run_spec_path, job_spec_name)
        logger.write_line('------- Runspec ({}) copied to {}'.format(run_spec_path, job_spec_name))
        self.job_launcher.launch_job(job_spec_name)
        logger.write_line('------- Job launched. Use XT commands to access results.')

    def execute_processing_stages(self):
        # Choose the device for the processing stages to use.
        if self.cuda and not torch.cuda.is_available():
            logger.write_line("WARNING: no GPU found! Failing over to cpu.")
        device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")

        # Log the chosen hyperparameter values.
        self.hp_handler.log_chosen_values(logger)

        # Step through the processing stages.
        processing_stages = self.spec_tree['processing_stages']
        stage_results = []
        for idx, list_item in enumerate(processing_stages):
            logger.stage_num = idx + 1

            st = '{} of {}'.format(logger.stage_num, len(processing_stages))
            logger.write_line('Processing stage {}'.format(st))
            self.write_to_repro_spec('  # {}\n'.format(st), '', 'a')

            processing_stage = self.spec_tree.create_component(list_item['processing_stage'],
                                                               device)
            if not processing_stage:
                st = "(not enabled)\n"
                logger.write_line('{}'.format(st))
                self.write_to_repro_spec('  # {}\n'.format(st), '', 'a')

            if processing_stage:
                # Write this processing stage to the repro spec.
                self.write_to_repro_spec('  - processing_stage:\n', '', 'a')
                self.write_to_repro_spec(processing_stage.spec_tree, '      ', 'a')
                self.write_to_repro_spec('\n', '', 'a')

                # Execute.
                logger.write_line('{}:  Started'.format(processing_stage.component_name))
                stage_result = processing_stage.execute()
                stage_results.append(stage_result)
                logger.write_line('{}:  Completed\n'.format(processing_stage.component_name))
                gc.collect()
        logger.finish_run(self.hp_handler.in_hp_search)
        logger.write_line('All processing stages completed.')
        return stage_results

    def write_to_repro_spec(self, content, indentation, mode):
        for path in self.repro_spec_paths:
            repro_spec_file = open(path, mode)
            if isinstance(content, SpecTree):
                content.dump(repro_spec_file, indentation)
            else:
                repro_spec_file.write(content)
            repro_spec_file.close()
