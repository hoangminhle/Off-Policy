import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class OPE_E2E():
    def __init__(self, spec_tree, device):
        self.spec_tree = spec_tree
        self.device = device
        self.dataset_seed = spec_tree['dataset_seed']
        # import pdb; pdb.set_trace()
    def execute(self):
        #* Assign the data seed to the data collector
        self.spec_tree['data_collector']['dataset_seed'] = self.dataset_seed
        self.data_collector = self.spec_tree.create_component('data_collector', self.device)
        self.data_collector.execute()

        #* Assign the same data seed to the evaluator
        self.spec_tree['off_policy_estimator']['dataset_seed'] = self.dataset_seed
        self.off_policy_evaluation = self.spec_tree.create_component('off_policy_estimator', self.device)
        self.off_policy_evaluation.execute()
