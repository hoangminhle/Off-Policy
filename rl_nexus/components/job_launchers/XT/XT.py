from xtlib.helpers.xt_config import get_merged_config
from xtlib.impl_compute import ImplCompute
from xtlib import console


class XT():
    def __init__(self, spec_tree):
        enabled                 = spec_tree['enabled']
        self.hp_tuning          = spec_tree['hp_tuning']
        self.job_mgr_args = {
            'total_runs':         spec_tree['total_runs'],
            'compute_nodes':      spec_tree['compute_nodes'],
            'runs_per_node':      spec_tree['runs_per_node'],
            'compute_target':     spec_tree['compute_target'],
            'low_priority':       spec_tree['low_priority'],
            'hold':               spec_tree['hold'],
            'monitor':            spec_tree['monitor']}

        if self.hp_tuning:
            assert self.hp_tuning in {'random', 'grid', 'dgd', 'bayesian'}, 'hp_tuning method "{}" not recognized'.format(self.hp_tuning)

        if self.job_mgr_args['total_runs'] > 10000:
            raise ValueError("XT property 'total_runs' cannot exceed 10000")

    def launch_job(self, job_spec_name):
        console.set_level("normal")  # Use "detail" for debugging.
        args = {"script": 'run.py', "script_args": [job_spec_name]}

        if self.hp_tuning:
            args["search_type"] = self.hp_tuning
            args["hp_config"] = 'uploaded_hp_config.yaml'

        # Translate XT component property names to xtlib names.
        xt_names = {
            "total_runs":         "runs",
            "compute_nodes":      "nodes",
            "runs_per_node":      "concurrent",
            "compute_target":     "target",
            "low_priority":       "low_pri",
            "hold":               "hold",
            "monitor":            "monitor"}
            
        for key, value in self.job_mgr_args.items():
            name = xt_names[key]
            args[name] = value

        # Assemble the full argument list.
        config = get_merged_config()
        ic = ImplCompute(config)
        full_args = ic.validate_and_add_defaults("run", args)

        # Launch the job.
        ic.run(full_args)
