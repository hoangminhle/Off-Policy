import os
from rl_nexus.utils import utils


class OPELogger():
    '''
        Centralizes various forms of low-frequency output, such as occasional metric reports.
        Not intended for high-frequency logging (multiple calls per second throughout a run).
    '''
    def __init__(self):
        self.file_paths = []
        self.stage_num = 0
        self.last_metric = None
        self.last_y_axis_name = None
        self.last_x_axis_name = None
        self.last_x_axis_value = None
        self.last_stage_num = None
        # XT related
        self.xt_run_name = os.getenv("XT_RUN_NAME", None)
        self.xt_run = None
        if self.xt_run_name:
            from xtlib.run import Run as XTRun
            self.xt_run = XTRun()
        self.estimate_record = {}
        self.mse_record = {}
        self.all_estimators = ['PDIS', 'WPDIS', 'MB-K', 'LSTD', 'LSTDQ', 'TDREG-K', 'MWL', 'MSWL', 'MQL', 'DualDICE', \
            'TDREG-N', 'FQE', 'MB-N', 'On_Policy', 'Behavior']
        for estimator in self.all_estimators:
            self.estimate_record[estimator] = []
            self.mse_record[estimator] = []
        # self.estimate_record['On_Policy'] = []
        # self.estimate_record['LSTDQ'] = []
        # self.mse_record['LSTDQ'] = []
        # self.estimate_record['MB'] = []
        # self.mse_record['MB'] = []
        # self.estimate_record['TDREG_Neural'] = []
        # self.mse_record['TDREG_Neural'] = []
        # self.estimate_record['FQE'] = []
        # self.mse_record['FQE'] = []


    def add_output_file(self, file_path):
        # Add one console output file.
        file_path = os.path.abspath(file_path)
        self.file_paths.append(file_path)
        utils.ensure_dir_exists(file=file_path)
        output_file = open(file_path, 'w')
        output_file.close()

    def write_line(self, line):
        # Write one line to stdout and all console files.
        print(line)
        for path in self.file_paths:
            output_file = open(path, 'a')
            output_file.write(line + '\n')
            output_file.close()

    def write_ope_metrics(self, dataset_seed, metrics, result):
        # report one line
        formatting_string = '{:6.4f}'
        sz = "Dataset {} - Relative Error:".format(dataset_seed)
        for estimator, error in metrics.items():
            sz_format = ' {{}}: {}'.format(formatting_string)
            sz += sz_format.format(estimator,error)
        self.write_line(sz)
        self.estimate_record['On_Policy'].append(result['On_Policy'])
        self.estimate_record['Behavior'].append(result['Behavior'])
        for estimator, error in metrics.items():
            self.estimate_record[estimator].append(result[estimator])
            self.mse_record[estimator].append(error)
        # summary_metrics = {}
        # summary_metrics['On_Policy'] = sum(self.estimate_record['On_Policy']) / len(self.estimate_record['On_Policy'])
        # summary_metrics['LSTDQ'] = sum(self.estimate_record['LSTDQ']) / len(self.estimate_record['LSTDQ'])
        # summary_metrics['squared_error'] = sum(self.mse_record['LSTDQ']) / len(self.mse_record['LSTDQ'])
        # print(summary_metrics)

        # if self.xt_run:
        #     xt_metrics = {}
        #     xt_metrics["True Val"] = result['On_Policy']
        #     # xt_metrics[x_axis_name] = x_axis_value
        #     for estimator, error in metrics.items():
        #         xt_metrics[estimator] = result[estimator]
        #         xt_metrics['squared_error'] = error
        #         # xt_metrics[estimator] = error
        #     # self.xt_run.log_metrics(data_dict=xt_metrics, step_name="Dataset")
        #     self.xt_run.log_metrics(data_dict=xt_metrics)


    def write_and_condense_metrics(self, total_seconds, x_axis_name, x_axis_value, saved, metrics, tf_writer):
        '''
            Outputs the given metric values for the last reporting period and condenses the metric.
        '''
        hours = total_seconds / 3600
        self.last_x_axis_name = x_axis_name
        self.last_x_axis_value = x_axis_value
        self.last_stage_num = self.stage_num

        # Report one line.
        sz = "{:7.3f} hrs  {:12,d} {}".format(hours, x_axis_value, x_axis_name)

        # Write one line of formatted metrics.
        for metric in metrics:
            sz_format = '      {} {{}}'.format(metric.formatting_string)
            sz += sz_format.format(metric.aggregate_value, metric.short_name)
        if saved:
            sz += "    SAVED"
        self.write_line(sz)

        if self.xt_run:
            # Log metrics to XT
            xt_metrics = {}
            xt_metrics["hrs"] = hours
            xt_metrics[x_axis_name] = x_axis_value
            for metric in metrics:
                xt_metrics[metric.short_name] = metric.aggregate_value
            self.xt_run.log_metrics(data_dict=xt_metrics, step_name=x_axis_name, stage='s{}'.format(self.stage_num))

        if tf_writer:
            # Log metrics to tensorboard.
            for metric in metrics:
                tf_writer.add_scalar(metric.long_name, metric.aggregate_value, x_axis_value)
            tf_writer.flush()

        # Condense the metrics
        for metric in metrics:
            metric.condense_values()

    def summarize_stage(self, metric):
        '''
            Outputs the metric value for the entire processing stage.
        '''
        metric.condense_values() # Condense any values accumulated since the last report.
        sz_format = 'Stage summary (mean {{}}):  {}'.format(metric.formatting_string)
        self.write_line(sz_format.format(metric.long_name, metric.lifetime_value))
        self.last_metric = metric
        self.last_y_axis_name = metric.short_name
        return metric.lifetime_value

    def finish_run(self, in_hp_search):
        '''
            Outputs the final stage's summary metric as hpmax (used for hyperparameter tuning).
        '''
        summary_metrics = {}
        summary_metrics['On_Policy'] = sum(self.estimate_record['On_Policy']) / len(self.estimate_record['On_Policy'])
        summary_metrics['Behavior'] = sum(self.estimate_record['Behavior']) / len(self.estimate_record['Behavior'])
        # summary_metrics['MB'] = sum(self.estimate_record['MB']) / len(self.estimate_record['MB'])
        # summary_metrics['squared_error'] = sum(self.mse_record['MB']) / len(self.mse_record['MB'])
        # summary_metrics['TDREG_Neural'] = sum(self.estimate_record['TDREG_Neural']) / len(self.estimate_record['TDREG_Neural'])
        # summary_metrics['squared_error'] = sum(self.mse_record['TDREG_Neural']) / len(self.mse_record['TDREG_Neural'])
        for estimator in self.all_estimators:
            if estimator != 'On_Policy' and estimator != 'Behavior' and len(self.estimate_record[estimator]) >0:
                summary_metrics[estimator] = sum(self.estimate_record[estimator]) / len(self.estimate_record[estimator])
                summary_metrics[estimator+'_se'] = sum(self.mse_record[estimator]) / len(self.mse_record[estimator])
        # summary_metrics['FQE'] = sum(self.estimate_record['FQE']) / len(self.estimate_record['FQE'])
        # summary_metrics['squared_error'] = sum(self.mse_record['FQE']) / len(self.mse_record['FQE'])

        # print(summary_metrics)

        if self.xt_run:
            self.xt_run.log_metrics(data_dict = summary_metrics)
        # if self.last_metric:
        #     # Log hpmax.
        #     explanation = 'Objective that would be maximized by hyperparameter tuning (hpmax):'
        #     hpmax = self.last_metric.lifetime_value
        #     if not self.last_metric.higher_is_better:
        #         hpmax = -hpmax
        #     if self.xt_run:
        #         # Log hpmax to XT
        #         xt_metrics = {}
        #         xt_metrics[self.last_x_axis_name] = self.last_x_axis_value
        #         xt_metrics['hpmax'] = hpmax
        #         self.xt_run.log_metrics(data_dict=xt_metrics, step_name=self.last_x_axis_name)
        #         self.xt_run.tag_job({'plotted_metric': 's{}-{}'.format(self.last_stage_num, self.last_y_axis_name)})
        #         # self.xt_run.tag_job({'primary_metric': 'hpmax'})  # To override xt_config.yaml's default of 'hpmax'.
        #         # self.xt_run.tag_job({'step_name': 'iters'})  # To override xt_config.yaml's default of 'iters'.
        #         if in_hp_search:
        #             explanation = 'Objective being maximized by hyperparameter tuning (hpmax):'
        #     sz_format = '{}  {}\n'.format(explanation, self.last_metric.formatting_string)
        #     self.write_line(sz_format.format(hpmax))


logger = OPELogger()

