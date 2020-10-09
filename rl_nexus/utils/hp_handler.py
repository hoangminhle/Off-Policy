import os, io
import oyaml as yaml
import random
import time
from rl_nexus.utils.spec_tree import SpecTree

hp_rand = random.Random(time.time() + os.getpid())


class HyperparameterHandler():
    def __init__(self):
        """
        Preprocesses the runspec before the call to yaml.load().
        Manages communication with XT regarding hyperparameters.
        """
        self.uploaded_hp_config_filename = 'uploaded_hp_config.yaml'
        self.downloaded_hp_config_filename = 'downloaded_hp_config.yaml'
        self.xt_run_name = os.getenv("XT_RUN_NAME")
        self.xt_run = None
        self.in_hp_search = False
        self.randint_in_spec = False
        if self.xt_run_name:
            from xtlib.run import Run as XTRun
            self.xt_run = XTRun()
            if os.path.isfile(self.downloaded_hp_config_filename):
                self.in_hp_search = True
        self.hparams = []

    def split_spec(self, run_spec_file):
        # Read the spec into 3 sections.
        pre_hp_section = []
        hp_section = []
        post_hp_section = []
        current_section = pre_hp_section
        for line in run_spec_file:
            if current_section == pre_hp_section:
                # Look for the start of the hp section.
                if line.startswith('hyperparameters:'):
                    current_section = hp_section
            elif current_section == hp_section:
                # Look for the end of the hp section.
                if line[0] not in ' -#\n\r':
                    current_section = post_hp_section
            else:
                assert current_section == post_hp_section
            # Append this line to the current section.
            current_section.append(line)
        return pre_hp_section, hp_section, post_hp_section

    def preprocess(self, run_spec_file):
        """ Modifies the hyperparameter section of a runspec before yaml.load() is called on it. """

        # Read the spec into 3 sections.
        pre_hp_section, hp_section, post_hp_section = self.split_spec(run_spec_file)

        # Modify the HP section, if present.
        if len(hp_section) > 0:
            self.hparams = self.parse_hp_section(hp_section)
            if self.in_hp_search:
                self.read_hp_config_file()
            else:
                for hp in self.hparams:
                    hp.choose_value(self.in_hp_search)
            parsed_hp_section = ['hyperparameters:\n']
            for hp in self.hparams:
                parsed_hp_section += hp.format_chosen_value()
            parsed_hp_section.append('\n')
        else:
            parsed_hp_section = []

        # Reassemble the modified runspec.
        spec_str = ''.join(pre_hp_section + parsed_hp_section + post_hp_section)

        # Check for randint.
        self.randint_in_spec = 'randint' in spec_str

        # Return the modified runspec.
        return spec_str

    def parse_hp_section(self, hp_section_in):
        """
        Parses the hyperparameters section of a runspec.
        Returns a list of Hparam objects. For example...
        Input string hp_section_in:
            hyperparameters:
              - name: &rscale
                  ordered_tuning_values: [2, 4, 8, 16, 32]
                  tuned_value: 32
              - name: &units
                  ordered_tuning_values: [128, 192, 256, 384, 512]
                  tuned_value: 384
        Output returned:
            List of Hparam objects:
                hp[0].name = 'rscale'
                     .values = [2, 4, 8, 16, 32]
                     .tuned_value = 32
                hp[1].name = 'units'
                     .values = [128, 192, 256, 384, 512]
                     .tuned_value = 384
        """
        hparams = []
        name_line = ''
        values_line = ''
        i = 0
        for full_line in hp_section_in:
            line = full_line.strip().rstrip()
            if line.startswith('hyperparameters:') or (len(line) == 0) or (line[0] == '#'):
                continue
            if i == 0:
                if line.startswith('- name:'):
                    name_line = line
                    i = 1
                else:
                    raise SyntaxError('First line of a hyperparameter definition must start with "- name:"\n=====> {}'.format(line))
            elif i == 1:
                if (line.startswith('ordered_tuning_values:')) or (line.startswith('unordered_tuning_values:')):
                    values_line = line
                    i = 2
                else:
                    raise SyntaxError('Second line of a hyperparameter definition must start with "ordered_tuning_values:" or "unordered_tuning_values:"\n=====> {}'.format(line))
            elif i == 2:
                if line.startswith('tuned_value:'):
                    hp = Hparam(name_line, values_line, line)
                    hparams.append(hp)
                    i = 0
                else:
                    raise SyntaxError('Third line of a hyperparameter definition must start with "tuned_value:"\n=====> {}'.format(line))
            else:
                raise SyntaxError('Unexpected line in the hyperparameters section of the runspec:{}'.format(line))
        return hparams

    def log_chosen_values(self, logger):
        """ Logs the chosen HP values to the console for reference, and (optionally) to XT. """
        if len(self.hparams) > 0:
            hparam_dict = {}
            logger.write_line("Chosen hyperparameter values:")
            for hp in self.hparams:
                hp.log_chosen_value(logger)
                hparam_dict[hp.name] = hp.chosen_value
            logger.write_line('')
            if self.xt_run:
                self.xt_run.log_hparams(hparam_dict)

    def write_hp_config_file(self):
        """ Generates the file that XT needs to support HP tuning. """
        assert len(self.hparams) > 0, 'Hyperparameters must be specified.'
        # Warn the user if randint is missing from a hyperparameter search.
        if not self.randint_in_spec:
            response = None
            while (response != 'y') and (response != 'n'):
                print("WARNING: Hyperparameter tuning typically requires randomization,")
                print("which is usually achieved by setting the environment or agent seed to randint,")
                print("but randint is missing from this runspec. Are you sure you want to proceed? [y/n]")
                response = input()
            if response == 'n':
                exit(0)
        # Write the hp config file for the job launcher.
        hp_config_file = open(self.uploaded_hp_config_filename, 'w')
        hp_config_file.write('hyperparameter-distributions:\n')
        for hp in self.hparams:
            value_list = []
            for value in hp.values:
                value_list.append(hp.yaml_value_from_python(value))
            values_str = ', '.join(value_list)
            hp_config_file.write('  {}: [{}]\n'.format(hp.name, values_str))
        hp_config_file.close()

    def read_hp_config_file(self):
        """ Reads the file containing the HP values chosen by XT. """
        assert len(self.hparams) > 0, 'Hyperparameters must be specified.'
        print('Reading chosen hp values from downloaded_hp_config.yaml')
        chosen_hp_value_dict = yaml.load(open(self.downloaded_hp_config_filename, 'r'), Loader=yaml.Loader)
        hp_runset = chosen_hp_value_dict['hyperparameter-runset']
        # for hp_name in hp_runset:
        #     print('{}  {}'.format(hp_name, hp_runset[hp_name]))
        assert len(hp_runset) == len(self.hparams)
        for hp in self.hparams:
            hp.chosen_value = hp_runset[hp.name]


class Hparam():
    def __init__(self, name_line, values_line, best_line):
        """ Parses one hparam's 3 lines from the runspec. """
        # Get the hp name.
        parts = name_line.split()
        assert parts[0] == '-'
        assert parts[1] == 'name:'
        assert parts[2][0] == '&'
        self.name = parts[2][1:]
        # Get the list of values.
        parts = values_line.split()
        assert (parts[0] == 'ordered_tuning_values:') or (parts[0] == 'unordered_tuning_values:')
        i1 = values_line.index('[')
        i2 = values_line.index(']')
        self.values_str = values_line[i1:i2+1]
        value_strs = values_line[i1+1:i2].replace(',', ' ').split()
        self.values = []
        for value_str in value_strs:
            self.values.append(self.python_value_from_yaml(value_str))
        # Get the tuned value.
        parts = best_line.split()
        assert parts[0] == 'tuned_value:'
        self.tuned_value = self.python_value_from_yaml(parts[1])
        self.chosen_value = None

    def python_value_from_yaml(self, value_str):
        """ Converts a yaml value string into a python value. """
        value = value_str
        if value_str == 'null':
            value = None
        elif (value_str == 'true') or (value_str == 'True'):
            value = True
        elif (value_str == 'false') or (value_str == 'False'):
            value = False
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
        return value

    def yaml_value_from_python(self, value):
        """ Converts a python value into a yaml value string. """
        if value is True:
            return 'true'
        elif value is False:
            return 'false'
        elif value is None:
            return 'null'
        elif isinstance(value, float):
            # Avoid a bug in yaml parsing of scientific notation.
            str = '{}'.format(value)
            if (len(str) > 1) and (str[1] == 'e'):
                str = str.replace('e', '.0e')
            return str
        elif isinstance(value, int):
            return '{}'.format(value)
        else:
            return value

    def choose_value(self, in_hp_search):
        """ Chooses one value, depending on the current context. """
        if in_hp_search:
            # Choose randomly. (Not relying on XT yet.)
            self.chosen_value = hp_rand.choice(self.values)
        else:
            # Choose the tuned value.
            self.chosen_value = self.tuned_value

    def format_chosen_value(self):
        """ Returns 2 lines for this hparam in the post-processed yaml. """
        name_line = '  - name: &{}\n'.format(self.name)
        best_line = '      {}\n'.format(self.yaml_value_from_python(self.chosen_value))
        return [name_line, best_line]

    def format_hp_block(self):
        """ Returns 3 lines for this hparam in a new runspec. """
        name_line = '  - name: &{}\n'.format(self.name)
        values_line = '      ordered_tuning_values: {}\n'.format(self.values_str)
        tuned_value_line = '      tuned_value: {}\n'.format(self.yaml_value_from_python(self.chosen_value))
        return [name_line, values_line, tuned_value_line]

    def log_chosen_value(self, logger):
        """ Logs one value to the console for reference. """
        logger.write_line('  {}: {}'.format(self.name, self.yaml_value_from_python(self.chosen_value)))
