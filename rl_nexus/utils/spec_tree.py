import os
import random
import time
import oyaml as yaml
import importlib
from rl_nexus.utils.nexus_logger import logger

spec_rand = random.Random(time.time() + os.getpid())
rl_nexus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
components_dir = os.path.join(rl_nexus_dir, 'components')


class SpecTree():
    def __init__(self, spec):
        self.full_component_name = spec['component']
        # Store the spec items for later retrieval by the component.
        self.settings = {}
        for key in spec:
            self.settings[key] = Setting(self, key, spec[key])
        self.accessed_settings = []
        self.instantiated_subtrees = []

    def get_value(self, name, is_subcomponent=False):
        if name not in self.settings:
            raise KeyError('No setting with name \"{}\" in current settings: {}'.format(name, self.settings))
        return self.settings[name].get_value(is_subcomponent)

    def get_surface_component_info(self, name, comp_spec_dict):
        # Returns Name, SpecTree, and Path to the component without instantiating it or loading its module.
        if 'component' not in comp_spec_dict:
            raise KeyError('Error creating component \"{}\": Expected \"component\" key under {} yaml.'.format(name, name))
        spec_subtree = SpecTree(comp_spec_dict)
        comp_id = comp_spec_dict['component']
        id_parts = str.split(comp_id, '/')
        comp_subdir = id_parts[0]
        comp_name = id_parts[1]
        comp_path = '{}.{}.{}.{}'.format(comp_subdir, comp_name, comp_name, comp_name)
        return comp_name, spec_subtree, comp_path

    def get_component_info(self, name):
        # Returns the uninstantiated Class, Name, and SpecTree for the named component.
        comp_spec_dict = self.get_value(name, is_subcomponent=True)
        if not isinstance(comp_spec_dict, dict):
            raise TypeError("Error retreiving info for component \"{}\": Expected a dictionary but got \"{}: {}\".".format(name, name, comp_spec_dict))
        comp_name, spec_subtree, comp_path = self.get_surface_component_info(name, comp_spec_dict)
        comp_class = self.get_component_class(comp_path)  # This loads the component's module.
        return comp_class, comp_name, spec_subtree

    def create_component(self, name, *args):
        # Instantiates a nexus component from its spec tree.
        if isinstance(name, dict):
            comp_spec_dict = name
        else:
            comp_spec_dict = self.get_value(name, is_subcomponent=True)

        if not isinstance(comp_spec_dict, dict):
            raise TypeError("Error creating component \"{}\": Expected a dictionary but got \"{}: {}\".".format(name, name, comp_spec_dict))

        comp_name, spec_subtree, comp_path = self.get_surface_component_info(name, comp_spec_dict)
        if ('enabled' not in spec_subtree.settings) or spec_subtree.settings['enabled'].value:
            comp_class = self.get_component_class(comp_path)
            try:
                component = comp_class(spec_subtree, *args)  # Pass any extra arguments through to the component.
            except Exception as err:
                print("Error creating component \"{}\": {}".format(comp_class, err))
                raise
            self.instantiated_subtrees.append((name, spec_subtree))
            setattr(component, 'component_name', comp_name)  # Just to support <environment> path resolution.
        else:
            component = None  # Component is not enabled.
        return component

    def get_component_class(self, path):
        # Equivalent to a typical import line.
        path = 'rl_nexus.components.' + path
        try:
            module_path, class_name = path.rsplit('.', 1)
        except ValueError:
            logger.write_line("{} doesn't look like a module path".format(path))
            exit(1)
        # The following line will execute any module-level code in the component's file.
        module = importlib.import_module(module_path)
        try:
            return getattr(module, class_name)
        except AttributeError:
            logger.write_line('Failed to import {}.{}'.format(module_path, class_name))
            exit(1)

    def dump(self, file, indentation):
        # Recursive reporting.
        file.write('{}component: {}\n'.format(indentation, self.full_component_name))
        for setting in self.accessed_settings:
            file.write('{}{}: {}\n'.format(indentation, setting.name, setting.yaml_value()))
        for key, subtree in self.instantiated_subtrees:
            file.write('{}{}:\n'.format(indentation, key))
            subtree.dump(file, indentation + '  ')

    def __getitem__(self, key):
        return self.get_value(key)

    def __setitem__(self, key, value):
        self.settings.__setitem__(key, Setting(self, key, value))

    def __delitem__(self, key):
        self.settings.__delitem__(key)

    def __contains__(self, key):
        return self.settings.__contains__(key)

    def __repr__(self):
        return str(self.settings)


class Setting():
    def __init__(self, spec_tree, name, value):
        self.spec_tree = spec_tree
        self.value_was_read = False
        self.name = name
        if value == 'randint':
            self.value = spec_rand.randint(0, 999999999)
        else:
            self.value = value

    def get_value(self, is_subcomponent=False):
        if not self.value_was_read:
            if not is_subcomponent:
                self.spec_tree.accessed_settings.append(self)
            self.value_was_read = True
        return self.value

    def yaml_value(self):
        if self.value is True:
            return 'true'
        elif self.value is False:
            return 'false'
        elif self.value is None:
            return 'null'
        elif isinstance(self.value, float):
            # Avoid a bug in yaml parsing of scientific notation.
            str = '{}'.format(self.value)
            if (len(str) > 1) and (str[1] == 'e'):
                str = str.replace('e', '.0e')
            return str
        else:
            return self.value

    def __repr__(self):
        return str(self.value)
