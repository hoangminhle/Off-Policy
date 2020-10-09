# yaml_utils.py: Miscellaneous functions for working with yaml components
import re
import os
import oyaml as yaml
from os.path import join as pjoin
from collections import OrderedDict

rl_nexus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve(val, string_replacements):
    """ Performs string replacements to val. """
    for query, replacement in string_replacements.items():
        if not isinstance(val, str):
            return val
        if query == val:
            val = replacement
        elif query in val:
            val = val.replace(query, str(replacement))
    match = re.search('<\w+>', str(val))
    if match:
        raise ValueError("Unresolved value {} in yaml string \'{}\'.".format(match.group(0), val))
    return val

def get_comp_spec(component_path):
    """ Returns the list of default values in a Component's yaml. """
    comp_name = component_path.split('/')[-1]
    yaml_path = pjoin(rl_nexus_dir, 'components', component_path)
    if os.path.isdir(yaml_path): # Components except Root require us to dig an extra level deeper.
        yaml_path = pjoin(yaml_path, comp_name)
    yaml_path += '.yaml'
    component_spec = yaml.load(stream=open(yaml_path, 'r'), Loader=yaml.Loader)
    return component_spec

def expand_item(val, string_replacements):
    """ Expands a particular yaml value. """
    if isinstance(val, dict):
        return expand_dict(val, string_replacements)
    elif isinstance(val, list):
        return [expand_item(subval, string_replacements) for subval in val]
    else:
        return resolve(val, string_replacements)

def expand_dict(spec, string_replacements):
    """ Expands a yaml dictionary, incorporating subcomponents where necessary. """
    expanded_spec = OrderedDict()
    for k, v in spec.items():
        if k == 'component':
            v_res = resolve(v, string_replacements)
            expanded_spec[k] = v_res
            comp_spec = get_comp_spec(v_res)
            for key in spec:
                if key == 'hyperparameters' or key == 'string_replacements' or key == 'forward_definitions':
                    continue
                if key not in comp_spec:
                    raise AttributeError("{}.yaml contains no \'{}\' property.".format(v_res, key))
            for default_key, default_val in comp_spec.items():
                if default_key not in spec:
                    expanded_spec[default_key] = expand_item(default_val, string_replacements)
        else:
            expanded_spec[k] = expand_item(v, string_replacements)
    return expanded_spec
