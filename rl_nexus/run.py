import argparse
import io, os, sys
import oyaml as yaml
from rl_nexus.components.root.Root_Component import Root_Component
from rl_nexus.utils.spec_tree import SpecTree
from rl_nexus.utils.hp_handler import HyperparameterHandler
from rl_nexus.utils.yaml_utils import expand_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_spec_path", default="spec.yaml", help="Runspec yaml file to execute.")
    parsed, unknown = parser.parse_known_args()
    # Parse any dynamically added args into string replacements
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Let the HP handler pre-process the run spec.
    hp_handler = HyperparameterHandler()
    spec_str = hp_handler.preprocess(open(args.run_spec_path, 'r'))

    # Load the run spec yaml.
    spec = yaml.load(stream=io.StringIO(spec_str), Loader=yaml.Loader)

    # Extract string replacements from spec
    string_replacements = spec["string_replacements"] if "string_replacements" in spec else {}

    # Try to get XT env variable for the mounted datastore path in remote runs
    datastore_path = os.getenv("XT_DATA_DIR")
    if datastore_path:
        datastore_path = datastore_path.rstrip('/')
    else:
        datastore_path = '../datastore'
    string_replacements['$datastore'] = datastore_path

    # Update the string replacements with any command-line arguments provided
    for key, value in vars(args).items():
        if key == "run_spec_path":
            continue
        print('Overloading string_replacement from command line: <{}>: {}'.format(key, value))
        string_replacements['<{}>'.format(key)] = value

    # Pull in all the defaults to create an expanded spec, performing string replacements as needed
    expanded_spec = expand_dict(spec, string_replacements)

    # For debugging...
    # yaml.dump(data=spec, stream=open('out.yaml', 'w'), Dumper=yaml.Dumper, indent=2)
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(spec)
    # exit(0)

    # Instantiate the SpecTree and Root.
    spec_tree = SpecTree(expanded_spec)
    root = Root_Component(spec_tree, args.run_spec_path, hp_handler)

    # Was a job launcher instantiated?
    if root.job_launcher:
        # Yes. Launch the run spec as a job.
        root.launch_job(args.run_spec_path)
    else:
        # No. Execute the run spec directly.
        root.execute_processing_stages()
