import os
from pathlib import Path

import numpy as np
import pysaliency
import yaml
from pysaliency.baseline_utils import BaselineModel

root_directory = Path(os.path.dirname(os.path.realpath(__file__)))
config = yaml.safe_load((root_directory / 'config.yaml').open())

output_directory = root_directory / 'output'
output_directory.mkdir(exist_ok=True)

source_path = Path(config['source_directory'])
source_config = yaml.safe_load((source_path / 'config.yaml').open())

dataset_config = config['dataset']

source_stimuli, source_fixations = pysaliency.load_dataset_from_config(source_config['dataset'])
target_stimuli, target_fixations = pysaliency.load_dataset_from_config(config['dataset'])

source_results = yaml.safe_load((source_path / 'output' / 'results.yaml').open())

source_stimuli.cached = False
target_stimuli.cached = False

assert len(source_config['regularizations']) == 1
params_optim = source_results['parameters']

def _get_model(stimuli, fixations):
    model = BaselineModel(
        source_stimuli, source_fixations,
        bandwidth=10**params_optim['log_bandwidth'],
        eps=10**params_optim['log_uniform'], caching=False)
    return model

assert source_config['model_type'] == 'centerbias'
assert len(source_config['regularizations']) == 1

if not source_config.get('within_stimulus_attributes'):
    print("bulding joint model")
    model = _get_model(source_stimuli, source_fixations)

else:
    attribute_name, = source_config['within_stimulus_attributes']
    attribute_values = sorted(np.unique(target_stimuli.attributes[attribute_name]))
    print("building attribute model for", attribute_name)

    sub_models = {}
    for attribute_value in attribute_values:
        source_indices = list(np.nonzero(source_stimuli.attributes[attribute_name] == attribute_value)[0])
        source_sub_stimuli, source_sub_fixations = pysaliency.create_subset(source_stimuli, source_fixations, stimuli_indices=source_indices)
        sub_model = _get_model(source_sub_stimuli, source_sub_fixations)

        target_indices = list(np.nonzero(target_stimuli.attributes[attribute_name] == attribute_value)[0])
        target_sub_stimuli, _ = pysaliency.create_subset(target_stimuli, target_fixations, stimuli_indices=target_indices)

        sub_models[target_sub_stimuli] = sub_model
    model = pysaliency.StimulusDependentModel(sub_models, caching=False)

print(model.information_gain(target_stimuli, target_fixations, verbose=True, average='image'))

pysaliency.export_model_to_hdf5(model, target_stimuli, output_directory / 'centerbias.hdf5')