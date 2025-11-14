import importlib
import os
import pathlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pysaliency
import yaml
from pysaliency.baseline_utils import (
    CrossvalidatedBaselineModel,
    CrossvalMultipleRegularizations,
    ScikitLearnImageCrossValidationGenerator,
)
from scipy.optimize import minimize

experiment_directory = Path(__file__).parent

config_file = experiment_directory / 'config.yaml'

config = yaml.safe_load(open(config_file))

dataset_config = config.pop('dataset')

log_bandwidth_min = config.pop('log_bandwidth_min', -3)
log_bandwidth_max = config.pop('log_bandwidth_max', 0)

log_regularization_uniform_min = config.pop('log_regularization_uniform_min', -20)
log_regularization_uniform_max = config.pop('log_regularization_uniform_max', 0)

output_directory = experiment_directory / 'output'
output_directory.mkdir(exist_ok=True)

results_file = output_directory / 'results.yaml'

stimuli, fixations = pysaliency.load_dataset_from_config(dataset_config)


def save_results(results):
    with open(results_file, 'w') as f:
        yaml.safe_dump(results, f, default_flow_style=False)


def load_results():
    if os.path.isfile(results_file):
        return yaml.safe_load(open(results_file))
    else:
        return {}


def load_model(stimuli, fixations, config):
    model_type = config.get('type', 'hdf5')
    if model_type == 'uniform':
        model = pysaliency.UniformModel()
    elif model_type == 'hdf5':
        model = pysaliency.HDF5Model(stimuli, config['model_file'], caching=False)
    else:
        raise ValueError('Invalid model type', model_type)

    return model


def import_from_string(name):
    if '.' in name:
        module_name, class_name = name.rsplit('.', 1)

        module = importlib.import_module(module_name)
        klazz = getattr(module, class_name)

        return klazz
    else:
        globals()[name]


def load_regularization_models(stimuli, fixations, config):
    regularization_models = OrderedDict()
    for model_data in config['regularizations']:
        print("Loading regularization model")
        print(model_data)
        model = load_model(stimuli, fixations, model_data)
        regularization_models[model_data['name']] = model

    return regularization_models


within_stimulus_attributes = config.get('within_stimulus_attributes')
maximal_source_count = config.get('maximal_source_count')

print("Using maximal source count", maximal_source_count)

crossvalidation = ScikitLearnImageCrossValidationGenerator(
    stimuli, fixations, within_stimulus_attributes=within_stimulus_attributes,
    leave_out_size=1,
    maximal_source_count=maximal_source_count,
)

regularization_models = load_regularization_models(stimuli, fixations, config)
manager = CrossvalMultipleRegularizations(stimuli, fixations, regularization_models, crossvalidation)
selected_cross_val_score = manager.score




params = [('log_bandwidth', (log_bandwidth_min, log_bandwidth_max))]
for model_data in config['regularizations']:
    reg_min = model_data.get('log_weight_min', -10)
    reg_max = model_data.get('log_weight_max', 0)
    params.append(('log_{}'.format(model_data['name']), (reg_min, reg_max)))
params = OrderedDict(params)


print(params)
print(list(params.keys()))


previous_results = load_results()
if all(key in previous_results.get('parameters', {}) for key in params) or all(key in previous_results for key in params):
    already_optimized = True
else:
    already_optimized = False

# Random search

random_search_config = config['random_search']

columns = list(params.keys()) + ['score']
random_search_data = pd.DataFrame(columns=columns, dtype=np.float64)

log_filename = pathlib.Path(output_directory) / 'random_search.csv'

if log_filename.is_file():
    print("loading data")
    random_search_data = pd.read_csv(log_filename, index_col=0)

print(random_search_data)

while len(random_search_data) < random_search_config['steps']:
    this_params = {}
    for param_name, param_bounds in params.items():
        this_params[param_name] = np.random.uniform(low=param_bounds[0], high=param_bounds[1])

    #print("next:", this_params)

    score = selected_cross_val_score(**this_params)

    new_row = pd.Series(this_params)
    new_row['score'] = score

    random_search_data = pd.concat((random_search_data, pd.DataFrame([new_row])), ignore_index=True)
    #print(random_search_data.dtypes)
    random_search_data['best_score'] = random_search_data['score'].cummax()

    print(random_search_data.tail())
    print("best:")
    print(random_search_data.loc[random_search_data['score'].idxmax()])

    random_search_data.to_csv(log_filename)



def selected_crossval_cost(x):
    """ scipy.optimize-compatible version of the score """
    print('.', end='', flush=True)
    return -selected_cross_val_score(*x)



if not already_optimized:

    best_row = random_search_data.loc[random_search_data['score'].idxmax()]
    print("best row", best_row)


    #best_params = bo.optimizer.max['params']
    x0 = np.array([best_row[key] for key in params])
    #x0 = bo.optimizer.X[bo.optimizer.Y.argmax()]
    bounds = list(params.values())
    res = minimize(selected_crossval_cost, x0, options={'disp': 10, 'iprint': 1},
                   bounds=bounds)

    #print(x0, bo.optimizer.Y.max())
    print(res)


    params_optim = {key: float(res.x[i]) for i, key in enumerate(params)}

    output = {'parameters':  params_optim}
    output['score'] = -float(res.fun)
    save_results(output)

else:
    print("Already optimized")
    params_optim = {key: previous_results['parameters'][key] for key in params}
    output = previous_results


def eval_model(model):
    performance =  (model.log_likelihood(stimuli, fixations, average='image', verbose=True)
                    - pysaliency.UniformModel().log_likelihood(stimuli, fixations, average='image')) / np.log(2)
    return float(performance)



stimuli.cached = False

assert len(config['regularizations']) == 1

def _get_model(stimuli, fixations):
    model = CrossvalidatedBaselineModel(stimuli, fixations,
                                        bandwidth=10**params_optim['log_bandwidth'],
                                        eps=10**params_optim['log_uniform'], caching=False)
    return model

if not config.get('within_stimulus_attributes'):
    model = _get_model(stimuli, fixations)

else:
    attribute_name, = config['within_stimulus_attributes']
    attribute_values = sorted(np.unique(stimuli.attributes[attribute_name]))

    sub_models = {}
    for attribute_value in attribute_values:
        indices = list(np.nonzero(stimuli.attributes[attribute_name] == attribute_value)[0])
        sub_stimuli, sub_fixations = pysaliency.create_subset(stimuli, fixations, stimuli_indices=indices)
        #print("SUB", sub_stimuli.cached)
        sub_model = _get_model(sub_stimuli, sub_fixations)
        sub_models[sub_stimuli] = sub_model
    model = pysaliency.StimulusDependentModel(sub_models, caching=False)

if config.get('export_hdf5', True):
    pysaliency.export_model_to_hdf5(model, stimuli, os.path.join(output_directory, 'centerbias.hdf5'))

output['log_likelihood'] = eval_model(model)
save_results(output)