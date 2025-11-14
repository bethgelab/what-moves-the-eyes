import logging
from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
import pysaliency
import yaml
from deepgaze_vs_scenewalk.models.gold_standard import HDF5ModelManager, get_gold_standard_uniform_centerbias_model
from deepgaze_vs_scenewalk.models.pseudo_crossvalidated_gold_standard import PseudoCrossvalidatedGoldStandard, interpolate_rbf_linear_squared

from combined_gaze_datasets.evaluation import eval_model

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('h5py').setLevel(logging.INFO)


experiment_directory = Path(__file__).parent

goldstandard_config_file = experiment_directory / 'goldstandard_model' / 'config.yaml'

gold_standard_config = yaml.safe_load(open(goldstandard_config_file))

logging.info(f"gold_standard_config: {gold_standard_config}")
logging.info("monkey patching gold_standard_config")

# monkey patch the dependencies
for regularization in gold_standard_config['regularizations']:
    if 'model_file' in regularization:
        regularization['model_file'] = str(experiment_directory / 'goldstandard_model' / regularization['model_file'])

gold_standard_config['dataset']['stimuli'] = str(experiment_directory / 'goldstandard_model' / gold_standard_config['dataset']['stimuli'])
gold_standard_config['dataset']['fixations'] = str(experiment_directory / 'goldstandard_model' / gold_standard_config['dataset']['fixations'])

logging.info("loading data")

stimuli, fixations = pysaliency.load_dataset_from_config(gold_standard_config['dataset'])
stimuli.cached = False

logging.info("Preloading regularization models")

hdf5_model_manager = HDF5ModelManager()

# load full hdf5 models for later reuse
for regularization in gold_standard_config['regularizations']:
    model_type = regularization.get('type', 'hdf5')
    if model_type == 'hdf5':
        logging.info(f"caching {regularization}")
        hdf5_model_manager.get_model(stimuli, regularization['model_file'])
        logging.debug("done")

logging.info("Building gold standard model")

df_parameters = pd.read_csv(experiment_directory / 'goldstandard_model' / 'output' / 'parameters.csv', index_col=0)

subject_models, crossval_model, upper_model = get_gold_standard_uniform_centerbias_model(
    stimuli,
    fixations,
    gold_standard_config,
    df_parameters,
    hdf5_model_manager=hdf5_model_manager,
    grid_spacing=1
)

logging.info("building subject aggregating gold standard model")

pseudo_gold_model = PseudoCrossvalidatedGoldStandard(
    stimuli,
    fixations,
    subject_models,
    interpolate=interpolate_rbf_linear_squared,
    caching=False,
)

logging.info("Exporting gold standard model")

pysaliency.export_model_to_hdf5(
    pseudo_gold_model,
    stimuli,
    experiment_directory / 'output' / 'pseudo_crossvalidated_gold_standard.hdf5',
    overwrite=False,
    flush=True,
)

logging.info("evaluating pseudo gold standard model")

centerbias_model = pysaliency.HDF5Model(
    stimuli,
    experiment_directory / 'goldstandard_model' / 'centerbias.hdf5',
    memory_cache_size=10,
)

pseudo_gold_model = pysaliency.HDF5Model(
    stimuli,
    experiment_directory / 'output' / 'pseudo_crossvalidated_gold_standard.hdf5',
    memory_cache_size=10,
)

eval_model(
    pseudo_gold_model,
    stimuli,
    fixations,
    baseline_model=centerbias_model,
    pickle_path=experiment_directory / 'output' / 'evaluation.pkl',
    csv_path=experiment_directory / 'output' / 'evaluation.csv',
)

