import logging
from pathlib import Path
from shutil import copy
import time

import numpy as np
import pandas as pd
import pysaliency
import yaml
from deepgaze_vs_scenewalk.models.gold_standard import HDF5ModelManager, get_gold_standard_uniform_centerbias_model
from deepgaze_vs_scenewalk.models.pseudo_crossvalidated_gold_standard import PseudoCrossvalidatedGoldStandard, interpolate_rbf_linear_squared

from combined_gaze_datasets.evaluation import eval_model
from pysaliency.precomputed_models import get_stimuli_filenames
from pysaliency.utils import get_minimal_unique_filenames

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('h5py').setLevel(logging.INFO)


from tqdm import tqdm

def export_model_to_hdf5_async(model, stimuli, filename, compression=9):
    """Export pysaliency model predictions for stimuli into hdf5 file in a way that
    allows for multiple parallel processes to write to the same file.

    model: Model or SaliencyMapModel
    stimuli: instance of FileStimuli or Stimuli with filenames attribute
    filename: where to save hdf5 file to
    compression: how much to compress the data
    overwrite: if False, an existing file will be appended to and
      if for some stimuli predictions already exist, they will be
      kept.
    flush: whether the hdf5 file should be flushed after each stimulus
    """
    filenames = get_stimuli_filenames(stimuli)
    names = get_minimal_unique_filenames(filenames)

    import h5py

    with h5py.File(filename, mode='a') as f:
        indices = [i for i in range(len(stimuli)) if names[i] not in f]
        # shuffling indices
        np.random.shuffle(indices)
        indices = list(indices)
        logging.debug(f"Skipping {len(stimuli) - len(indices)} already existing entries")


    with tqdm(total=len(stimuli), initial=len(stimuli) - len(indices)) as pbar:
        while indices:
            last_start = time.time()

            with h5py.File(filename, mode='r') as f:
                while indices:
                    k = indices.pop()
                    logging.debug(f"Checking {k} {names[k]}")
                    if names[k] in f:
                        logging.debug(f"Skipping {names[k]}, already done")
                        pbar.update(1)
                        continue
                    else:
                        logging.debug(f"Found unprocessed item after {time.time() - last_start}")
                        break

            logging.debug(f"Processing {k} {names[k]}")

            stimulus = stimuli[k]

            if isinstance(model, pysaliency.SaliencyMapModel):
                smap = model.saliency_map(stimulus)
            elif isinstance(model, pysaliency.Model):
                smap = model.log_density(stimulus)
            else:
                raise TypeError(type(model))

            logging.debug(f"Saving {names[k]}")
            with h5py.File(filename, mode='a') as f:
                f.create_dataset(names[k], data=smap, compression=compression)
                f.flush()

            pbar.update(1)



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

export_model_to_hdf5_async(
    pseudo_gold_model,
    stimuli,
    experiment_directory / 'output' / 'pseudo_crossvalidated_gold_standard.hdf5',
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

