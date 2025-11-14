import logging
from pathlib import Path
from shutil import copy

import pysaliency


logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('h5py').setLevel(logging.INFO)

stimuli_train = pysaliency.read_hdf5('pysaliency_datasets/COCO-Freeview/stimuli_train.hdf5')
stimuli_train.cached = False

stimuli_validation = pysaliency.read_hdf5('pysaliency_datasets/COCO-Freeview/stimuli_validation.hdf5')
stimuli_validation.cached = False

stimuli_all = pysaliency.datasets.concatenate_stimuli([stimuli_train, stimuli_validation])
stimuli_all.to_hdf5('stimuli_all.hdf5')

# can't save directly to output because it would screw up the relative filenames
copy('stimuli_all.hdf5', 'output/stimuli_all.hdf5')

model_train = pysaliency.HDF5Model(
    stimuli=stimuli_train,
    filename='pseudo_crossvalidated_gold_standard_train.hdf5',
    caching=False
)

model_validation = pysaliency.HDF5Model(
    stimuli=stimuli_validation,
    filename='pseudo_crossvalidated_gold_standard_validation.hdf5',
    caching=False
)

model_all = pysaliency.StimulusDependentModel({
    stimuli_train: model_train,
    stimuli_validation: model_validation
}, caching=False)

pysaliency.export_model_to_hdf5(
    model=model_all,
    stimuli=stimuli_all,
    filename='output/pseudo_crossvalidated_gold_standard.hdf5',
)
