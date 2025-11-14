import logging
from pathlib import Path

import numpy as np
import pysaliency
import yaml
from deepgaze_vs_scenewalk.models.gold_standard import fit_gold_standard
from scipy.special import logsumexp

logging.basicConfig(level=logging.INFO)



class StimuliMappingModel(pysaliency.Model):
    """model which maps stimuli from one dataset to another before predicting with a parent model"""
    def __init__(self, parent_model, external_stimuli, internal_stimuli, verbose=False, **kwargs):
        super().__init__(**kwargs)

        self.parent_model = parent_model

        if not len(external_stimuli) == len(internal_stimuli):
            raise ValueError('external_stimuli and internal_stimuli must have the same length')

        self.external_stimuli = external_stimuli
        self.internal_stimuli = internal_stimuli
        self.verbose = verbose

    def _map_stimulus(self, stimulus):
        external_id = pysaliency.datasets.as_stimulus(stimulus).stimulus_id
        index = self.external_stimuli.stimulus_ids.index(external_id)
        internal_stimulus = self.internal_stimuli[index]

        return internal_stimulus

    def _log_density(self, stimulus, attributes=None, out=None):
        internal_stimulus = self._map_stimulus(stimulus)
        prediction = self.parent_model.log_density(internal_stimulus)
        return prediction


class RenormalizingModel(pysaliency.Model):
    def __init__(self, parent_model, threshold_warning=1e-7, threshold_error=1e-3, dtype=np.float64, **kwargs):
        """Makes sure that log densities are in high precision and properly normalized

        This wrapper model is intended to be applied after e.g. deeplearning models with float32 precision or models
        which might suffer from numerical instabilities.
        """
        super().__init__(**kwargs)

        self.parent_model = parent_model
        self.threshold_warning = threshold_warning
        self.threshold_error = threshold_error
        self.dtype = dtype

    def _log_density(self, stimulus, attributes=None, out=None):
        prediction = self.parent_model.log_density(stimulus).astype(self.dtype)

        norm = logsumexp(prediction)
        if np.abs(norm) > self.threshold_error:
            raise ValueError(f'prediction norm is too large: {norm}')
        if np.abs(norm) > self.threshold_warning:
            logging.warning(f'prediction norm is too large: {norm} (applying renormalization)')

        return prediction - norm



experiment_directory = Path(__file__).parent

config_file = experiment_directory / 'config.yaml'

config = yaml.safe_load(open(config_file))


### prepare spatial DeepGaze model

logging.info('Preparing spatial DeepGaze model')

stimuli = pysaliency.read_hdf5(config['dataset']['stimuli'])

full_stimuli = pysaliency.read_hdf5('combined_dataset/Combined_dataset/combined_stimuli.hdf5')
full_stimuli.cached = False

mit1003_twosize_stimuli = full_stimuli[full_stimuli.attributes['dataset'] == 0]

spatial_deepgaze_model = RenormalizingModel(pysaliency.ResizingModel(
    StimuliMappingModel(
        pysaliency.HDF5Model(
            full_stimuli,
            'combined_dataset_spatial_deepgaze.hdf5',
        ),
        external_stimuli=stimuli,
        internal_stimuli=mit1003_twosize_stimuli,
    ),
    verbose=False,
))

pysaliency.export_model_to_hdf5(
    spatial_deepgaze_model,
    stimuli,
    'output/spatial_deepgaze.hdf5',
    overwrite=False,
)

### fit gold standard

logging.info('Fitting gold standard')

fit_gold_standard(
    config=config,
    output_directory=experiment_directory / 'output',
)