"""Federico's Version"""

from typing import Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pysaliency
from scipy.ndimage import zoom
from tqdm.auto import tqdm

from .jax_model import MAP_SIZE, PRECISION, SceneWalk, dict_to_fixation, none_to_nan


class PysaliencySceneWalk(pysaliency.ScanpathModel):
    def __init__(
        self,
        base_model,
        pixel_per_degree,
        use_original_log_likelihoods=False,
        batch_size=100,
        sw_params=None,
        clip_out_of_stim_fixations=True,
        **sw_hyperparams,
    ):
        """
        base model: model used for saliency/empirical denities
        pixel per degree: scene walk compute in degree of visual angle, this is how the pixel coordinate will be converted
        use_original_log_likelihoods: this wrapper can either use the log likelihoods returned by SceneWalk
            (and shift them to account for the stimulus size), or it can use what is returned as conditional log density,
            (which is rescaled with scipy.ndimage.zoom, so values will be slighly different). If you favor speed, use True,
            if you want to be perfectly consistent with the conditional log densities, use False.
        clip_out_of_stim_fixations: if True, fixations outside the stimulus will be clipped to the stimulus size.
                                    if False, their likelihood will be set to NaN.
        """
        super().__init__()
        self.base_model = base_model
        self.pixel_per_degree = pixel_per_degree
        self.use_original_log_likelihoods = use_original_log_likelihoods
        self.batch_size = batch_size
        self.clip_out_of_stim_fixations = clip_out_of_stim_fixations

        assert sw_hyperparams is not None, "SceneWalk hyperparameters must be provided as keyword arguments"

        self.sw_hyperparams = sw_hyperparams
        self.sw_params = sw_params
        self._current_init_size = (0, 0)
        self.sw: Optional[SceneWalk] = None
        self.temperature_scale = "early_fix_exponents_scaling" in sw_hyperparams

    def setup_model(self, stimulus):
        height, width = pysaliency.datasets.as_stimulus(stimulus).size

        d_range = {
            "x": np.array([0.00000, width / self.pixel_per_degree]),
            "y": np.array([0.00000, height / self.pixel_per_degree]),
        }

        if self.sw_params is not None:
            hyperparameters = self.sw_hyperparams | {
                "data_range": d_range,
                "inputs_in_deg": True,
                "detail_mode": True,
            }
            sw = SceneWalk.from_trained(
                hyperparams=hyperparameters,
                trained_params=self.sw_params,
            )
        else:
            sw = SceneWalk(**self.sw_hyperparams, data_range=d_range, inputs_in_deg=False)

        self._current_init_size = (height, width)
        self.sw = sw

    def saliency_map(self, stimulus):
        log_density = self.base_model.log_density(stimulus)
        saliency_map = np.exp(log_density)

        x_factor = MAP_SIZE / saliency_map.shape[1]
        y_factor = MAP_SIZE / saliency_map.shape[0]

        saliency_map = zoom(saliency_map, [y_factor, x_factor], order=1, mode="nearest")
        saliency_map /= saliency_map.sum()
        saliency_map = jnp.array(saliency_map, dtype=PRECISION)

        return saliency_map

    def run_scanpath(self, stimulus, xs, ys, durations):
        # print("Scanpath", xs, ys, durations)
        height, width = pysaliency.datasets.as_stimulus(stimulus).size

        if self._current_init_size != (height, width):
            del self.sw
            jax.clear_caches()
            self.setup_model(stimulus)

        saliency_map = self.saliency_map(stimulus)

        if not len(xs) == len(ys) == len(durations):
            raise ValueError(
                f"got inconsistent scanpath lengths: len(xs)={len(xs)}, "
                f"len(ys)={len(ys)}, len(durations)={len(durations)}"
            )

        xs = xs / self.pixel_per_degree
        ys = ys / self.pixel_per_degree

        # For proper handling in jax we need to convert any None to jnp.nan
        xs = jnp.array(none_to_nan(xs), dtype=PRECISION)
        ys = jnp.array(none_to_nan(ys), dtype=PRECISION)
        durations = jnp.array(none_to_nan(durations), dtype=PRECISION)

        scanpath = self.sw.get_scanpath_likelihood_detail(xs, ys, durations, saliency_map)  # type: ignore

        # print("scanpath", scanpath)

        if len(scanpath) != len(xs) - 1:
            print(f"wrong scanpath length! Expected {len(xs) - 1}, got {len(scanpath)}")
            raise ValueError
        assert len(scanpath) == len(xs) - 1

        return scanpath

    def get_batch_likelihoods(self, stimuli, xs_list, ys_list, durations_list):
        """
        Process multiple scanpaths in parallel and computes their likelihoods under the model.

        Parameters
        ----------
        stimuli : list
            List of stimuli (must be of same size)
        xs_list : list of arrays
            List of x coordinates for each scanpath
        ys_list : list of arrays
            List of y coordinates for each scanpath
        durations_list : list of arrays
            List of durations for each scanpath

        Returns
        -------
        list:
            List of scanpath evaluation results
        """
        # Verify all stimuli have same size
        height, width = pysaliency.datasets.as_stimulus(stimuli[0]).size
        for stim in stimuli:
            h, w = pysaliency.datasets.as_stimulus(stim).size
            if (h, w) != (height, width):
                raise ValueError("All stimuli must have the same dimensions")

        # Initialize model if needed
        if self._current_init_size != (height, width):
            del self.sw
            jax.clear_caches()
            self.setup_model(stimuli[0])

        # Get saliency maps
        saliency_maps = [self.saliency_map(stim) for stim in stimuli]
        saliency_maps = jnp.stack(saliency_maps)

        # Find max length for padding
        max_length = max(len(xs) for xs in xs_list)

        # Pad scanpaths
        batch_xs = []
        batch_ys = []
        batch_durations = []

        for xs, ys, durations in zip(xs_list, ys_list, durations_list):
            # Convert to degrees
            xs_deg = np.array(xs) / self.pixel_per_degree
            ys_deg = np.array(ys) / self.pixel_per_degree

            # Pad arrays
            padded_xs = jnp.array(none_to_nan(list(xs_deg) + [jnp.nan] * (max_length - len(xs_deg))), dtype=PRECISION)
            padded_ys = jnp.array(none_to_nan(list(ys_deg) + [jnp.nan] * (max_length - len(ys_deg))), dtype=PRECISION)
            padded_durations = jnp.array(
                none_to_nan(list(durations) + [jnp.nan] * (max_length - len(durations))), dtype=PRECISION
            )

            batch_xs.append(padded_xs)
            batch_ys.append(padded_ys)
            batch_durations.append(padded_durations)

        # Stack into arrays
        batch_xs = jnp.stack(batch_xs)
        batch_ys = jnp.stack(batch_ys)
        batch_durations = jnp.stack(batch_durations)

        assert self.sw is not None, "Model not initialized"

        # Evaluate batch
        batch_results = self.sw.get_batch_scanpath_likelihood_detail_core(
            batch_xs, batch_ys, batch_durations, saliency_maps
        )

        # Convert results back to Fixation objects, trim to original lengths
        # and convert to numpy arrays
        results = []
        for i in range(len(batch_results["n"])):
            scanpath_len = min(len(xs_list[i]), len(ys_list[i]), len(durations_list[i])) - 1
            scanpath_results = dict_to_fixation({k: v[i, :scanpath_len] for k, v in batch_results.items()})
            results.append(scanpath_results)

        return results

    def unscaled_attention_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        xs = np.hstack((pysaliency.utils.remove_trailing_nans(x_hist), [0]))
        ys = np.hstack((pysaliency.utils.remove_trailing_nans(y_hist), [0]))

        durations = np.hstack((pysaliency.utils.remove_trailing_nans(attributes["duration_hist"]), [None]))

        scanpath_eval = self.run_scanpath(stimulus, xs, ys, durations)

        return scanpath_eval[-1].att_map

    def attention_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        xs = np.hstack((pysaliency.utils.remove_trailing_nans(x_hist), [0]))
        ys = np.hstack((pysaliency.utils.remove_trailing_nans(y_hist), [0]))

        durations = np.hstack((pysaliency.utils.remove_trailing_nans(attributes["duration_hist"]), [None]))

        scanpath_eval = self.run_scanpath(stimulus, xs, ys, durations)

        height, width = pysaliency.datasets.as_stimulus(stimulus).size

        attention_map = scanpath_eval[-1].att_map

        factor_x = width / attention_map.shape[1]
        factor_y = height / attention_map.shape[0]

        rescaled_density = zoom(attention_map.astype(float), [factor_y, factor_x], order=1, mode="nearest")
        rescaled_density /= np.sum(rescaled_density)

        return rescaled_density

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        xs = np.hstack((pysaliency.utils.remove_trailing_nans(x_hist), [0]))
        ys = np.hstack((pysaliency.utils.remove_trailing_nans(y_hist), [0]))

        durations = np.hstack((pysaliency.utils.remove_trailing_nans(attributes["duration_hist"]), [None]))

        scanpath_eval = self.run_scanpath(stimulus, xs, ys, durations)

        height, width = pysaliency.datasets.as_stimulus(stimulus).size

        density = scanpath_eval[-1].main_map

        factor_x = width / density.shape[1]
        factor_y = height / density.shape[0]

        rescaled_density = zoom(density.astype(float), [factor_y, factor_x], order=1, mode="nearest")
        rescaled_density /= np.sum(rescaled_density)

        return np.log(rescaled_density)

    def conditional_log_densities(self, stimuli, fixations, verbose=False):
        scanpaths, indices = pysaliency.datasets.scanpaths_from_fixations(fixations, verbose=False)
        scanpaths = scanpaths.scanpaths
        print(f"{len(fixations)} -> {len(scanpaths.xs)}")
        conditional_log_densities = []

        for scanpath_index in tqdm(range(len(scanpaths.xs)), disable=False):
            stimulus = stimuli[scanpaths.n[scanpath_index]]
            height, width = stimulus.size
            xs = pysaliency.utils.remove_trailing_nans(scanpaths.xs[scanpath_index])
            ys = pysaliency.utils.remove_trailing_nans(scanpaths.ys[scanpath_index])
            durations = pysaliency.utils.remove_trailing_nans(scanpaths.fixation_attributes["duration"][scanpath_index])
            scanpath_eval = self.run_scanpath(stimulus, xs, ys, durations)

            conditional_log_densities.append(np.full((height, width), fill_value=np.nan))
            factor_x = width / scanpath_eval[0].main_map.shape[1]
            factor_y = height / scanpath_eval[0].main_map.shape[0]

            for i, fixation_eval in enumerate(scanpath_eval):
                density = fixation_eval.main_map
                rescaled_density = zoom(density.astype(float), [factor_y, factor_x], order=1, mode="nearest")
                rescaled_density /= np.sum(rescaled_density)
                log_density = np.log(rescaled_density)
                conditional_log_densities.append(log_density)

        # assert len(conditional_log_densities) == len(scanpaths)

        # can't use fancy indexing for a python list
        return [conditional_log_densities[index] for index in indices]

    def log_likelihoods_serial(self, stimuli, fixations, verbose=False):
        """
        Original serial version of the log_likelihoods function.
        To be used for debugging.
        """

        scanpaths, indices = pysaliency.datasets.scanpaths_from_fixations(fixations, verbose=False)
        scanpaths = scanpaths.scanpaths
        scanpath_likelihoods = []

        for scanpath_index in tqdm(range(len(scanpaths.xs))):
            stimulus = stimuli[scanpaths.n[scanpath_index]]
            xs = pysaliency.utils.remove_trailing_nans(scanpaths.xs[scanpath_index])
            ys = pysaliency.utils.remove_trailing_nans(scanpaths.ys[scanpath_index])
            durations = pysaliency.utils.remove_trailing_nans(scanpaths.fixation_attributes["duration"][scanpath_index])
            scanpath_eval = self.run_scanpath(stimulus, xs, ys, durations)

            # Only spatial component to be comparable to GS and DG
            if self.use_original_log_likelihoods:
                LLs = np.hstack(([np.nan], [fixation_eval.ll_spat for fixation_eval in scanpath_eval]))

                # likelihood are in bit, need to convert to nat
                LLs *= np.log(2)

                # Likelihoods are in bit for 128x128 grid, convert to actual size
                LLs += np.log(128 * 128) - np.log(np.prod(stimulus.size))
            else:
                LLs = [np.nan]

                for i, fixation_eval in enumerate(scanpath_eval):
                    density = np.array(fixation_eval.main_map)
                    rescaled_density = cv2.resize(
                        density.astype(float), (stimulus.size[1], stimulus.size[0]), interpolation=cv2.INTER_LINEAR
                    )
                    rescaled_density /= np.sum(rescaled_density)
                    log_density = np.log(rescaled_density)
                    if 0 <= ys[i + 1] < log_density.shape[0] and 0 <= xs[i + 1] < log_density.shape[1]:
                        # Fixation is within stimulus bounds
                        y_loc = int(ys[i + 1])
                        x_loc = int(xs[i + 1])
                        LLs.append(log_density[y_loc, x_loc])
                    elif self.clip_out_of_stim_fixations:
                        # Clip out-of-bounds fixations to the stimulus edges
                        y_loc = int(np.clip(ys[i + 1], 0, log_density.shape[0] - 1))
                        x_loc = int(np.clip(xs[i + 1], 0, log_density.shape[1] - 1))
                        LLs.append(log_density[y_loc, x_loc])
                    else:
                        # Out-of-bounds fixation and no clipping
                        LLs.append(np.nan)
                LLs = np.array(LLs)
                assert len(LLs) == len(xs)

            scanpath_likelihoods.append(LLs)

        scanpath_likelihoods = np.hstack(scanpath_likelihoods)
        return scanpath_likelihoods[indices]

    def get_batch_likelihoods_rescaled_map(self, batch_stimuli, batch_xs, batch_ys, batch_durations):
        """
        Compute log likelihoods for a batch of scanpaths, getting them from rescaled model
        density maps at each fixation's location.

        Parameters
        ----------
        batch_stimuli : list
            List of stimuli for each scanpath
        batch_xs : list of arrays
            List of x coordinates for each scanpath
        batch_ys : list of arrays
            List of y coordinates for each scanpath
        batch_durations : list of arrays
            List of durations for each scanpath

        Returns
        -------
        list:
            List of log likelihood arrays for each scanpath
        """
        # Run batch scanpaths to get detailed outputs
        batch_scanpath_results = self.get_batch_likelihoods(batch_stimuli, batch_xs, batch_ys, batch_durations)

        batch_log_likelihoods = []

        # Process results for each scanpath
        for i, scanpath_eval in enumerate(batch_scanpath_results):
            stimulus = batch_stimuli[i]
            xs = batch_xs[i]
            ys = batch_ys[i]

            # Compute log likelihoods from density maps
            LLs = [np.nan]

            for k, fixation_eval in enumerate(scanpath_eval):
                density = np.array(fixation_eval.main_map)
                rescaled_density = cv2.resize(
                    density.astype(float), (stimulus.size[1], stimulus.size[0]), interpolation=cv2.INTER_LINEAR
                )
                rescaled_density /= np.sum(rescaled_density)
                log_density = np.log(rescaled_density)
                if 0 <= ys[k + 1] < log_density.shape[0] and 0 <= xs[k + 1] < log_density.shape[1]:
                    # Fixation is within stimulus bounds
                    y_loc = int(ys[k + 1])
                    x_loc = int(xs[k + 1])
                    LLs.append(log_density[y_loc, x_loc])
                elif self.clip_out_of_stim_fixations:
                    # Clip out-of-bounds fixations to the stimulus edges
                    y_loc = int(np.clip(ys[k + 1], 0, log_density.shape[0] - 1))
                    x_loc = int(np.clip(xs[k + 1], 0, log_density.shape[1] - 1))
                    LLs.append(log_density[y_loc, x_loc])
                else:
                    # Out-of-bounds fixation and no clipping
                    LLs.append(np.nan)

            batch_log_likelihoods.append(np.array(LLs))

        return batch_log_likelihoods

    def log_likelihoods(self, stimuli, fixations, verbose=False):
        """
        Compute log likelihoods for fixations with batch processing.

        Parameters
        ----------
        stimuli : list
            List of stimuli
        fixations : Fixations
            Fixations to compute log likelihoods for
        verbose : bool
            Whether to show progress bars

        Returns
        -------
        array:
            Log likelihoods for each fixation
        """
        scanpaths, indices = pysaliency.datasets.scanpaths_from_fixations(fixations, verbose=False)
        scanpaths = scanpaths.scanpaths

        # Group scanpaths by stimulus size
        size_to_scanpaths = {}
        for i in range(len(scanpaths.xs)):
            stimulus = stimuli[scanpaths.n[i]]
            size = stimulus.size
            if size not in size_to_scanpaths:
                size_to_scanpaths[size] = []
            size_to_scanpaths[size].append((i, stimulus, scanpaths.n[i]))

        all_scanpath_likelihoods = {}

        # Process each group of same-sized stimuli
        for size, scanpath_group in tqdm(
            size_to_scanpaths.items(),
            desc="Processing stimulus sizes",
            disable=(not verbose) or len(size_to_scanpaths) == 1,
            position=0,
        ):
            # Process in batches
            batch_size = self.batch_size
            for batch_start in tqdm(
                range(0, len(scanpath_group), batch_size),
                desc=f"Processing batches for size {size}",
                disable=not verbose,
                position=1,
            ):
                batch_end = min(batch_start + batch_size, len(scanpath_group))
                batch = scanpath_group[batch_start:batch_end]

                # Extract data
                batch_indices = [idx for idx, _, _ in batch]
                batch_stimuli = [stim for _, stim, _ in batch]

                batch_xs = [pysaliency.utils.remove_trailing_nans(scanpaths.xs[i]) for i in batch_indices]
                batch_ys = [pysaliency.utils.remove_trailing_nans(scanpaths.ys[i]) for i in batch_indices]
                batch_durations = [
                    pysaliency.utils.remove_trailing_nans(scanpaths.fixation_attributes["duration"][i])
                    for i in batch_indices
                ]

                if self.use_original_log_likelihoods:
                    # Use batch processing and fast path
                    batch_scanpath_results = self.get_batch_likelihoods(
                        batch_stimuli, batch_xs, batch_ys, batch_durations
                    )

                    for i, idx in enumerate(batch_indices):
                        scanpath_eval = batch_scanpath_results[i]
                        LLs = np.hstack(([np.nan], [fixation_eval.ll_spat for fixation_eval in scanpath_eval]))

                        # Convert from bits to nats and adjust for grid size
                        LLs *= np.log(2)
                        LLs += np.log(128 * 128) - np.log(np.prod(batch_stimuli[i].size))
                        all_scanpath_likelihoods[idx] = LLs
                else:
                    # Use batch processing for all scanpaths in the batch
                    batch_lls = self.get_batch_likelihoods_rescaled_map(
                        batch_stimuli, batch_xs, batch_ys, batch_durations
                    )

                    # Store results in the all_scanpath_likelihoods dictionary
                    for i, idx in enumerate(batch_indices):
                        all_scanpath_likelihoods[idx] = batch_lls[i]

        # Combine results in the original order
        scanpath_likelihoods = [all_scanpath_likelihoods[i] for i in range(len(scanpaths.xs))]
        scanpath_likelihoods = np.hstack(scanpath_likelihoods)

        return scanpath_likelihoods[indices]


class PysaliencySceneWalkSchuett(PysaliencySceneWalk):
    def __init__(
        self,
        base_model,
        pixel_per_degree,
        batch_size=100,
        use_original_log_likelihoods=False,
        sw_params=None,
        **sw_hyperparams,
    ):
        _sw_params = {
            "omegaAttention": 2.4e30,
            "omegaInhib": 1.97,
            "sigmaAttention": 5.9,
            "sigmaInhib": 4.5,
            "gamma": 44.780,
            "lamb": 0.8115,
            "inhibStrength": 0.3637,
            "zeta": 0.0722,
        }

        if sw_params:
            _sw_params.update(sw_params)

        super().__init__(
            base_model=base_model,
            pixel_per_degree=pixel_per_degree,
            batch_size=batch_size,
            use_original_log_likelihoods=use_original_log_likelihoods,
            inhib_method="subtractive",
            att_map_init_type="zero",
            shifts="off",
            locdep_decay_switch="off",
            omp="off",
            dynamic_duration=False,
            sw_params=_sw_params,
            **sw_hyperparams,
        )
