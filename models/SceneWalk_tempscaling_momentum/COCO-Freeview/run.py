import json
import os
import pickle
from pathlib import Path

import hydra
import jax
import numpy as np
import pandas as pd

# jax.config.update('jax_platform_name', 'cpu')
import pysaliency
from deepgaze_vs_scenewalk.scenewalk_jax.jax_model import (
    ParameterPrior,
)
from deepgaze_vs_scenewalk.scenewalk_jax.jax_training import compute_density_maps, jax_has_gpu
from deepgaze_vs_scenewalk.scenewalk_jax.pysaliency_wrapper import PysaliencySceneWalk
from deepgaze_vs_scenewalk.scenewalk_jax.training_wrappers import train_from_config
from omegaconf import DictConfig, OmegaConf

jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    assert jax_has_gpu(), "No GPU found, exiting."

    print(OmegaConf.to_yaml(cfg))

    # Import fixations and density model for validation
    val_stimuli = pysaliency.FileStimuli.read_hdf5("pysaliency_datasets/COCO-Freeview/stimuli_validation.hdf5")
    val_fixations = pysaliency.FixationTrains.read_hdf5("pysaliency_datasets/COCO-Freeview/fixations_validation.hdf5")

    # val_stimuli, val_fixations = pysaliency.create_subset(val_stimuli, val_fixations, list(range(5)))

    val_empirical_density_model = pysaliency.HDF5Model(
        stimuli=val_stimuli,
        filename="pseudo_crossvalidated_gold_standard_validation.hdf5",
        caching=True,
    )

    densities_val = compute_density_maps(val_empirical_density_model, val_fixations, val_stimuli)

    # Import fixations and density model for training
    stimuli = pysaliency.FileStimuli.read_hdf5("pysaliency_datasets/COCO-Freeview/stimuli_train.hdf5")
    all_fixations = pysaliency.FixationTrains.read_hdf5("pysaliency_datasets/COCO-Freeview/fixations_train.hdf5")

    # stimuli, all_fixations = pysaliency.create_subset(stimuli, all_fixations, list(range(5)))

    empirical_density_model = pysaliency.HDF5Model(
        stimuli=stimuli,
        filename="pseudo_crossvalidated_gold_standard_train.hdf5",
        caching=True,
    )

    densities = compute_density_maps(empirical_density_model, all_fixations, stimuli)

    trained_params, loss, sw_jax = train_from_config(
        cfg,
        fixations=all_fixations,
        stimuli=stimuli,
        densities=densities,
        val_fixations=val_fixations,
        val_stimuli=val_stimuli,
        val_densities=densities_val,
    )

    # Save the trained model to json
    output_dir = Path("output")
    with (output_dir / "model_params.json").open("w") as f:
        json.dump(trained_params, f, indent=4)

    # Save the model parameters to pickle
    with (output_dir / "model_params.pkl").open("wb") as f:
        pickle.dump(trained_params, f)

    #####################################
    # Evaluation
    #####################################

    # Convert the priors to a dictionary (from omegaconf object)
    priors = hydra.utils.call(cfg.priors)

    # Instantiate priors.
    priors = {param: ParameterPrior(*priors[param]) for param in priors}

    ## Validation
    sw_wrapped_validation = PysaliencySceneWalk(
        base_model=val_empirical_density_model,
        pixel_per_degree=cfg.pixel_per_degree,
        use_original_log_likelihoods=False,
        batch_size=100,
        sw_params=trained_params,
        **hydra.utils.call(cfg.model) | {"parameter_priors": priors},
    )
    fixations_subset_eval = val_fixations[val_fixations.scanpath_history_length >= 1]
    log_likelihoods_validation = sw_wrapped_validation.log_likelihoods(val_stimuli, fixations_subset_eval, verbose=True)
    igs_validation = (
        log_likelihoods_validation - pysaliency.UniformModel().log_likelihoods(val_stimuli, fixations_subset_eval)
    ) / np.log(2)

    print(f"Average log likelihood (validation): {np.mean(log_likelihoods_validation)}")

    # Save the log likelihoods to a .csv file
    ll_df = pd.DataFrame(
        {
            "LL": log_likelihoods_validation,
            "IG": igs_validation,
        }
    )
    os.makedirs(output_dir / "validation", exist_ok=True)
    ll_df.to_csv(output_dir / "validation" / "results_per_fixation.csv.gz", index=False)

    # Save average results to a .csv file
    avg_ll_df_val = pd.DataFrame(
        {
            "Average LL": [np.mean(log_likelihoods_validation)],
            "Average IG": [np.mean(igs_validation)],
        }
    )
    avg_ll_df_val.to_csv(output_dir / "validation" / "results.csv", index=False)

    # Train
    sw_wrapped_train = PysaliencySceneWalk(
        base_model=empirical_density_model,  # Use the training density model
        pixel_per_degree=cfg.pixel_per_degree,
        use_original_log_likelihoods=False,
        batch_size=100,
        sw_params=trained_params,
        **hydra.utils.call(cfg.model) | {"parameter_priors": priors},
    )
    fixations_subset = all_fixations[all_fixations.scanpath_history_length >= 1]
    log_likelihoods_train = sw_wrapped_train.log_likelihoods(stimuli, fixations_subset, verbose=True)
    igs_train = (log_likelihoods_train - pysaliency.UniformModel().log_likelihoods(stimuli, fixations_subset)) / np.log(
        2
    )
    print(f"Average log likelihood (train): {np.mean(log_likelihoods_train)}")
    ll_df = pd.DataFrame(
        {
            "LL": log_likelihoods_train,
            "IG": igs_train,
        }
    )
    os.makedirs(output_dir / "train", exist_ok=True)
    ll_df.to_csv(output_dir / "train" / "results_per_fixation.csv.gz", index=False)

    # Save average results to a .csv file
    avg_ll_df = pd.DataFrame(
        {
            "Average LL": [np.mean(log_likelihoods_train)],
            "Average IG": [np.mean(igs_train)],
        }
    )
    avg_ll_df.to_csv(output_dir / "train" / "results.csv", index=False)


if __name__ == "__main__":
    main()
