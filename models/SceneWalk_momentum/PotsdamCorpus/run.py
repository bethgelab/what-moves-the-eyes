import json
import pickle
from pathlib import Path

import hydra
import jax
import numpy as np
import pandas as pd
import pysaliency
from deepgaze_vs_scenewalk.scenewalk_jax.jax_model import (
    ParameterPrior,
)
from deepgaze_vs_scenewalk.scenewalk_jax.jax_training import (
    compute_density_maps,
)
from deepgaze_vs_scenewalk.scenewalk_jax.pysaliency_wrapper import PysaliencySceneWalk
from deepgaze_vs_scenewalk.scenewalk_jax.training_wrappers import train_from_config
from deepgaze_vs_scenewalk.scenewalk_jax.utils.crossval_utils import (
    get_completed_folds,
    iterate_crossvalidation_config,
    load_existing_fold_results,
    pool_crossvalidation_results,
)
from omegaconf import DictConfig, OmegaConf

jax.config.update("jax_enable_x64", True)


def train_and_evaluate_fold(cfg, fold_data, empirical_density_model):
    """Train and evaluate a single fold."""
    fold_output_dir = Path(fold_data["directory"])
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training fold {fold_data['fold_no']}")

    # Compute densities for training data
    train_densities = compute_density_maps(
        empirical_density_model, fold_data["train_fixations"], fold_data["train_stimuli"]
    )

    # Train the model on training data
    trained_params, loss, sw_jax = train_from_config(
        cfg, fixations=fold_data["train_fixations"], stimuli=fold_data["train_stimuli"], densities=train_densities
    )

    # Save the trained model
    with (fold_output_dir / "model_params.json").open("w") as f:
        json.dump(trained_params, f, indent=4)

    with (fold_output_dir / "model_params.pkl").open("wb") as f:
        pickle.dump(trained_params, f)

    # Convert the priors to a dictionary (from omegaconf object)
    priors = hydra.utils.call(cfg.priors)
    priors = {param: ParameterPrior(*priors[param]) for param in priors}

    # Create wrapped model for evaluation
    sw_wrapped = PysaliencySceneWalk(
        base_model=empirical_density_model,
        pixel_per_degree=cfg.pixel_per_degree,
        use_original_log_likelihoods=False,
        batch_size=100,
        sw_params=trained_params,
        **hydra.utils.call(cfg.model) | {"parameter_priors": priors},
    )

    # Evaluate on each split
    splits = {
        "train": (fold_data["train_stimuli"], fold_data["train_fixations"]),
        "val": (fold_data["val_stimuli"], fold_data["val_fixations"]),
        "test": (fold_data["test_stimuli"], fold_data["test_fixations"]),
    }

    fold_results = {}

    for split_name, (stimuli, fixations) in splits.items():
        if len(fixations) == 0:
            print(f"Skipping empty {split_name} split for fold {fold_data['fold_no']}")
            continue

        fixations_subset = fixations[fixations.scanpath_history_length >= 1]

        if len(fixations_subset) == 0:
            print(f"No valid fixations in {split_name} split for fold {fold_data['fold_no']}")
            continue

        log_likelihoods = sw_wrapped.log_likelihoods(stimuli, fixations_subset, verbose=True)
        igs = (log_likelihoods - pysaliency.UniformModel().log_likelihoods(stimuli, fixations_subset)) / np.log(2)

        # Save per-fixation results
        ll_df = pd.DataFrame(
            {
                "LL": log_likelihoods,
                "IG": igs,
            }
        )
        ll_df.to_csv(fold_output_dir / f"results_per_fixation_{split_name}.csv.gz", index=False)

        # Store average results
        fold_results[split_name] = {
            "Average LL": np.mean(log_likelihoods),
            "Average IG": np.mean(igs),
            "Split": split_name,
            "Fold": fold_data["fold_no"],
        }

        print(
            f"Fold {fold_data['fold_no']} {split_name} - Average LL: {np.mean(log_likelihoods):.4f}, Average IG: {np.mean(igs):.4f}"
        )

    # Save fold summary results
    if fold_results:
        fold_summary_df = pd.DataFrame(list(fold_results.values()))
        fold_summary_df.to_csv(fold_output_dir / "results.csv", index=False)

    return fold_results


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Import fixations and density model
    stimuli = pysaliency.FileStimuli.read_hdf5("pysaliency_datasets/PotsdamCorpus/stimuli.hdf5")
    all_fixations = pysaliency.ScanpathFixations.read_hdf5("pysaliency_datasets/PotsdamCorpus/fixations.hdf5")

    empirical_density_model = pysaliency.HDF5Model(
        stimuli=stimuli,
        filename="pseudo_crossvalidated_gold_standard.hdf5",
        caching=True,
    )

    # Setup output directory
    base_output_dir = Path("output")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    crossval_config = cfg.crossvalidation

    # Check for existing completed folds
    completed_folds = get_completed_folds(base_output_dir, crossval_config)
    total_folds = crossval_config["folds"]

    if completed_folds:
        print(f"Found {len(completed_folds)} completed folds: {completed_folds}")
        print(f"Resuming from fold {max(completed_folds) + 1}")

        # Load existing results
        all_fold_results = load_existing_fold_results(base_output_dir, crossval_config)
    else:
        print("Starting cross-validation...")
        all_fold_results = []

    # Process remaining folds
    folds_processed_this_session = 0

    for fold_data in iterate_crossvalidation_config(stimuli, all_fixations, crossval_config, cfg, str(base_output_dir)):
        fold_no = fold_data["fold_no"]

        # Skip if this fold is already completed
        if fold_no in completed_folds:
            print(f"Skipping already completed fold {fold_no}")
            continue

        print(f"Processing fold {fold_no} ({fold_no + 1}/{total_folds})")

        try:
            fold_results = train_and_evaluate_fold(cfg, fold_data, empirical_density_model)
            all_fold_results.extend(fold_results.values())
            folds_processed_this_session += 1

            print(f"✓ Completed fold {fold_no}")

        except Exception as e:
            print(f"✗ Error processing fold {fold_no}: {e}")
            break

    # Update completed folds list
    final_completed_folds = get_completed_folds(base_output_dir, crossval_config)

    print("\nCross-validation status:")
    print(f"  Total folds: {total_folds}")
    print(f"  Completed folds: {len(final_completed_folds)}")
    print(f"  Folds processed this session: {folds_processed_this_session}")
    print(f"  Remaining folds: {total_folds - len(final_completed_folds)}")

    if len(final_completed_folds) == total_folds:
        print("✓ All folds completed!")
    else:
        missing_folds = [i for i in range(total_folds) if i not in final_completed_folds]
        print(f"Missing folds: {missing_folds}")
        print("Run the script again to complete remaining folds.")

    # Save overall cross-validation summary (from all available results)
    if all_fold_results:
        cv_summary_df = pd.DataFrame(all_fold_results)
        cv_summary_df.to_csv(base_output_dir / "crossvalidation_summary.csv", index=False)

        # Print overall statistics
        for split in ["train", "val", "test"]:
            split_results = cv_summary_df[cv_summary_df["Split"] == split]
            if len(split_results) > 0:
                avg_ll = split_results["Average LL"].mean()
                std_ll = split_results["Average LL"].std()
                avg_ig = split_results["Average IG"].mean()
                std_ig = split_results["Average IG"].std()
                n_folds = len(split_results)
                print(
                    f"Overall {split} ({n_folds} folds) - LL: {avg_ll:.4f} ± {std_ll:.4f}, IG: {avg_ig:.4f} ± {std_ig:.4f}"
                )

    # Pool results across all completed folds
    pool_crossvalidation_results(cfg, stimuli, all_fixations, base_output_dir)


if __name__ == "__main__":
    main()
