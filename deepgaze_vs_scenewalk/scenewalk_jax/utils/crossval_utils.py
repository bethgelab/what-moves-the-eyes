import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysaliency
from pysaliency.filter_datasets import iterate_crossvalidation


def iterate_crossvalidation_config(stimuli, fixations, crossval_config, config, output_dir="output"):
    for fold_no, (
        train_stimuli,
        train_fixations,
        val_stimuli,
        val_fixations,
        test_stimuli,
        test_fixations,
    ) in enumerate(
        iterate_crossvalidation(
            stimuli,
            fixations,
            crossval_folds=crossval_config["folds"],
            val_folds=crossval_config["val_folds"],
            test_folds=crossval_config["test_folds"],
            stratified_attributes=crossval_config.get("stratified_attributes", None),
        )
    ):
        yield {
            "config": config,
            "directory": os.path.join(output_dir, f"crossval-{crossval_config['folds']}-{fold_no}"),
            "fold_no": fold_no,
            "crossval_folds": crossval_config["folds"],
            "train_stimuli": train_stimuli,
            "train_fixations": train_fixations,
            "val_stimuli": val_stimuli,
            "val_fixations": val_fixations,
            "test_stimuli": test_stimuli,
            "test_fixations": test_fixations,
        }


def is_fold_completed(fold_output_dir):
    """Check if a fold has been completed by looking for expected output files."""
    fold_path = Path(fold_output_dir)

    # Check for essential files that indicate completion
    required_files = ["model_params.json", "model_params.pkl", "results.csv"]

    # Also check for at least one results file (since splits might be empty)
    result_patterns = [
        "results_per_fixation_train.csv.gz",
        "results_per_fixation_val.csv.gz",
        "results_per_fixation_test.csv.gz",
    ]

    # All required files must exist
    for filename in required_files:
        if not (fold_path / filename).exists():
            return False

    # At least one result file must exist
    has_results = any((fold_path / filename).exists() for filename in result_patterns)

    return has_results


def get_completed_folds(base_output_dir, crossval_config):
    """Get list of completed fold numbers."""
    completed_folds = []

    for fold_no in range(crossval_config["folds"]):
        fold_dir = base_output_dir / f"crossval-{crossval_config['folds']}-{fold_no}"
        if is_fold_completed(fold_dir):
            completed_folds.append(fold_no)

    return completed_folds


def load_existing_fold_results(base_output_dir, crossval_config):
    """Load results from all completed folds."""
    all_fold_results = []
    completed_folds = get_completed_folds(base_output_dir, crossval_config)

    for fold_no in completed_folds:
        fold_dir = base_output_dir / f"crossval-{crossval_config['folds']}-{fold_no}"
        results_file = fold_dir / "results.csv"

        if results_file.exists():
            fold_results = pd.read_csv(results_file)
            # Convert to list of dictionaries
            all_fold_results.extend(fold_results.to_dict("records"))
            print(f"Loaded results from completed fold {fold_no}")

    return all_fold_results


def pool_crossvalidation_results(cfg, stimuli, all_fixations, base_output_dir):
    """Pool results across all cross-validation folds."""
    print("Pooling cross-validation results...")

    # Add fixation index for mapping results back
    fixations_subset = all_fixations[all_fixations.scanpath_history_length >= 1].copy()
    fixations_subset.fixation_index = np.arange(len(fixations_subset))
    fixations_subset.__attributes__.append("fixation_index")

    crossval_config = cfg.crossvalidation
    completed_folds = get_completed_folds(base_output_dir, crossval_config)

    if not completed_folds:
        print("No completed folds found for pooling.")
        return

    print(f"Pooling results from completed folds: {completed_folds}")

    # Pool results for each split
    for split_name in ["train", "val", "test"]:
        parts = []

        for crossval_fold in completed_folds:
            # Get the appropriate fixations for this fold and split
            if split_name == "test":
                _, split_fixations = pysaliency.filter_datasets.test_split(
                    stimuli,
                    fixations_subset,
                    crossval_folds=crossval_config["folds"],
                    fold_no=crossval_fold,
                    val_folds=crossval_config["val_folds"],
                    test_folds=crossval_config["test_folds"],
                )
            elif split_name == "val":
                _, split_fixations = pysaliency.filter_datasets.validation_split(
                    stimuli,
                    fixations_subset,
                    crossval_folds=crossval_config["folds"],
                    fold_no=crossval_fold,
                    val_folds=crossval_config["val_folds"],
                    test_folds=crossval_config["test_folds"],
                )
            else:  # train
                _, split_fixations = pysaliency.filter_datasets.train_split(
                    stimuli,
                    fixations_subset,
                    crossval_folds=crossval_config["folds"],
                    fold_no=crossval_fold,
                    val_folds=crossval_config["val_folds"],
                    test_folds=crossval_config["test_folds"],
                )

            results_file = (
                base_output_dir
                / f"crossval-{crossval_config['folds']}-{crossval_fold}"
                / f"results_per_fixation_{split_name}.csv.gz"
            )

            if not results_file.exists():
                print(f"Warning: Results file not found: {results_file}")
                continue

            results = pd.read_csv(results_file)
            if hasattr(split_fixations, "fixation_index"):
                results["fixation_index"] = split_fixations.fixation_index
                results = results.set_index("fixation_index")
                parts.append(results)

        if parts:
            pooled_results = pd.concat(parts).sort_index()
            pooled_results.to_csv(base_output_dir / f"pooled_results_{split_name}.csv.gz")

            # Also save summary statistics
            summary_stats = pd.DataFrame(
                {
                    "Split": [split_name],
                    "Average LL": [pooled_results["LL"].mean()],
                    "Average IG": [pooled_results["IG"].mean()],
                    "Std LL": [pooled_results["LL"].std()],
                    "Std IG": [pooled_results["IG"].std()],
                    "N_fixations": [len(pooled_results)],
                    "N_folds": [len(completed_folds)],
                }
            )
            summary_stats.to_csv(base_output_dir / f"pooled_summary_{split_name}.csv", index=False)

            print(
                f"Pooled {split_name} results ({len(completed_folds)} folds) - Average LL: {pooled_results['LL'].mean():.4f}, Average IG: {pooled_results['IG'].mean():.4f}"
            )
