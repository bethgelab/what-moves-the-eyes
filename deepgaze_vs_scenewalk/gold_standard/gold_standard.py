import importlib
import logging
import os
import pathlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pysaliency
import yaml
from boltons.iterutils import chunked_iter
from pysaliency.baseline_utils import (
    CrossvalMultipleRegularizations,
    KDEGoldModel,
    ScikitLearnImageSubjectCrossValidationGenerator,
    ScikitLearnWithinImageCrossValidationGenerator,
)
from scipy.optimize import minimize
from tqdm import tqdm


class HDF5ModelManager:
    def __init__(self):
        self.models = []

    def get_model(self, stimuli, filename):
        stimulus_filenames = set(stimuli.filenames)
        for model_data in self.models:
            if model_data["filename"] == filename and stimulus_filenames.issubset(model_data["stimulus_filenames"]):
                return model_data["model"]

        model = pysaliency.HDF5Model(stimuli, filename, caching=False)
        self.models.append(
            {
                "model": model,
                "filename": filename,
                "stimulus_filenames": stimulus_filenames,
            }
        )
        return model


def load_model(stimuli, fixations, config, hdf5_model_manager):
    model_type = config.get("type", "hdf5")
    if model_type == "uniform":
        model = pysaliency.UniformModel()
    elif model_type == "subject_dependent":
        model = _load_subject_model(stimuli, fixations, config["model_directory"])
    elif model_type == "hdf5":
        model = hdf5_model_manager.get_model(stimuli, config["model_file"])
        # model = pysaliency.HDF5Model(stimuli, config['model_file'], caching=False)
    else:
        raise ValueError("Invalid model type", model_type)

    return model


def _load_subject_model(stimuli, fixations, directory):
    subject_models = {}
    for s in range(fixations.subject_count):
        subject_model = pysaliency.HDF5Model(
            stimuli, os.path.join(directory, "subject{}.hdf5".format(s)), caching=False
        )
        subject_models[s] = subject_model

    model = pysaliency.SubjectDependentModel(subject_models)

    return model


def import_from_string(name):
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)

        module = importlib.import_module(module_name)
        klazz = getattr(module, class_name)

        return klazz
    else:
        globals()[name]


def load_regularization_models(stimuli, fixations, config, hdf5_model_manager: HDF5ModelManager):
    regularization_models = OrderedDict()
    for model_data in config["regularizations"]:
        logging.debug("Loading regularization model")
        logging.debug(model_data)
        model = load_model(stimuli, fixations, model_data, hdf5_model_manager)
        regularization_models[model_data["name"]] = model

    return regularization_models


def get_model_for_gold_mixture(stimuli, model_data, subject, hdf5_model_manager: HDF5ModelManager):
    if model_data.get("type", "hdf5") in ["hdf5", "uniform"]:
        return load_model(stimuli, None, model_data, hdf5_model_manager)
    elif model_data["type"] == "subject_dependent":
        if subject is None:
            return pysaliency.HDF5Model(
                stimuli, os.path.join(model_data["model_directory"], "mixture.hdf5"), caching=False
            )
        else:
            return pysaliency.HDF5Model(
                stimuli, os.path.join(model_data["model_directory"], "subject{}.hdf5".format(subject)), caching=False
            )
    else:
        raise ValueError(model_data)


def get_gold_standard_uniform_centerbias_model(
    stimuli, fixations, config, df_parameters, hdf5_model_manager: HDF5ModelManager, grid_spacing=1
):
    subjects = sorted(set(fixations.subjects))

    subject_models_per_stimulus = {subject: {} for subject in subjects}
    upper_model_per_stimulus = {}

    logging.debug("Preloading stimulus ids")
    # This ensures we the stimulus ids will be passed on to the substimuli later and avoid that all of them
    # keep at least one stimulus in memory (the default behavior of pysaliency)
    list(tqdm(stimuli.stimulus_ids))

    logging.debug("Building stimulus specific models")
    for stimulus_index in tqdm(range(len(stimuli))):
        parameters = df_parameters.loc[stimulus_index]

        all_indices = list(np.nonzero(np.array(stimuli.stimulus_ids) == stimuli.stimulus_ids[stimulus_index])[0])
        if len(all_indices) > 1:
            logging.warning(
                f"WARNING, more than one stimulus with the same ID {stimuli.stimulus_ids[stimulus_index]} at indices {all_indices}"
            )
            if all_indices.index(stimulus_index) > 0:
                logging.warning("Skipping non-first stimulus with the same ID")
                continue

        this_stimuli, this_fixations = pysaliency.create_subset(stimuli, fixations, stimuli_indices=all_indices)
        # this_stimuli, this_fixations = pysaliency.create_subset(stimuli, fixations, stimuli_indices=[stimulus_index])
        this_subject_models, this_gold_standard_crossval, this_gold_standard_upper = (
            get_gold_standard_uniform_centerbias_model_for_stimulus(
                this_stimuli,
                this_fixations,
                config,
                parameters,
                subjects,
                hdf5_model_manager=hdf5_model_manager,
                grid_spacing=grid_spacing,
            )
        )

        for subject in subjects:
            subject_models_per_stimulus[subject][this_stimuli] = this_subject_models[subject]
        upper_model_per_stimulus[this_stimuli] = this_gold_standard_upper

    subject_models = {
        subject: pysaliency.StimulusDependentModel(
            subject_models_per_stimulus[subject], caching=False, check_stimuli=False
        )
        for subject in subjects
    }
    crossval_model = pysaliency.SubjectDependentModel(subject_models)
    upper_model = pysaliency.StimulusDependentModel(upper_model_per_stimulus, caching=False, check_stimuli=True)

    return subject_models, crossval_model, upper_model


def get_gold_standard_uniform_centerbias_model_for_stimulus(
    stimuli, fixations, config, parameters, subjects, hdf5_model_manager: HDF5ModelManager, grid_spacing=1
):
    log_bandwidth = parameters["log_bandwidth"]

    _mixture_models = []
    _weights = []

    def get_gold_subject_model(stimuli, fixations, mixture_weights, mixture_models):
        kde_model = KDEGoldModel(
            stimuli,
            fixations,
            bandwidth=10**log_bandwidth,
            eps=0,
            keep_aspect=True,
            caching=False,
            grid_spacing=grid_spacing,
        )

        _mixture_models = [kde_model] + mixture_models
        weights = [1.0 - np.sum(mixture_weights)] + mixture_weights

        mixture_model = pysaliency.MixtureModel(_mixture_models, weights=weights, caching=True, memory_cache_size=4)
        return mixture_model

    subject_models = {}
    for s in tqdm(subjects, disable=True):
        # if not np.any(fixations.subjects == s):
        #     print("Skipping inexistent subject", s)
        #     continue

        _mixture_models = []
        _weights = []

        for model_data in config["regularizations"]:
            _mixture_models.append(
                get_model_for_gold_mixture(stimuli, model_data, s, hdf5_model_manager=hdf5_model_manager)
            )
            _weights.append(10 ** parameters["log_{}".format(model_data["name"])])

        subject_model = get_gold_subject_model(stimuli, fixations[fixations.subjects != s], _weights, _mixture_models)
        subject_models[s] = subject_model

    gold_standard_crossval = pysaliency.SubjectDependentModel(subject_models)

    _mixture_models = []
    _weights = []

    for model_data in config["regularizations"]:
        _mixture_models.append(
            get_model_for_gold_mixture(stimuli, model_data, None, hdf5_model_manager=hdf5_model_manager)
        )
        _weights.append(10 ** parameters["log_{}".format(model_data["name"])])

    gold_standard_upper = get_gold_subject_model(stimuli, fixations, _weights, _mixture_models)

    return subject_models, gold_standard_crossval, gold_standard_upper


def load_gold_models(stimuli: pysaliency.Stimuli, fixations: pysaliency.Fixations, directory: pathlib.Path):
    subject_models = {}
    for s in range(fixations.subject_count):
        if not np.any(fixations.subjects == s):
            logging.info(f"Skipping inexistent subject {s}")
            continue
        subject_model = pysaliency.HDF5Model(
            stimuli, os.path.join(directory, "subject{}.hdf5".format(s)), caching=False
        )
        subject_models[s] = subject_model

    gold_standard_crossval = pysaliency.SubjectDependentModel(subject_models)

    gold_standard_upper = pysaliency.HDF5Model(stimuli, os.path.join(directory, "mixture.hdf5"), caching=False)

    return subject_models, gold_standard_crossval, gold_standard_upper


def eval_model(
    model: pysaliency.ScanpathModel,
    stimuli: pysaliency.Stimuli,
    fixations: pysaliency.Fixations,
    pickle_path=None,
    batch_size=50,
):
    # TODO: once multi metric eval is done in pysaliency,
    # switch to this paradigm

    if pickle_path is not None and os.path.isfile(pickle_path):
        logging.debug("Loading from pickle")
        df = pd.read_pickle(pickle_path)
    else:
        df = pd.DataFrame({"LL": [], "AUC": [], "NSS": []})

    if isinstance(model, pysaliency.models.SubjectDependentModel):
        density_model = model.get_saliency_map_model_for_NSS()
    elif isinstance(model, pysaliency.Model):
        density_model = pysaliency.saliency_map_models.DensitySaliencyMapModel(model)
    # elif isinstance(model, pysaliency.models.StimulusDependentScanpathModel) and all(isinstance(m, pysaliency.SubjectDependentModel) for m in model.stimuli_models):
    #     density_model = pysaliency.saliency_map_models.Stimu
    else:
        raise TypeError(model)

    while len(df) < len(fixations):
        with tqdm(total=len(fixations), initial=len(df)) as pbar:
            for fixation_indices in chunked_iter(range(len(df), len(fixations)), batch_size):
                logging.debug(f"Run batch {fixation_indices}")
                fixation_batch = fixations[fixation_indices]
                ns = fixation_batch.n
                subjects = fixation_batch.subjects
                unique = sorted(set(zip(ns, subjects)))
                logging.debug(f"image/subject combinations {unique}")
                log_likelihoods = model.information_gains(stimuli, fixation_batch, verbose=False)
                AUCs = density_model.AUCs(stimuli, fixation_batch, verbose=False)
                NSSs = density_model.NSSs(stimuli, fixation_batch, verbose=False)

                next_part = pd.DataFrame(
                    {
                        "LL": log_likelihoods,
                        "AUC": AUCs,
                        "NSS": NSSs,
                    },
                    index=fixation_indices,
                )
                df = pd.concat((df, next_part), ignore_index=False)
                logging.debug(df.mean())
                if pickle_path is not None:
                    logging.debug(f"saving to {pickle_path}")
                    df.to_pickle(pickle_path)
                pbar.update(len(fixation_batch))

    return df


def fit_gold_standard(config, output_directory: Path):
    dataset_config = config.pop("dataset")

    log_bandwidth_min = config.pop("log_bandwidth_min", -3)
    log_bandwidth_max = config.pop("log_bandwidth_max", 0)

    output_directory.mkdir(exist_ok=True)

    parameters_file = output_directory / "parameters.csv"
    results_file = output_directory / "results.yaml"

    stimuli, fixations = pysaliency.load_dataset_from_config(dataset_config)
    stimuli.cached = False

    # print("checking stimulus ids")
    # if not len(stimuli) == len(set(tqdm(stimuli.stimulus_ids))):
    #     n_stimuli = len(stimuli)
    #     n_unique_stimuli = len(set(stimuli.stimulus_ids))
    # raise ValueError(f"Stimulus IDs are not unique: {n_stimuli} stimuli, but only {n_unique_stimuli} unique stimulus IDs.")

    total_sample_size = config.get("total_sample_size")
    all_fixations = fixations

    if total_sample_size is not None and total_sample_size < len(fixations):
        logging.info(f"Selecting subset of {total_sample_size} fixations")
        fixation_subset_indices = np.random.RandomState(seed=42).choice(
            len(fixations), size=total_sample_size, replace=False
        )
        fixations = fixations[fixation_subset_indices]

    def save_results(results):
        with open(results_file, "w") as f:
            yaml.safe_dump(results, f, default_flow_style=False)

    # def load_results():
    #     if os.path.isfile(results_file):
    #         return yaml.safe_load(open(results_file))
    #     else:
    #         return {}

    hdf5_model_manager = HDF5ModelManager()

    # load full hdf5 models for later reuse
    for regularization in config["regularizations"]:
        model_type = regularization.get("type", "hdf5")
        if model_type == "hdf5":
            logging.info(f"caching {regularization}")
            hdf5_model_manager.get_model(stimuli, regularization["model_file"])
            logging.debug("done")

    params = [("log_bandwidth", (log_bandwidth_min, log_bandwidth_max))]
    for model_data in config["regularizations"]:
        reg_min = model_data.get("log_weight_min", -10)
        reg_max = model_data.get("log_weight_max", 0)
        params.append(("log_{}".format(model_data["name"]), (reg_min, reg_max)))
    params = OrderedDict(params)

    logging.info(params)
    logging.info(list(params.keys()))

    if parameters_file.is_file():
        df_parameters = pd.read_csv(parameters_file, index_col=0)
    else:
        df_parameters = pd.DataFrame(columns=list(params.keys()) + ["score"])

    for stimulus_index in range(len(stimuli)):
        if stimulus_index in df_parameters.index:
            logging.debug(f"Already computed {stimulus_index}")
            continue

        logging.info(f"Stimulus, {stimulus_index}")

        all_indices = list(np.nonzero(np.array(stimuli.stimulus_ids) == stimuli.stimulus_ids[stimulus_index])[0])
        if len(all_indices) > 1:
            logging.warn(
                f"WARNING, more than one stimulus with the same ID {stimuli.stimulus_ids[stimulus_index]} at indices {all_indices}"
            )

        this_stimuli, this_fixations = pysaliency.create_subset(stimuli, fixations, stimuli_indices=all_indices)

        if fixations.subject_count > 1:
            logging.debug("Using leave-one-subject-out crossvalidation")
            crossvalidation = ScikitLearnImageSubjectCrossValidationGenerator(this_stimuli, this_fixations)
        else:
            logging.debug("using 10-fold crossvalidation over image fixations")
            crossvalidation = ScikitLearnWithinImageCrossValidationGenerator(this_stimuli, this_fixations)

        n_jobs = config.get("n_jobs", None)
        regularization_models = load_regularization_models(this_stimuli, this_fixations, config, hdf5_model_manager)
        manager = CrossvalMultipleRegularizations(
            this_stimuli, this_fixations, regularization_models, crossvalidation, n_jobs=n_jobs, verbose=1
        )

        selected_cross_val_score = manager.score

        # Random search

        random_search_config = config["random_search"]

        columns = list(params.keys()) + ["score"]
        random_search_data = pd.DataFrame(columns=columns, dtype=np.float64)

        with tqdm(total=random_search_config["steps"], initial=len(random_search_data)) as pbar:
            while len(random_search_data) < random_search_config["steps"]:
                this_params = {}
                for param_name, param_bounds in params.items():
                    this_params[param_name] = np.random.uniform(low=param_bounds[0], high=param_bounds[1])

                score = selected_cross_val_score(**this_params)

                new_row = pd.Series(this_params)
                new_row["score"] = score

                random_search_data = pd.concat((random_search_data, pd.DataFrame([new_row])), ignore_index=True)
                # print(random_search_data.dtypes)
                random_search_data["best_score"] = random_search_data["score"].cummax()
                pbar.update(1)

        def selected_crossval_cost(x, selected_cross_val_score=selected_cross_val_score):
            """scipy.optimize-compatible version of the score"""
            # print('.', end='', flush=True)
            return -selected_cross_val_score(*x)

        best_row = random_search_data.loc[random_search_data["score"].idxmax()]
        logging.debug(f"best row {best_row}")

        x0 = np.array([best_row[key] for key in params])
        bounds = list(params.values())
        res = minimize(selected_crossval_cost, x0, options={"disp": 10, "iprint": 1}, bounds=bounds)

        logging.info(res)

        for index in all_indices:
            row = {key: float(res.x[i]) for i, key in enumerate(params)}
            row["score"] = -float(res.fun)
            row = pd.Series(row, name=index)

            # print(row)
            df_parameters = pd.concat((df_parameters, pd.DataFrame([row])), ignore_index=False)

        # print(df_parameters.tail())

        df_parameters.to_csv(parameters_file)

    logging.info("creating models")
    grid_spacing = config.get("grid_spacing", 1)
    subject_models, crossval_model, upper_model = get_gold_standard_uniform_centerbias_model(
        stimuli, fixations, config, df_parameters, hdf5_model_manager=hdf5_model_manager, grid_spacing=grid_spacing
    )

    if config.get("export_hdf5", True) or config.get("export_hdf5_subjects", False):
        for s in range(fixations.subject_count):
            if not np.any(fixations.subjects == s):
                logging.debug(f"Skipping inexistent subject {s}")
                continue
            logging.info(f"Exporting subject model {s}")
            pysaliency.export_model_to_hdf5(
                subject_models[s],
                stimuli,
                os.path.join(output_directory, "subject{}.hdf5".format(s)),
                overwrite=False,
                flush=True,
            )
    if config.get("export_hdf5", True) or config.get("export_hdf5_mixture", False):
        logging.info("Exporting mixture model")
        pysaliency.export_model_to_hdf5(
            upper_model, stimuli, os.path.join(output_directory, "mixture.hdf5"), overwrite=False, flush=True
        )

    if config.get("export_hdf5", True) or (
        config.get("export_hdf5_mixture", False) and config.get("export_hdf5_subjects", False)
    ):
        subject_models, crossval_model, upper_model = load_gold_models(output_directory)

    if config.get("evaluate_models", True):
        logging.info("evaluating models")

        df_crossval = eval_model(
            crossval_model, stimuli, all_fixations, pickle_path=output_directory / "eval_crossval.pkl"
        )
        df_crossval.to_csv(output_directory / "eval_crossval.csv")
        df_crossval.to_pickle(output_directory / "eval_crossval.pkl")

        df_upper = eval_model(upper_model, stimuli, all_fixations, pickle_path=output_directory / "eval_upper.pkl")
        df_upper.to_csv(output_directory / "eval_upper.csv")
        df_upper.to_pickle(output_directory / "eval_upper.pkl")

        output = {}
        output["log_likelihood"] = float(
            pysaliency.utils.average_values(df_crossval["LL"], all_fixations, average="image")
        )
        output["log_likelihood_upper"] = float(
            pysaliency.utils.average_values(df_upper["LL"], all_fixations, average="image")
        )
        save_results(output)
