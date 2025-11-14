import warnings

import equinox as eqx
import hydra
import jax
import numpy as np
import omegaconf
import optax
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from .jax_model import ParameterPrior, SceneWalk
from .jax_training import (
    FixationsDatasetAcrossSubjects,
    fit_scenewalk,
    keep_params_sign,
    keep_params_within_bounds,
    pad_collate,
)


def check_stim_data_range(cfg, sizes):
    max_y_px, max_x_px = next(iter(sizes))

    max_y_deg, max_x_deg = max_y_px / cfg.pixel_per_degree, max_x_px / cfg.pixel_per_degree

    x_max_range = cfg.model.data_range.x[1]
    y_max_range = cfg.model.data_range.y[1]

    if max_x_deg > x_max_range:
        warnings.warn(
            f"Maximum x degree value ({max_x_deg}) is larger than the model data range ({x_max_range}). "
            "This will lead to unexpected behaviour.",
            UserWarning,
            stacklevel=2,
        )

    if max_y_deg > y_max_range:
        warnings.warn(
            f"Maximum y degree value ({max_y_deg}) is larger than the model data range ({y_max_range}). "
            "This will lead to unexpected behaviour.",
            UserWarning,
            stacklevel=2,
        )


def train_from_config(
    cfg: omegaconf.DictConfig,
    fixations,
    stimuli,
    densities,
    val_fixations=None,
    val_stimuli=None,
    val_densities=None,
):
    # Checks and warnings
    sizes = set(stimuli.sizes)
    if len(sizes) > 1:
        raise ValueError(f"Multiple sizes found in the stimuli, this is not supported: {sizes}")

    check_stim_data_range(cfg, sizes)

    if cfg.optimizer.enforce_bounds and cfg.optimizer.keep_param_signs:
        warnings.warn(
            "Both enforce_bounds and keep_param_signs are set to True. This may lead to unexpected behaviour, "
            "and will substantially increase CUDA memory usage.",
            UserWarning,
            stacklevel=2,
        )

    # Convert the priors to a dictionary (from omegaconf object)
    priors = hydra.utils.call(cfg.priors)

    # Instantiate priors.
    priors = {param: ParameterPrior(*priors[param]) for param in priors}

    # Initialise SceneWalk model and check it
    sw_jax = SceneWalk.from_trained(
        hyperparams=hydra.utils.call(cfg.model) | {"parameter_priors": priors},
        trained_params=hydra.utils.call(cfg.init_params),
    )

    print(f"Initialised the following SceneWalk model: \n{sw_jax.whoami()}")

    # Subselect fixations for training and validation

    if cfg.val_size is not None and val_fixations is None:
        # In this case we do not provide a set val split and we need to create one from the training data
        train_indices, val_indices = train_test_split(
            np.arange(len(fixations.scanpaths)), test_size=0.2, random_state=42
        )

        train_fixations = fixations.filter_scanpaths(train_indices)
        val_fixations = fixations.filter_scanpaths(val_indices)
        # Stimuli and densities are the same in this case, we split by scanpaths
        val_stimuli = stimuli
        val_densities = densities
    else:
        train_fixations = fixations

    resize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((cfg.MAP_SIZE, cfg.MAP_SIZE), antialias=True),
        ]
    )

    train_dataset = FixationsDatasetAcrossSubjects(
        train_fixations.scanpaths.xs,
        train_fixations.scanpaths.ys,
        train_fixations.scanpaths.fixation_attributes["durations"],
        densities,
        train_fixations.scanpaths.n,
        transform=resize,
        px_per_degree=cfg.pixel_per_degree,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0
    )

    if val_fixations is not None:
        assert val_stimuli is not None, "Validation stimuli must be provided if validation fixations are provided"
        assert val_densities is not None, "Validation densities must be provided if validation fixations are provided"
        assert set(val_stimuli.sizes) == sizes, "Validation stimuli sizes do not match training stimuli sizes"

        val_dataset = FixationsDatasetAcrossSubjects(
            val_fixations.scanpaths.xs,
            val_fixations.scanpaths.ys,
            val_fixations.scanpaths.fixation_attributes["durations"],
            val_densities,
            val_fixations.scanpaths.n,
            transform=resize,
            px_per_degree=cfg.pixel_per_degree,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0
        )
    else:
        val_loader = None

    @optax.inject_hyperparams
    def define_optimizer(
        optimizer_args: dict | omegaconf.DictConfig,
        lr: float | optax.Schedule = 1e-3,
        clip_norm: bool = True,
        keep_signs: bool = False,
        enforce_bounds: bool = False,
    ):
        optimizer = hydra.utils.call(optimizer_args, learning_rate=lr)

        optim_list = [optimizer]

        if clip_norm:
            optim_list.insert(0, optax.clip_by_global_norm(1))
        if keep_signs:
            optim_list.append(keep_params_sign())
        if enforce_bounds:
            optim_list.append(keep_params_within_bounds(sw_jax.PARAMETER_BOUNDS))

        return optax.chain(*optim_list)

    onecycle_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.training.epochs * len(train_loader),
        peak_value=cfg.training.peak_lr,
        pct_start=0.2,
    )

    optimizer = define_optimizer(
        optimizer_args=cfg.optimizer.object,
        lr=onecycle_schedule,
        clip_norm=cfg.optimizer.clip_norm,
        keep_signs=cfg.optimizer.keep_param_signs,
        enforce_bounds=cfg.optimizer.enforce_bounds,
    )

    eqx.clear_caches()
    jax.clear_caches()

    sw_jax, loss = fit_scenewalk(
        sw_jax,
        optimizer,
        train_loader,
        val_loader,
        epochs=cfg.training.epochs,
        method=cfg.training.strategy,
        prior_weight=cfg.training.prior_weight,
        early_stopping=cfg.training.early_stopping,
        fix_params=cfg.training.fix_params,
        verbose=cfg.training.verbose,
    )

    trained_params = sw_jax.get_params()

    # Convert jax arrays to values
    trained_params = {k: v.tolist() for k, v in trained_params.items()}

    return trained_params, loss, sw_jax
