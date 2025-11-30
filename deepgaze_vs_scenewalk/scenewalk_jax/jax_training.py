import datetime
import os
import socket
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpyro.distributions as dist
import optax
import pysaliency
import torch
from optax._src import base
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

from .jax_model import EPS, PARAMETER_PRIORS, SceneWalk, load_model_parameters


def is_not_property(obj, attr_name):
    """Check if the attribute is not a property."""
    attr = getattr(obj.__class__, attr_name, None)
    return not isinstance(attr, property)


def jax_has_gpu() -> bool:
    """Utility function to check if JAX has access to a GPU."""
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except Exception:
        return False


def flatten_object_array(array: np.ndarray):
    """
    Utility function to flatten an array of objects into a single array.
    Useful for datasets not provided through pysaliency.
    """
    return np.concatenate(array) if array.dtype == np.dtype("object") else array


def pad_to_length(array, length):
    """
    Utility function to pad an array to a given length.
    Useful for datasets not provided through pysaliency.
    """
    if len(array) < length:
        return np.pad(array, (0, int(length - len(array))), constant_values=np.nan)
    return array


def log_prior_loss(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    affected_params_keys = [param_name for param_name in PARAMETER_PRIORS if param_name in params]
    affected_params_values = jnp.array([params[param_name] for param_name in affected_params_keys])

    # Define means and standard deviations for the truncated normal
    prior_means = jnp.array([PARAMETER_PRIORS[param_name][0] for param_name in affected_params_keys])
    prior_stds = jnp.array([PARAMETER_PRIORS[param_name][1] for param_name in affected_params_keys])

    # Define bounds for truncation
    lower_bound = jnp.array([PARAMETER_PRIORS[param_name][2] for param_name in affected_params_keys])
    upper_bound = jnp.array([PARAMETER_PRIORS[param_name][3] for param_name in affected_params_keys])

    log_probs = dist.TruncatedNormal(loc=prior_means, scale=prior_stds, low=lower_bound, high=upper_bound).log_prob(
        affected_params_values
    )

    return jnp.sum(log_probs)


def batched_base_loss(model: SceneWalk, *, x_path, y_path, dur_path, fix_dens, dataset_len=None):
    """
    Mean negative log-likelihood loss per fixation for a batch of scanpaths.
    """
    # Compute the log-likelihood of the scanpath
    batch_ll = eqx.filter_vmap(model.get_scanpath_likelihood)(x_path, y_path, dur_path, fix_dens)

    # Using nan mean because log-likelihoods for the fixations are padded with NaNs at the end.
    # Also scaling by the grid size and converting to nats.
    return -(jnp.nanmean(batch_ll) + jnp.log2(128 * 128))


def batched_map_loss(model: SceneWalk, x_path, y_path, dur_path, fix_dens, dataset_len, prior_weight):
    """
    Computes the batched MAP (Maximum A Posteriori) loss for the given model.

    This function calculates the negative log-likelihood loss and the log prior loss,
    scales the prior loss by the number of fixations in the batch, and returns the
    combined loss.

    Args:
        model (SceneWalk): The SceneWalk model instance.
        dataset_len (int): The length of the dataset, in fixations.
        x_path (array-like): The x-coordinates of the fixation points.
        y_path (array-like): The y-coordinates of the fixation points.
        dur_path (array-like): The durations of the fixation points.
        fix_dens (array-like): The fixation density.

    Returns:
        float: The computed batched MAP loss.
    """
    negative_log_lik = batched_base_loss(model, x_path=x_path, y_path=y_path, dur_path=dur_path, fix_dens=fix_dens)

    log_prior = log_prior_loss(model.get_params())

    # Scale the prior loss by the number of fixations in the batch
    batch_n_fixations = jnp.sum(jnp.isfinite(x_path))
    prior_scaling = (batch_n_fixations / dataset_len) * prior_weight

    return negative_log_lik - (prior_scaling * log_prior)


def evaluate_model(model, val_loader, mean=True):
    val_losses = [batched_base_loss(model, **batch) for batch in val_loader]
    if mean:
        return np.mean(val_losses)
    return np.sum(val_losses)


def format_params_readable(params):
    """Format model parameters in a more readable way."""
    max_key_length = max(len(key) for key in params.keys())
    formatted_lines = []

    # Group parameters by their status (NaN vs non-NaN)
    nan_params = {}
    normal_params = {}

    for key, value in params.items():
        # Convert JAX arrays to Python floats for better display
        if hasattr(value, "item"):
            try:
                value_float = value.item()
                if np.isnan(value_float):
                    nan_params[key] = "NaN"
                else:
                    normal_params[key] = f"{value_float:.6f}"
            except:
                # Handle case when item() fails (e.g., for arrays with multiple elements)
                if np.any(np.isnan(value)):
                    nan_params[key] = "Contains NaN"
                else:
                    normal_params[key] = str(value)
        else:
            normal_params[key] = str(value)

    # Format non-NaN parameters first
    if normal_params:
        formatted_lines.append("\n=== Normal Parameters ===")
        for key, value in sorted(normal_params.items()):
            formatted_lines.append(f"{key:{max_key_length}s} = {value}")

    # Format NaN parameters
    if nan_params:
        formatted_lines.append("\n=== NaN Parameters ===")
        for key, value in sorted(nan_params.items()):
            formatted_lines.append(f"{key:{max_key_length}s} = {value}")

    return "\n".join(formatted_lines)


def fit_scenewalk(
    model: SceneWalk,
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 1,
    method: Literal["MAP", "MLE"] = "MAP",
    prior_weight: float = 1.0,
    early_stopping: bool = True,
    fix_params: Optional[list[str]] = None,
    verbose: bool = True,
) -> Tuple[SceneWalk, list]:
    train_losses = []

    # Initialize the optimizer
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Tensorboard writer
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = current_time + "_" + socket.gethostname()
    writer = SummaryWriter(log_dir=os.path.join("output", "tensorboard", log_dir))

    loss = partial(batched_map_loss, prior_weight=prior_weight) if method == "MAP" else batched_base_loss

    # Define stepping function
    @eqx.filter_jit
    def step(model: SceneWalk, opt_state, dataset_len, x_path, y_path, dur_path, fix_dens):
        loss_value, grads = eqx.filter_value_and_grad(loss)(
            model, x_path=x_path, y_path=y_path, dur_path=dur_path, fix_dens=fix_dens, dataset_len=dataset_len
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)  # type: ignore

        # Filter out non-trainable parameters -> Set their updates to None
        not_trainable = model._not_trainable_params() + (fix_params or [])
        for param_name in not_trainable:
            if is_not_property(model, param_name):
                updates = eqx.tree_at(
                    lambda m, p=param_name: getattr(m, p),
                    pytree=updates,
                    replace=None,
                    is_leaf=lambda x: x is None,
                )

        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Early stopping criterion
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3)

    # Loop for epochs
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    # Get number of fixations in the dataset to scale prior loss correctly.
    dataset_len = train_loader.dataset.total_n_fixations  # type: ignore

    # Store the last 5 parameter sets
    parameter_history = []
    max_history_size = 5

    diverged = False
    for epoch in epoch_bar:
        if diverged:
            break
        # Initialize accumulated loss for the epoch
        accumulated_loss = 0.0

        # Loop for batches
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for i, batch in enumerate(batch_bar):
            parameters = model.get_params()

            # Add current parameters to history with epoch and batch information
            parameter_history.append((epoch, i, parameters))
            if len(parameter_history) > max_history_size:
                parameter_history.pop(0)

            model, opt_state, loss_value = step(model, opt_state, dataset_len, **batch)

            if jnp.isnan(loss_value):
                print("Batch loss contains NaNs. Training diverged. Exiting.")
                print("Parameters from the last five steps:")
                for step_epoch, step_i, step_params in parameter_history:
                    print(f"\n--- Parameters at epoch {step_epoch + 1}, batch {step_i} ---")
                    print(format_params_readable(step_params))
                # save buffered parameters
                # with open("last_parameters.pkl", "wb") as f:
                #     pickle.dump(parameter_history, f)
                diverged = True
                break

            accumulated_loss += loss_value

            batch_bar.update()
            if i % 5 == 0:
                batch_bar.set_postfix(loss=loss_value)
                if verbose:
                    tqdm.write(f"Epoch {epoch + 1} - Batch {i}: Loss = {loss_value:.4f}")

        # Calculate average loss for the epoch
        avg_loss = accumulated_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Update epoch progress bar with the average loss value
        epoch_bar.set_postfix(loss=avg_loss)

        if verbose:
            # Print average loss at the end of each epoch
            tqdm.write(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Log model parameters
        parameters = model.get_params()
        for param_name, param_value in parameters.items():
            if jnp.isscalar(param_value):
                writer.add_scalar(f"Parameters/{param_name}", np.array(param_value), epoch)
            else:
                writer.add_histogram(f"Parameters/{param_name}", np.array(param_value), epoch)
        if verbose:
            tqdm.write(format_params_readable(parameters))

        # Tensorboard logging
        writer.add_scalar("Loss/Training", np.array(avg_loss), epoch)
        writer.add_scalar("Learning Rate", np.array(opt_state.hyperparams["lr"]), epoch)  # type: ignore

        # Validation loss
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader)
            if verbose:
                tqdm.write(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/Validation", np.array(val_loss), epoch)

        # Early stopping
        if early_stopper(val_loss if val_loader is not None else avg_loss, model) and early_stopping:
            tqdm.write("Early stopping criterion met. Stopping training.")
            break

    writer.close()

    # Restore parameters of the best epoch
    model = load_model_parameters(model, early_stopper.get_best_model_params())

    return model, train_losses


class FixationsDatasetAcrossSubjects(Dataset):
    """Fixations dataset."""

    def __init__(
        self,
        xs,
        ys,
        durations,
        densities,
        image_ids,
        transform: Optional[Callable] = None,
        flatten_objects=False,
        px_per_degree=None,
    ):
        """
        Arguments:
            ...todo
        """
        flatten_function = flatten_object_array if flatten_objects else lambda x: x

        self.all_xs = flatten_function(xs)
        self.all_ys = flatten_function(ys)
        self.all_durations = flatten_function(durations)

        self.all_image_ids = flatten_function(image_ids)
        self.densities = densities

        self.transform = transform
        self.px_per_degree = px_per_degree

    def __len__(self):
        """Return the number of scanpaths in the dataset."""
        return len(self.all_xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_fix = self.all_xs[idx]
        y_fix = self.all_ys[idx]
        dur_fix = self.all_durations[idx]

        # Unique deals with datasets that provide a duplicate image id for each fixation
        original_density_fix = self.densities[np.unique(self.all_image_ids[idx]).item()]

        if self.transform:
            density_fix_unnorm = self.transform(original_density_fix).numpy().squeeze()
            density_fix = density_fix_unnorm / np.sum(density_fix_unnorm)

            if self.px_per_degree is None:
                # Scale the fixation coordinates to the new size
                new_size = density_fix.shape
                original_size = original_density_fix.shape
                scale_x = new_size[1] / original_size[1]
                scale_y = new_size[0] / original_size[0]

                x_fix = np.copy(x_fix) * scale_x
                y_fix = np.copy(y_fix) * scale_y
            else:
                # Scale to degrees of visual angle
                x_fix = np.copy(x_fix) / self.px_per_degree
                y_fix = np.copy(y_fix) / self.px_per_degree

        sample = {
            "x_path": x_fix,
            "y_path": y_fix,
            "dur_path": dur_fix,
            "fix_dens": density_fix,
        }

        return sample

    @property
    def total_n_fixations(self):
        return sum(len(x) for x in self.all_xs)


def compute_density_maps(
    model: pysaliency.models.Model,
    fixations: pysaliency.datasets.fixations.ScanpathFixations,
    stimuli: pysaliency.datasets.Stimuli,
):
    densities = {}
    for image_id in tqdm(np.unique(fixations.n), desc="Computing densities for all images"):
        empirical_log_density = model.log_density(stimuli[image_id])
        empirical_density = np.exp(empirical_log_density)
        densities[image_id] = empirical_density
    return densities


def pad_collate(batch):
    """
    Pads the sequences in the batch to the maximum sequence length and returns a dictionary with the padded sequences.
    To be used with the DataLoader class.

    Args:
        batch (list): A list of dictionaries, where each dictionary represents a sample in the batch. Each dictionary should contain the following keys:
            - "x_path": A list representing the x_path sequence.
            - "y_path": A list representing the y_path sequence.
            - "dur_path": A list representing the dur_path sequence.
            - "fix_dens": An array representing the fix_dens map.

    Returns:
        dict: A dictionary containing the padded sequences, now correctly batched The keys are as follows:
            - "x_path": A JAX array representing the padded x_path sequences.
            - "y_path": A JAX array representing the padded y_path sequences.
            - "dur_path": A JAX array representing the padded dur_path sequences.
            - "fix_dens": A JAX array representing the fix_dens maps.
    """
    max_seq_len = max(len(item["x_path"]) for item in batch)
    pad_to_max = partial(pad_to_length, length=max_seq_len)
    x_fix = [pad_to_max(item["x_path"]) for item in batch]
    y_fix = [pad_to_max(item["y_path"]) for item in batch]
    dur_fix = [pad_to_max(item["dur_path"]) for item in batch]
    fix_dens = [item["fix_dens"] for item in batch]

    return {
        "x_path": jnp.array(x_fix),
        "y_path": jnp.array(y_fix),
        "dur_path": jnp.array(dur_fix),
        "fix_dens": jnp.array(fix_dens),
    }


def subselect_scanpaths(fixations: pysaliency.datasets.fixations.ScanpathFixations, stimuli, size=None):
    if size is None:
        size = [768, 1024]

    shapes_mask = [stim.shape[:2] == size for stim in stimuli]
    valid_images = np.unique(fixations.scanpaths.n)[shapes_mask]
    valid_scanpaths_mask = [x in valid_images for x in fixations.scanpaths.n]

    return valid_scanpaths_mask


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_loss: float = jnp.inf
    counter: int = 0

    def __call__(self, loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model_params = model.get_params()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        else:
            return False

    def get_best_model_params(self):
        return self.best_model_params


def keep_params_sign() -> base.GradientTransformation:
    """Modifies the updates to preserve parameter signs during optimization."""

    def init_fn(params):
        # Track original signs of parameters to handle zero cases appropriately
        initial_signs = jtu.tree_map(
            lambda p: jnp.sign(p) if p is not None else None, params, is_leaf=lambda x: x is None
        )
        del params

        return {"initial_signs": initial_signs}

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)

        initial_signs = state["initial_signs"]

        def _preserve_sign_with_path(path, p, u, original_sign):
            if u is None or original_sign is None:
                return u

            # Convert path to string for debugging
            path_str = "/".join(str(p) for p in path)

            # Determine the effective sign to enforce
            # If parameter is currently zero, use the original sign
            current_sign = jnp.where(p == 0, original_sign, jnp.sign(p))

            # Only enforce sign preservation for non-zero original signs
            enforce_sign = original_sign != 0

            # Check if update would change sign relative to the effective sign
            would_change_sign = (p + u) * current_sign < 0

            # Create mask for sign changes, preventing 0s from changing sign
            sign_change_mask = jnp.where(enforce_sign, would_change_sign, False)

            # Use jax.lax.cond for conditional debug printing
            # has_sign_change = jnp.any(sign_change_mask)

            # Debug when sign would change
            # jax.lax.cond(
            #     has_sign_change,
            #     lambda _: jax.debug.print(
            #         "Sign change detected for {path}: original_val={p}, update={u}, new_val={new_val}, " + "original_sign={orig_sign}, current_sign={curr_sign}, mask={mask}",
            #         path=path_str,
            #         p=p,
            #         u=u,
            #         new_val=p + u,
            #         orig_sign=original_sign,
            #         curr_sign=current_sign,
            #         mask=sign_change_mask,
            #     ),
            #     lambda _: None,
            #     operand=None,
            # )

            # When sign would change inappropriately, clip the update to prevent it
            clipped_update = jnp.where(sign_change_mask, -p + jnp.sign(p) * (100 * EPS), u)

            # Debug after clipping
            # jax.lax.cond(
            #     has_sign_change,
            #     lambda _: jax.debug.print("After clipping for {path}: clipped_update={clipped}, final_value={final}", path=path_str, clipped=clipped_update, final=p + clipped_update),
            #     lambda _: None,
            #     operand=None,
            # )

            return clipped_update

        # Partition parameters and updates
        params, static = eqx.partition(params, eqx.is_array)
        updates, updates_static = eqx.partition(updates, eqx.is_array)
        initial_signs, _ = eqx.partition(initial_signs, eqx.is_array)

        # Apply sign preservation with path tracking (param names in tree) and original signs
        updates = jtu.tree_map_with_path(
            _preserve_sign_with_path, params, updates, initial_signs, is_leaf=lambda x: x is None
        )

        # Combine back with static parts
        updates = eqx.combine(updates, updates_static)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def keep_params_within_bounds(bounds: dict) -> base.GradientTransformation:
    """
    Modifies the updates to keep specific parameters within bounds during optimization.

    Args:
        bounds (dict): A dictionary mapping parameter paths to (lower_bound, upper_bound) tuples.
                        Only parameters with specified bounds will be constrained.

    """

    def init_fn(params):
        # No special state is needed for this transformation
        return {}

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)

        def _clip_to_bounds_with_path(path, p, u):
            if u is None:
                return u

            # Convert path to string for checking if this parameter has bounds
            path_str = path[-1] if isinstance(path[-1], str) else str(path[-1])
            path_str = path_str.split(".")[-1]

            # Check if this parameter has specified bounds
            if path_str in bounds:
                lower_bound, upper_bound = bounds[path_str]

                # Compute the updated parameter value
                updated_param = p + u

                # Clip the updated parameter to the interval [lower_bound, upper_bound]
                clipped_param = jnp.clip(updated_param, lower_bound, upper_bound)

                # Adjust the update to reflect the clipping
                clipped_update = clipped_param - p

                # Optional debug information (commented out by default)
                # outside_bounds = (updated_param < lower_bound) | (updated_param > upper_bound)
                # has_clipping = jnp.any(outside_bounds)
                # jax.lax.cond(
                #     has_clipping,
                #     lambda _: jax.debug.print(
                #         "Bounds clipping for {path}: param={p}, update={u}, new_val={new_val}, " + "clipped_val={clipped}, bounds=[{lower}, {upper}",
                #         path=path_str,
                #         p=p,
                #         u=u,
                #         new_val=updated_param,
                #         clipped=clipped_param,
                #         lower=lower_bound,
                #         upper=upper_bound,
                #     ),
                #     lambda _: None,
                #     operand=None,
                # )

                return clipped_update
            else:
                # Parameter doesn't have bounds specified, leave the update unchanged
                return u

        # Partition parameters and updates
        params, static = eqx.partition(params, eqx.is_array)
        updates, updates_static = eqx.partition(updates, eqx.is_array)

        # Apply clipping only to the parameters with specified bounds
        updates = jtu.tree_map_with_path(_clip_to_bounds_with_path, params, updates, is_leaf=lambda x: x is None)

        # Combine back with static parts
        updates = eqx.combine(updates, updates_static)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
