"""
JAX Scenewalk Model Implementation

Federico D'Agostino, 2024

University of Tübingen

---
Original NumPy implementation by:

Lisa Schwetlick, 2019

University of Potsdam
"""

import functools
import warnings
from collections import OrderedDict
from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jaxtyping import Array, Float

PRECISION = jnp.float64
MAP_SIZE = 128
RNG_KEY = jax.random.PRNGKey(42)
EPS = jnp.finfo(PRECISION).eps.item()
ADD_EPS = (
    EPS  # To avoid division by zero in more places than the original implementation. Set to 0 for original behavior.
)

TRIAL_LENGTH = 8
SAMPLE_RATE = 1000

jax.config.update("jax_enable_x64", PRECISION == jnp.float64)
jax.numpy.set_printoptions(precision=16)

# Can be set to False to disable JIT compilation for debugging purposes.
USE_JIT = True


def conditional_jit(func):
    if USE_JIT:
        return eqx.filter_jit(func)
    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


PARAMETER_BOUNDS = {
    "omegaAttention": (0, 10000),
    "omfrac": (EPS, 1000),
    "omegaInhib": (0, 10000),
    "sigmaAttention": (0, 10000),
    "sigmaInhib": (0, 10000),
    "gamma": (0, 15),
    "lamb": (0, 15),
    "inhibStrength": (0, 10000),
    "zeta": (0, 1),  # default range, updated based on logged_z
    "cb_sd_x": (0, 10000),
    "cb_sd_y": (0, 10000),
    "first_fix_OmegaAttention": (0, 10000),
    "tau_pre": (0, 10000),
    "tau_post": (0, 10000),
    "sigmaShift": (0, 10000),
    "shift_size": (0, 10000),
    "omega_prevloc": (0, 10000),
    "omega_prevloc_frac": (EPS, 1000),
    "ompfactor": (0, 10),  # default range, updated based on logged_ompf
    "chi": (0.0001, 5),
    "t_alpha": (0, 10000),
    "t_beta": (-10000, EPS),
    "t_p": (EPS, 1000),
    "momentum_sigma": (0.01, 10),
    "saccadic_bias_weigth": (0.0, 1.0),
    "fixation_transition_rate": (0.01, 1.0),
    "long_saccade_threshold": (0.5, 15.0),
    "return_sigma": (0.01, 10),
    "fixation_transition_midpoint": (1, 10),
    "short_saccade_threshold": (0.1, 5.0),
    "gamma_base": (0, 15),
    "gamma_decay": (0, 5),
    "lamb_base": (0, 15),
    "lamb_decay": (0, 5),
    "alpha_left_bias": (1.0, 3.0),  # Range for leftward bias strength
    "alpha_vert_bias": (0.5, 1.5),  # Range for vertical vs horizontal bias strength
    "left_bias_decay_rate": (0.01, 2.0),  # Rate at which left bias decays with fixation number
    "dir_bias_growth_rate": (0.01, 2.0),  # Rate at which directional bias grows with fixation number
}


class ParameterPrior(NamedTuple):
    mean: float
    std: float
    lower_bound: float
    upper_bound: float


PARAMETER_PRIORS = {
    "omegaAttention": ParameterPrior(8, 11, EPS, 100),
    "omfrac": ParameterPrior(10, 2, EPS, 1000),  # FD
    "omega_prevloc_frac": ParameterPrior(10, 2, EPS, 1000),  # FD
    "omegaInhib": ParameterPrior(7, 130, EPS, 100),
    "sigmaAttention": ParameterPrior(6, 5, EPS, 30),
    "sigmaInhib": ParameterPrior(4, 4, EPS, 30),
    "gamma": ParameterPrior(1, 3, EPS, 5),
    "lamb": ParameterPrior(1, 10, EPS, 5),
    "inhibStrength": ParameterPrior(0.3, 1, EPS, 2),
    "zeta": ParameterPrior(-2, 1, -5, 0),
    "sigmaShift": ParameterPrior(4, 70, EPS, 30),
    "shift_size": ParameterPrior(0.5, 2, EPS, 3),
    "ompfactor": ParameterPrior(-1.2, 1, -5, 0),
    "first_fix_OmegaAttention": ParameterPrior(1.5, 2, EPS, 1000),  # FD
    "cb_sd_x": ParameterPrior(4, 2, EPS, 1000),  # FD
    "cb_sd_y": ParameterPrior(3, 2, EPS, 1000),  # FD
    "gamma_base": ParameterPrior(1, 0.5, EPS, 5),
    "gamma_decay": ParameterPrior(0.5, 0.2, 0, 5),
    "lamb_base": ParameterPrior(1, 0.5, EPS, 5),
    "lamb_decay": ParameterPrior(0.5, 0.2, 0, 5),
    # "alpha_left_bias": ParameterPrior(1.5, 0.5, 1.0, 3.0),  # Prior for leftward bias strength
    # "alpha_vert_bias": ParameterPrior(0.5, 0.5, 0.0, 1.5),  # Prior for vertical vs horizontal bias strength
}

# Parameters whose value can be expressed as a log in the model
LOGGABLE_PARAMETERS = ["zeta", "ompfactor", "inhibStrength"]


@dataclass
class Fixation:
    """
    Dataclass for concisely saving Fixation data.

    Attributes:
        n (int): The index or identifier for the fixation.
        pos (tuple): The (x, y) coordinates of the fixation position.
        next_pos (tuple): The (x, y) coordinates of the next fixation position.
        dur (float): The duration of the fixation in seconds.
        ll_temp (float): The temporal log-likelihood given by the model at this fixation.
        ll_spat (float): The spatial log-likelihood given by the model at this fixation.
        ll (float): The total log-likelihood assigned by the model to this fixation.
        att_map (jnp.ndarray): The attention map after the current fixation.
        inhib_map (jnp.ndarray): The inhibition map after the current fixation.
        main_map (jnp.ndarray): The saliency map at the end of the "main" phase of SceneWalk for the current fixation.
        final_map (jnp.ndarray): The "final" saliency map at the moment of saccade onset.
                                 If the model uses a presaccadic attention shift, this
                                 will be included and hence the map will be different from the
                                 main map.
        dens (jnp.ndarray): The internal fixation density map.
    """

    n: int
    pos: tuple
    next_pos: tuple
    dur: float
    ll_temp: float
    ll_spat: float
    ll: float
    att_map: jnp.ndarray
    inhib_map: jnp.ndarray
    main_map: jnp.ndarray
    final_map: jnp.ndarray
    dens: jnp.ndarray


class SceneWalk(eqx.Module):
    """
    SceneWalk model for predicting and simulating scanpaths during scene viewing.
    
    Based on the following papers:
    - ...
    - ...
    
    The model predicts fixation locations by maintaining dynamic attention and inhibition maps
    that evolve over a sequence of fixations, reflecting both bottom-up and top-down processes
    in eye movement control.
    
    Some of the most important features for the
    workflow are:
        - `whoami()` displays a string describing the configuration.
        - `get_param_list_order()` displays the exposed parameters that result \
        from the chosen configuration.
        - `update_params()` lets you set parameter values.
        - `get_scanpath_likelihood` returns a scan path likelihood given the \
        model.
        - `simulate_scanpath` returns a scan path simulated by the model.
    
    Core Initialization Parameters
    -----------------------------
    inhib_method : str
        How attention and inhibition maps are combined. Options:
        - "subtractive": Inhibition is subtracted from attention (through the `inhibStrength` parameter)
        - "divisive": Inhibition divides attention (through the `inhibStrength` parameter)
    
    att_map_init_type : str
        How the attention map is initialized. Options:
        - "zero": Uniform low-activation map 
        - "cb": Centered bias (parameters: `cb_sd_x`, `cb_sd_y`, `first_fix_OmegaAttention`)
    
    shifts : str
        Type of peri-saccadic attentional shifts to model. Options:
        - "off": No shifts, basic model
        - "pre": Presaccadic shift only (exposes the `tau_pre` parameter)
        - "post": Postsaccadic shift only (parameters: `sigmaShift`, `shift_size`, `tau_post`)
        - "both": Both pre- and postsaccadic shifts (requires all shift parameters)
    
    locdep_decay_switch : str
        Whether to use location-dependent attention decay. Options:
        - "off": Global decay rate
        - "on": Slower decay around previous fixation 
                (parameters: `omega_prevloc` or `omega_prevloc_frac`, depending on `coupled_facil`)
    
    omp : str
        Oculomotor potential preferences (systematic tendencies in eye movements to follow cardinal directions). 
        Options:
        - "off": No oculomotor bias
        - "add": Additive oculomotor potential (parameters: `chi`, `ompfactor`)
        - "mult": Multiplicative oculomotor potential (parameters: `chi`, `ompfactor`)
        - "attention": Cardinal attention bias applied multiplicatively to attention map 
                       (parameters: `chi`, `ompfactor`, `alpha_left_bias`, `alpha_vert_bias`, 
                        `left_bias_decay_rate`, `dir_bias_growth_rate`)
    
    dynamic_duration : bool
        Whether to model fixation durations:
        - True: Model durations based on image features (parameters: `t_alpha`, `t_beta`, `t_p`)
        - False: Fixed durations
    
    data_range : dict
        Dictionary specifying the range of the viewed scene in degrees of visual angle.
        Format: {'x': [min_x, max_x], 'y': [min_y, max_y]}
    
    Additional Initialization Parameters
    ----------------------------------
    inputs_in_deg : bool, default=True
        Whether input coordinates are in degrees (True) or pixels (False)
    
    warn_me : bool, default=False
        Whether to issue warnings during computation. Not available when using JIT.
    
    exponents : int, default=2
        Controls whether to use separate exponents for attention and inhibition:
        - 1: gamma=lamb (single exponent)
        - 2: independent gamma and lamb values
    
    coupled_oms : bool, default=False
        How to parameterize decay rates:
        - True: omegaInhib = omegaAttention/omfrac
        - False: omegaInhib is independent
    
    coupled_sigmas : bool, default=False
        How to parameterize Gaussian widths:
        - True: sigmaInhib = sigmaAttention
        - False: sigmaInhib is independent
    
    logged_cf : bool, default=False
        Whether `inhibStrength` is expressed in log space:
        - True: actual inhibStrength = 10^(parameter value)
        - False: used as provided
    
    logged_z : bool, default=False
        Whether `zeta` is expressed in log space:
        - True: actual zeta = 10^(parameter value)
        - False: used as provided
    
    logged_ompf : bool, default=False
        Whether `ompfactor` is expressed in log space:
        - True: actual ompfactor = 10^(parameter value)
        - False: used as provided
    
    coupled_facil : bool, default=False
        How to parameterize `omega_prevloc`:
        - True: omega_prevloc = omegaAttention/omega_prevloc_frac
        - False: omega_prevloc is independent
    
    estimate_times : bool, default=False
        Whether to expose tau parameters (peri-saccadic attention shifts durations) for optimization
    
    saclen_shift : bool, default=False
        Whether attentional shift size depends on saccade length:
        - True: Shift scales with saccade amplitude
        - False: Fixed shift size
    
    early_fix_exponents_scaling : bool, default=False
        Whether to use different exponents (gamma/lamb) for early fixations:
        - True: Use separate gamma/lamb values for the first four fixations
        - False: Use the same values for all fixations
    
    detail_mode : bool, default=False
        Whether to return detailed information about the evolving attention maps. Not available when using JIT.
    
    parameter_priors : dict, default=None
        Optional dictionary of ParameterPrior objects to override default priors
    
    kwargs_dict : dict, default=None
        Additional model attributes (legacy support for older SW versions)
    
    Notes
    -----
    - Time data is always given in seconds (not ms)
    - Position data is always given in degrees (if inputs_in_deg=True)
    - The model uses an internal grid representation (default 128x128)
    - Parameters and priors for log-transformed parameters should be expressed
    appropriately in log space
    
    
    Model Parameters Glossary
    -------------------------

    Basic Parameters (always required):
    - omegaAttention: Decay rate for the attention map
    - sigmaAttention: Width of the attention Gaussian (in degrees)
    - gamma: Exponent for the inhibition map
    - zeta: Additive noise parameter 
    - inhibStrength: Strength of inhibition 
    - foR_size: Size of the facilitation of return gaussian applied at the previous fixation location

    Conditionally Required Parameters:
    - omegaInhib: Decay rate for inhibition (required when coupled_oms=False)
    - omfrac: Fraction relating omegaAttention to omegaInhib (required when coupled_oms=True)
    - sigmaInhib: Width of inhibition Gaussian (required when coupled_sigmas=False)
    - lamb: Exponent for attention map (required when exponents=2)
    - omega_prevloc: Decay rate at previously fixated locations (required when locdep_decay_switch="on" and coupled_facil=False)
    - omega_prevloc_frac: Fraction relating omegaAttention to omega_prevloc (required when locdep_decay_switch="on" and coupled_facil=True)

    Center Bias Parameters (required when att_map_init_type="cb"):
    - cb_sd_x: Standard deviation of center bias in x-direction. Units are in pixels in the internal grid
    - cb_sd_y: Standard deviation of center bias in y-direction. Units are in pixels in the internal grid
    - first_fix_OmegaAttention: Special decay rate for the first fixation with center bias

    Shift Parameters:
    - tau_pre: Duration of presaccadic attention shift (required when shifts="pre" or shifts="both")
    - tau_post: Duration of postsaccadic attention shift (required when shifts="post" or shifts="both")
    - sigmaShift: Width of shifted attention Gaussian (required when shifts="post" or shifts="both")
    - shift_size: Size of attention shift (required when shifts="post" or shifts="both")

    Oculomotor Parameters (required when omp="add", omp="mult", or omp="attention"):
    - ompfactor: Strength of oculomotor bias
    - chi: Steepness of oculomotor potential or distance decay factor for attention bias
    
    Additional Oculomotor Parameters (required when omp="attention"):
    - alpha_left_bias: Strength of leftward bias for early fixations
    - alpha_vert_bias: Relative strength of vertical vs horizontal cardinal directions
    - left_bias_decay_rate: Rate at which left bias decays with fixation number
    - dir_bias_growth_rate: Rate at which directional bias grows with increasing fixation number

    Duration Parameters (required when dynamic_duration=True):
    - t_alpha: Base rate parameter for gamma distribution of durations
    - t_beta: Sensitivity of duration to activation (should be 0 when dynamic_duration=False)
    - t_p: Shape parameter for gamma distribution of durations

    Early Fixation Exponents (required when early_fix_exponents_scaling=True):
    - gamma_base: Base value for inhibition exponent
    - lamb_base: Base value for attention exponent
    - gamma_decay: Decay rate for inhibition exponent
    - lamb_decay: Decay rate for attention exponent
    """

    inhib_method: str = eqx.field(static=False)
    att_map_init_type: str = eqx.field(static=False)
    shifts: str = eqx.field(static=False)
    locdep_decay_switch: str = eqx.field(static=False)
    omp: str = eqx.field(static=False)
    dynamic_duration: bool = eqx.field(static=False)
    data_range: dict = eqx.field(static=False)
    MAP_SIZE: int = eqx.field(static=False)
    inputs_in_deg: bool = eqx.field(static=False, default=True)
    warn_me: bool = eqx.field(static=False, default=False)
    exponents: int = eqx.field(static=False, default=2)
    coupled_oms: bool = eqx.field(static=False, default=False)
    coupled_sigmas: bool = eqx.field(static=False, default=False)
    logged_cf: bool = eqx.field(static=False, default=False)
    logged_z: bool = eqx.field(static=False, default=False)
    logged_ompf: bool = eqx.field(static=False, default=False)
    coupled_facil: bool = eqx.field(static=False, default=False)
    estimate_times: bool = eqx.field(static=False, default=False)
    saclen_shift: bool = eqx.field(static=False, default=False)
    saccadic_momentum: bool = eqx.field(static=False, default=False)
    early_fix_exponents_scaling: bool = eqx.field(static=False, default=False)

    EPS: float = eqx.field(static=False)
    detail_mode: bool = eqx.field(static=False, default=False)
    trial_length: int = eqx.field(static=False)
    sample_rate: int = eqx.field(static=False)
    PARAMETER_BOUNDS: dict = eqx.field(static=False)
    PARAMETER_PRIORS: dict[str, ParameterPrior] = eqx.field(static=False)

    # Parameters
    omegaAttention: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    omfrac: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    _omegaInhib: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    sigmaAttention: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    sigmaInhib: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    gamma: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    _lamb: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    _inhibStrength: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    _zeta: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    sigmaShift: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    shift_size: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    first_fix_OmegaAttention: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    cb_sd: Float[Array, "2"] = eqx.field(converter=jnp.asarray, default=(jnp.nan, jnp.nan))
    tau_pre: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=50 / 1000)
    tau_post: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=100 / 1000)
    foR_size: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=2)
    _omega_prevloc: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)
    omega_prevloc_frac: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1)
    chi: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.3)
    _ompfactor: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=jnp.nan)

    t_alpha: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1 / 0.12144)
    t_beta: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0)
    t_p: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=3.23)

    # Saccadic momentum parameters
    saccadic_bias_weigth: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)
    momentum_sigma: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1)
    fixation_transition_rate: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.2)
    long_saccade_threshold: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=5.0)
    return_sigma: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1)
    fixation_transition_midpoint: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=3)
    short_saccade_threshold: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=2.0)

    # Attention and inhibition exponents decay parameters
    gamma_base: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1.0)
    gamma_decay: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)
    lamb_base: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1.0)
    lamb_decay: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)

    # Cardinal direction bias parameters
    alpha_left_bias: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=1.5)
    alpha_vert_bias: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)
    # New parameters for bias decay/growth
    left_bias_decay_rate: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)
    dir_bias_growth_rate: Float[Array, "1"] = eqx.field(converter=jnp.asarray, default=0.5)

    def __init__(
        self,
        inhib_method,
        att_map_init_type,
        shifts,
        locdep_decay_switch,
        omp,
        dynamic_duration,
        data_range,
        inputs_in_deg=True,
        warn_me=False,
        exponents=2,
        coupled_oms=False,
        coupled_sigmas=False,
        logged_cf=False,
        logged_z=False,
        logged_ompf=False,
        coupled_facil=False,
        estimate_times=False,
        saclen_shift=False,
        saccadic_momentum=False,
        early_fix_exponents_scaling=False,
        detail_mode=False,
        parameter_priors=None,
        kwargs_dict=None,
    ):
        self.inhib_method = inhib_method
        self.att_map_init_type = att_map_init_type
        self.shifts = shifts
        self.locdep_decay_switch = locdep_decay_switch
        self.omp = omp
        self.dynamic_duration = dynamic_duration
        self.data_range = data_range
        self.MAP_SIZE = MAP_SIZE
        self.inputs_in_deg = inputs_in_deg
        self.warn_me = warn_me
        self.exponents = exponents
        self.coupled_oms = coupled_oms
        self.coupled_sigmas = coupled_sigmas
        self.logged_cf = logged_cf
        self.logged_z = logged_z
        self.logged_ompf = logged_ompf
        self.coupled_facil = coupled_facil
        self.estimate_times = estimate_times
        self.saclen_shift = saclen_shift
        self.saccadic_momentum = saccadic_momentum
        self.early_fix_exponents_scaling = early_fix_exponents_scaling

        self.EPS = EPS

        self.detail_mode = detail_mode

        self.trial_length = TRIAL_LENGTH
        self.sample_rate = SAMPLE_RATE

        # add kwargs as object attributes
        if kwargs_dict is not None:
            self.__dict__.update(kwargs_dict)

        self.PARAMETER_BOUNDS = PARAMETER_BOUNDS

        # Default bounds need to be updated based on the model configuration
        self.update_parameter_bounds()

        self.PARAMETER_PRIORS = PARAMETER_PRIORS if parameter_priors is None else parameter_priors

    def __check_init__(self):
        all_valid, basics_not_none, out_of_bounds = self.check_params_in_bounds()

        if not all_valid and not basics_not_none:
            warnings.warn(
                "The current SceneWalk model is un-initialised. Please load or randomly init parameters.",
                UserWarning,
                stacklevel=2,
            )
            return

        if not all_valid:
            warnings.warn(f"Some parameters are out of bounds: {out_of_bounds}", RuntimeWarning, stacklevel=2)
        if not basics_not_none:
            warnings.warn("Some basic parameters are None", RuntimeWarning, stacklevel=2)
        if config_issues := self.check_params_for_config():
            raise ValueError(
                f"Invalid parameter configuration provided given your SceneWalk initialisation: \n{', '.join(f'{k}: {v}' for k, v in config_issues.items())}"
            )

    @classmethod
    def from_trained(cls, hyperparams, trained_params):
        """
        Create a new SceneWalk model instance from a set of trained parameters.

        Parameters
        ----------
        hyperparams : dict
            A dictionary containing the hyperparameters of the model.
        trained_params : dict
            A dictionary containing the trained parameters of the model.

        Returns
        -------
        SceneWalk
            A new SceneWalk model instance with the provided parameters.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            model = cls(**hyperparams)
        model = load_model_parameters(model, trained_params)
        return model

    def update_parameter_bounds(self):
        """
        Update the parameter bounds based on the model configuration for logged parameters.
        Also updates the bounds for the private parameters.
        """
        if self.logged_z:
            self.PARAMETER_BOUNDS["zeta"] = (10**-20, 10**0)
            self.PARAMETER_BOUNDS["_zeta"] = (-20, 0)
        else:
            self.PARAMETER_BOUNDS["_zeta"] = self.PARAMETER_BOUNDS["zeta"]

        if self.logged_ompf:
            self.PARAMETER_BOUNDS["ompfactor"] = (10**-20, 10**1)
            self.PARAMETER_BOUNDS["_ompfactor"] = (-20, 1)
        else:
            self.PARAMETER_BOUNDS["_ompfactor"] = self.PARAMETER_BOUNDS["ompfactor"]

        if self.logged_cf:
            self.PARAMETER_BOUNDS["inhibStrength"] = (10**-20, 10**1)
            self.PARAMETER_BOUNDS["_inhibStrength"] = (-20, 1)
        else:
            self.PARAMETER_BOUNDS["_inhibStrength"] = self.PARAMETER_BOUNDS["inhibStrength"]

    @property
    def att_map_init(self):
        att_map_init_funcs = {"zero": self.initialize_map_unif, "cb": self.initialize_center_bias}
        return att_map_init_funcs[self.att_map_init_type]

    @property
    def combine(self):
        inhib_funcs = {"subtractive": self.combine_subtractive, "divisive": self.combine_divisive}
        return inhib_funcs[self.inhib_method]

    @property
    def evolve_maps(self):
        evolve_maps_funcs = {
            "off": self.evolve_maps_main,
            "pre": self.evolve_maps_presac,
            "post": self.evolve_maps_postsac,
            "both": self.evolve_maps_both,
        }
        return evolve_maps_funcs[self.shifts]

    @property
    def differential_time_att(self):
        att_decay_funcs = {"off": self.differential_time_att_org, "on": self.differential_time_att_locdep_round}
        return att_decay_funcs[self.locdep_decay_switch]

    @property
    def cb_sd_x(self):
        return self.cb_sd[0]

    @property
    def cb_sd_y(self):
        return self.cb_sd[1]

    # --------------------------------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------------------------------
    def check_params_in_bounds(self):
        """
        Checks if all required parameters are not None and not NaN, and if all parameters are within specified bounds.

        Returns:
            tuple: A tuple containing three elements:
                - bool: True if all parameters are within bounds, False otherwise.
                - bool: True if all required parameters are not None and not NaN, False otherwise.
                - list: A list of parameters that are out of bounds.
        """
        basics_not_none = all(
            getattr(self, param) is not None and not jnp.isnan(getattr(self, param))
            for param in [
                "omegaAttention",
                "omegaInhib",
                "sigmaAttention",
                "sigmaInhib",
                "gamma",
                "lamb",
                "inhibStrength",
                "zeta",
            ]
        )
        if not basics_not_none:
            return False, False, []

        in_bounds = {
            param: (lower <= getattr(self, param) <= upper).item()
            for param, (lower, upper) in self.PARAMETER_BOUNDS.items()
            if hasattr(self, param) and getattr(self, param) is not None and not jnp.isnan(getattr(self, param))
        }

        all_valid = all(in_bounds.values())
        out_of_bounds = [param for param, valid in in_bounds.items() if valid is False]

        return all_valid, basics_not_none, out_of_bounds

    def check_params_for_config(self):
        """
        Checks whether all necessary parameters are present for the current
        configuration.

        Returns
        -------
        dict
            A dictionary with informative messages about any missing or invalid parameters.
            If all parameters are valid, an empty dictionary is returned.
        """
        issues = {}

        # basic params
        if self.omegaAttention is None or jnp.isnan(self.omegaAttention):
            issues["omegaAttention"] = "Parameter 'omegaAttention' must be set."
        if self.omegaInhib is None or jnp.isnan(self.omegaInhib):
            issues["omegaInhib"] = "Parameter 'omegaInhib' must be set."
        if self.sigmaAttention is None or jnp.isnan(self.sigmaAttention):
            issues["sigmaAttention"] = "Parameter 'sigmaAttention' must be set."
        if self.sigmaInhib is None or jnp.isnan(self.sigmaInhib):
            issues["sigmaInhib"] = "Parameter 'sigmaInhib' must be set."
        if self.gamma is None or jnp.isnan(self.gamma):
            issues["gamma"] = "Parameter 'gamma' must be set."
        if self.lamb is None or jnp.isnan(self.lamb):
            issues["lamb"] = "Parameter 'lamb' must be set."
        if self.inhibStrength is None or jnp.isnan(self.inhibStrength):
            issues["inhibStrength"] = "Parameter 'inhibStrength' must be set."
        if self.zeta is None or jnp.isnan(self.zeta):
            issues["zeta"] = "Parameter 'zeta' must be set."

        # Center bias
        if self.att_map_init_type == "cb":
            if len(self.cb_sd) != 2:
                issues["cb_sd"] = "Parameter 'cb_sd' must be a tuple of length 2."
            if self.first_fix_OmegaAttention is None or jnp.isnan(self.first_fix_OmegaAttention):
                issues["first_fix_OmegaAttention"] = "Parameter 'first_fix_OmegaAttention' must be set."

        if self.shifts in ["pre", "both"] and (self.tau_pre is None or jnp.isnan(self.tau_pre)):
            issues["tau_pre"] = "Parameter 'tau_pre' must be set."

        if self.shifts in ["post", "both"]:
            if self.sigmaShift is None or jnp.isnan(self.sigmaShift):
                issues["sigmaShift"] = "Parameter 'sigmaShift' must be set."
            if self.shift_size is None or jnp.isnan(self.shift_size):
                issues["shift_size"] = "Parameter 'shift_size' must be set."
            if self.tau_post is None or jnp.isnan(self.tau_post):
                issues["tau_post"] = "Parameter 'tau_post' must be set."

        if self.locdep_decay_switch == "on" and (self.omega_prevloc is None or jnp.isnan(self.omega_prevloc)):
            issues["omega_prevloc"] = "Parameter 'omega_prevloc' must be set."

        if self.dynamic_duration:
            if self.t_alpha is None or jnp.isnan(self.t_alpha):
                issues["t_alpha"] = "Parameter 't_alpha' must be set."
            if self.t_beta is None or jnp.isnan(self.t_beta):
                issues["t_beta"] = "Parameter 't_beta' must be set."
            if self.t_p is None or jnp.isnan(self.t_p):
                issues["t_p"] = "Parameter 't_p' must be set."
        if self.early_fix_exponents_scaling:
            if self.gamma_base is None or jnp.isnan(self.gamma_base):
                issues["gamma_base"] = "Parameter 'gamma_base' must be set."
            if self.gamma_decay is None or jnp.isnan(self.gamma_decay):
                issues["gamma_decay"] = "Parameter 'gamma_decay' must be set."
            if self.exponents == 2:
                if self.lamb_base is None or jnp.isnan(self.lamb_base):
                    issues["lamb_base"] = "Parameter 'lamb_base' must be set."
                if self.lamb_decay is None or jnp.isnan(self.lamb_decay):
                    issues["lamb_decay"] = "Parameter 'lamb_decay' must be set."

        if self.omp == "attention":
            if self.chi is None or jnp.isnan(self.chi):
                issues["chi"] = "Parameter 'chi' must be set for attention OMP mode."
            if self.ompfactor is None or jnp.isnan(self.ompfactor):
                issues["ompfactor"] = "Parameter 'ompfactor' must be set for attention OMP mode."
            if self.alpha_left_bias is None or jnp.isnan(self.alpha_left_bias):
                issues["alpha_left_bias"] = "Parameter 'alpha_left_bias' must be set for attention OMP mode."
            if self.alpha_vert_bias is None or jnp.isnan(self.alpha_vert_bias):
                issues["alpha_vert_bias"] = "Parameter 'alpha_vert_bias' must be set for attention OMP mode."
            if self.left_bias_decay_rate is None or jnp.isnan(self.left_bias_decay_rate):
                issues["left_bias_decay_rate"] = "Parameter 'left_bias_decay_rate' must be set for attention OMP mode."
            if self.dir_bias_growth_rate is None or jnp.isnan(self.dir_bias_growth_rate):
                issues["dir_bias_growth_rate"] = "Parameter 'dir_bias_growth_rate' must be set for attention OMP mode."
        elif self.omp in ["add", "mult"]:
            if self.chi is None or jnp.isnan(self.chi):
                issues["chi"] = "Parameter 'chi' must be set for add/mult OMP mode."
            if self.ompfactor is None or jnp.isnan(self.ompfactor):
                issues["ompfactor"] = "Parameter 'ompfactor' must be set for add/mult OMP mode."

        return issues

    def whoami(self):
        """Returns the model identity as a string"""
        id_str = ""
        id_str += f"{str(self.inhib_method).capitalize()} SceneWalk model, "
        id_str += f"initialized with {self.att_map_init_type} activation, "
        id_str += f"in {str(self.exponents)} exponents mode"
        if self.shifts in ["pre", "both"]:
            id_str += ", using a presaccadic shift"
        if self.shifts in ["post", "both"]:
            id_str += ", using a postsaccadic shift"
        if self.locdep_decay_switch == "on":
            id_str += ", using location dependent attention decay"
        if self.coupled_oms:
            id_str += ", with om_i as a fraction"
        if self.coupled_sigmas:
            id_str += ", with coupled sigmas"
        if self.coupled_facil:
            id_str += ", with coupled facilitation"
        if self.logged_cf:
            id_str += ", with logged cf"
        if self.logged_z:
            id_str += ", with logged z"
        if self.saclen_shift:
            id_str += ", with eta=saclen"
        if self.omp != "off":
            id_str += ", with omp"
        if self.logged_ompf:
            id_str += "logged"
        if self.dynamic_duration:
            id_str += ", with dynamic duration"
        if self.saccadic_momentum:
            id_str += ", with bidirectional saccadic bias (momentum and return saccades)"
        if self.early_fix_exponents_scaling:
            id_str += ", with early fixations exponents scaling"
        return f"{id_str}."

    def clear_params(self):
        raise NotImplementedError("This function is not implemented for the JAX version of SceneWalk")

    def get_params(self):
        """Returns the current parameters as a dictionary"""
        param_dict = OrderedDict(
            {
                "omegaAttention": self.omegaAttention,
                "omegaInhib": self.omegaInhib,
                "sigmaAttention": self.sigmaAttention,
                "sigmaInhib": self.sigmaInhib,
                "gamma": self.gamma,
                "foR_size": self.foR_size,
            }
        )
        if self.exponents == 2:
            param_dict["lamb"] = self.lamb
        param_dict["inhibStrength"] = self._inhibStrength
        param_dict["zeta"] = self._zeta

        if self.shifts in ["post", "both"]:
            param_dict["sigmaShift"] = self.sigmaShift
            param_dict["shift_size"] = self.shift_size
        # Center bias
        if self.att_map_init_type == "cb":
            param_dict["first_fix_OmegaAttention"] = self.first_fix_OmegaAttention
            param_dict["cb_sd_x"] = self.cb_sd[0]
            param_dict["cb_sd_y"] = self.cb_sd[1]
        if self.locdep_decay_switch == "on":
            param_dict["omega_prevloc"] = self.omega_prevloc
        if self.coupled_facil:
            param_dict["omega_prevloc_frac"] = self.omega_prevloc_frac
        if self.estimate_times:
            param_dict["tau_pre"] = self.tau_pre
            param_dict["tau_post"] = self.tau_post
        if self.omp != "off":
            param_dict["chi"] = self.chi
            param_dict["ompfactor"] = self._ompfactor
            if self.omp == "attention":
                param_dict["alpha_left_bias"] = self.alpha_left_bias
                param_dict["alpha_vert_bias"] = self.alpha_vert_bias
                param_dict["left_bias_decay_rate"] = self.left_bias_decay_rate
                param_dict["dir_bias_growth_rate"] = self.dir_bias_growth_rate
        if self.dynamic_duration:
            param_dict["t_alpha"] = self.t_alpha
            param_dict["t_beta"] = self.t_beta
            param_dict["t_p"] = self.t_p
        if self.coupled_oms:
            param_dict["omfrac"] = self.omfrac
        if self.saccadic_momentum:
            param_dict["momentum_sigma"] = self.momentum_sigma
            param_dict["saccadic_bias_weigth"] = self.saccadic_bias_weigth
            param_dict["fixation_transition_rate"] = self.fixation_transition_rate
            param_dict["long_saccade_threshold"] = self.long_saccade_threshold
            # Add new direction change parameters
            param_dict["return_sigma"] = self.return_sigma
            param_dict["fixation_transition_midpoint"] = self.fixation_transition_midpoint
            param_dict["short_saccade_threshold"] = self.short_saccade_threshold

        if self.early_fix_exponents_scaling:
            param_dict["gamma_base"] = self.gamma_base
            param_dict["gamma_decay"] = self.gamma_decay

            # Only include lamb parameters if exponents=2
            if self.exponents == 2:
                param_dict["lamb_base"] = self.lamb_base
                param_dict["lamb_decay"] = self.lamb_decay

        return param_dict

    def get_private_params(self):
        private_params_dict = OrderedDict(
            {
                "_inhibStrength": self._inhibStrength,
                "_zeta": self._zeta,
            }
        )
        if self.omp != "off":
            private_params_dict["_ompfactor"] = self._ompfactor

        if not self.coupled_oms:
            private_params_dict["_omegaInhib"] = self._omegaInhib

        if not self.coupled_facil:
            private_params_dict["_omega_prevloc"] = self._omega_prevloc

        if self.exponents == 2:
            private_params_dict["_lamb"] = self._lamb

        if self.att_map_init_type == "cb":
            private_params_dict["cb_sd"] = self.cb_sd

        return private_params_dict

    def _not_trainable_params(self):
        # Get all trainable parameters
        trainable_params = self.get_params() | self.get_private_params()

        # Get all model parameters that are fields (not properties or methods)
        all_params = {
            field.name for field in fields(self) if not isinstance(getattr(self.__class__, field.name, None), property)
        }

        # Parameters that are in the model but not in trainable_params
        not_trainable = [param for param in all_params if param not in trainable_params]

        return not_trainable

    def get_param_list_order(self):
        """Returns the names and order of the parameters required by the
        current model configuration"""
        warnings.warn(
            "get_param_list_order is deprecated and will be removed in a future version. Use get_params instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        param_names = []
        if self.coupled_oms:
            param_names.extend(["omegaAttention", "omfrac"])
        else:
            param_names.extend(["omegaAttention", "omegaInhib"])
        if self.coupled_sigmas:
            param_names.extend(["sigmaAttention"])
        else:
            param_names.extend(["sigmaAttention", "sigmaInhib"])
        param_names.extend(["gamma"])
        if self.exponents == 2:
            param_names.extend(["lamb"])
        param_names.extend(["inhibStrength", "zeta"])
        if self.shifts in ["post", "both"]:
            param_names.extend(["sigmaShift", "shift_size"])
        if self.att_map_init_type == "cb":
            param_names.extend(["first_fix_OmegaAttention", "cb_sd_x", "cb_sd_y"])
        if self.locdep_decay_switch == "on":
            if self.coupled_facil:
                param_names.extend(["omega_prevloc_frac"])
            else:
                param_names.extend(["omega_prevloc"])
        if self.estimate_times:
            param_names.extend(["tau_pre", "tau_post"])
        if self.omp != "off":
            param_names.extend(["chi", "ompfactor"])
            if self.omp == "attention":
                param_names.extend(
                    ["alpha_left_bias", "alpha_vert_bias", "left_bias_decay_rate", "dir_bias_growth_rate"]
                )
        if self.dynamic_duration:
            param_names.extend(["t_alpha", "t_beta", "t_p"])
        if self.saccadic_momentum:
            param_names.extend(
                [
                    "momentum_sigma",
                    "saccadic_bias_weigth",
                    "fixation_transition_rate",
                    "long_saccade_threshold",
                    "return_sigma",
                    "fixation_transition_midpoint",
                    "short_saccade_threshold",
                ]
            )

        if self.early_fix_exponents_scaling:
            param_names.extend(
                [
                    "gamma_base",
                    "gamma_decay",
                ]
            )
            if self.exponents == 2:
                param_names.extend(
                    [
                        "lamb_base",
                        "lamb_decay",
                    ]
                )

        return param_names

    def _get_all_fields(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @conditional_jit
    def convert_deg_to_px_fixed(self, dat, data_range_min, data_range_max, grid_sz, cutoff):
        data_range_diff = data_range_max - data_range_min

        dat_px = ((dat - data_range_min) / data_range_diff) * grid_sz
        dat_px = jnp.floor(dat_px)

        if cutoff:
            dat_px = jnp.clip(dat_px, 0, grid_sz - 1)
            dat_px = jnp.maximum(dat_px, 0)

        return dat_px

    @conditional_jit
    def convert_deg_to_px_not_fixed(self, dat, data_range_min, data_range_max, grid_sz):
        data_range_diff = data_range_max - data_range_min

        dat_px = dat / data_range_diff * grid_sz
        dat_px = jnp.maximum(dat_px, 0.1)

        return dat_px

    def convert_deg_to_px(self, dat, dim, fix=False, cutoff=True, grid_sz=None):
        if grid_sz is None:
            grid_sz = self.MAP_SIZE  # [{"x": 1, "y": 0}[dim]]

        # Use a boolean flag to determine if fast_return applies
        fast_return = jnp.any(jnp.logical_not(self.inputs_in_deg) | jnp.any(dat == jnp.nan))

        # Use lax.cond to make this branch JIT-compatible
        result = jax.lax.cond(
            fast_return,
            lambda _: jnp.floor(dat) if fix else dat,
            lambda _: self._handle_deg_to_px_conversion(dat, dim, fix, cutoff, grid_sz),
            operand=None,
        )

        return result

    @conditional_jit
    def _handle_deg_to_px_conversion(self, dat, dim, fix, cutoff, grid_sz):
        # Precomputed data range values
        data_range_min = jnp.min(jnp.asarray(self.data_range[dim]))
        data_range_max = jnp.max(jnp.asarray(self.data_range[dim]))

        if fix:
            return self.convert_deg_to_px_fixed(dat, data_range_min, data_range_max, grid_sz, cutoff)
        else:
            return self.convert_deg_to_px_not_fixed(dat, data_range_min, data_range_max, grid_sz)

    def convert_px_to_deg(self, dat, dim, grid_sz=None):
        """
        Converts pixel values to degrees on the grid.

        Parameters
        ----------
        dat : int or float
            number to convert in pixels
        dim : {'x' or 'y'}
            dimension along which to convert
        grid_sz : int
            enables setting the grid size to another size than the internal \
            model size

        Returns
        -------
        float
            degree value
        """
        if grid_sz is None:
            grid_sz = self.MAP_SIZE  # [{"x": 1, "y": 0}[dim]]
        if self.inputs_in_deg:
            return dat
        return (dat / grid_sz) * (jnp.max(self.data_range[dim]) - jnp.min(self.data_range[dim])) + jnp.min(
            self.data_range[dim]
        )

    def get_unit_vector(self, point1, point2):
        """
        Get the unit vector between 2 points.
        point1 is the origin. Vector goes toward point2.

        Parameters
        ----------
        point1, point2 : iterables of shape (2,)
            Points between which to find the unit vector.

        Returns
        -------
        tuple
            A tuple containing:
            - unit_vector (jax.numpy.ndarray): Unit vector [unit vector x, unit vector y].
            - vec_magnitude (float): Magnitude of the vector.
        """
        # Delta vector
        d_x = point2[0] - point1[0]
        d_y = point2[1] - point1[1]
        vec_magnitude = jnp.sqrt((d_x**2) + (d_y**2))

        # Handle case when vec_magnitude is zero
        unit_vector = jnp.where(
            vec_magnitude == 0,
            jnp.array([0.0, 0.0]),
            jnp.array([d_x / vec_magnitude, d_y / vec_magnitude]),
        )

        # Issue warning if enabled and vec_magnitude is zero
        if self.warn_me and vec_magnitude == 0:
            warnings.warn("No movement between two fixations", stacklevel=2)

        return unit_vector, vec_magnitude

    def simulate_durations(self, amount, rng_key=None):
        """
        generate durations from gamma distribtion.

        Parameters
        ----------
        amount : int
            number of durations to output

        Returns
        -------
        array
            vector of durations
        """
        durations = []
        for _ in range(amount):
            duration = int(jnp.floor(dist.Gamma(100, 0.4).sample(key=rng_key))) / 1000
            durations.append(duration)
        return durations

    def empirical_fixation_density(self, x_locs_deg, y_locs_deg):
        """
        Calculates the empirical fixation density given fixation points and
        range information using the "scott" bandwidth (as in Schütt et al, 2017)

        Parameters
        ----------
        x_locs_deg, y_locs_deg : float
            coordinates in degrees

        Returns
        -------
        int
            x coordinates in px
        int
            y coordinates in px
        array
            fixation density map
        """
        from scipy.stats import kde

        # reduce grid to 128x128
        # watch out these are NOT indexes!!
        x_px = [self.convert_deg_to_px(x, "x", fix=True) for x in x_locs_deg]
        y_px = [self.convert_deg_to_px(y, "y", fix=True) for y in y_locs_deg]
        # smooth
        k = kde.gaussian_kde([x_px, y_px], bw_method="scott")
        # resolution of image grid is 128x128
        ii, jj = self.MAP_SIZE
        xi, yi = jnp.mgrid[0:ii, 0:jj]
        xi = xi.astype(PRECISION)
        yi = yi.astype(PRECISION)
        # apply smoothed data to grid
        zi = k(jnp.vstack([yi.flatten(), xi.flatten()]))
        # normalize
        zi = zi / jnp.sum(zi)
        zi = zi.reshape(xi.shape)
        return x_px, y_px, zi

    def fixation_picker_max(self, likelihood_map, get_lik=False, rng_key=None):
        """
        Picks the next fixation location according to the maximum activation
        value (deterministic).

        Parameters
        ----------
        likelihood_map : array
            (usually 128x128) map from which to pick
        get_lik : bool
            if true, return the point's likelihood value
        rng_key : jax.random.PRNGKey, optional
            Random key for sampling. If None, uses a default key.

        Returns
        -------
        float
            x value in degrees
        float
            y value in degrees
        float (optional)
            likelihood
        """
        if rng_key is None:
            rng_key = RNG_KEY
        max_locs = jnp.argwhere(likelihood_map == jnp.max(likelihood_map))
        i, j = max_locs[dist.Categorical(jnp.ones(len(max_locs))).sample(key=rng_key)]
        x_deg = self.convert_px_to_deg(j, "x")
        y_deg = self.convert_px_to_deg(i, "y")
        if get_lik:
            lik = jnp.log2(jnp.array(likelihood_map[i, j], dtype=PRECISION))
            return (x_deg, y_deg, lik)
        return (x_deg, y_deg)

    def fixation_picker_stoch(self, likelihood_map, get_lik=False, rng_key=None):
        """
        Picks the next fixation location according to the cumulative probability
        method (linear selection algorithm)

        Parameters
        ----------
        likelihood_map : array
            (usually 128x128) map from which to pick
        get_lik : bool
            if true, return the point's likelihood value

        Returns
        -------
        float
            x value in degrees
        float
            y value in degrees
        float (optional)
            likelihood
        """

        likelihood_map = likelihood_map / (jnp.sum(likelihood_map) + EPS)
        r = dist.Categorical(likelihood_map.flatten()).sample(key=rng_key)
        i, j = jnp.unravel_index(r, likelihood_map.shape)
        x_deg = self.convert_px_to_deg(j, "x").astype(PRECISION)
        y_deg = self.convert_px_to_deg(i, "y").astype(PRECISION)
        if get_lik:
            lik = jnp.log2(jnp.array(likelihood_map[i, j], dtype=PRECISION))
            return (x_deg, y_deg, lik)
        return (x_deg, y_deg)

    # --------------------------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------------------------

    @conditional_jit
    def initialize_map_unif(self):
        """
        Initializes a map with near 0 activation everywhere

        Returns
        -------
        array
            initial starting map
        """
        map_init = self.EPS * jnp.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=PRECISION)
        return map_init

    def radial_distance(self, x, sigma_x, y, sigma_y):
        # Transpose fixes weird meshgrid thing
        return ((((self._xx - x) ** 2) / (sigma_x**2)) + (((self._yy - y) ** 2) / (sigma_y**2))).T

    @conditional_jit
    def gaussian_pdf(self, sigma_x, sigma_y, exponent):
        return (1 / (2 * jnp.pi * sigma_x * sigma_y + ADD_EPS)) * jnp.exp(-(exponent / (2 * (1**2))))

    def initialize_center_bias(self):
        """
        Initializes a map with with a central gaussian

        Returns
        -------
        array
            initial starting map
        """
        cb_sd_x = self.convert_deg_to_px(jnp.array(self.cb_sd[0]), "x")
        cb_sd_y = self.convert_deg_to_px(jnp.array(self.cb_sd[1]), "y")
        mu = jnp.array([64, 64])
        rad = self.radial_distance(mu[0], cb_sd_x, mu[1], cb_sd_y)
        mapAtt_init = self.gaussian_pdf(cb_sd_x, cb_sd_y, rad)
        # normalize
        mapAtt_init = mapAtt_init / jnp.sum(mapAtt_init)
        return mapAtt_init

    def make_attention_gauss(self, fixs_x, fixs_y):
        """
        make gaussian window at fixation point for attention

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        array
            gaussian attention map (usually 128x128)
        """
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        sigmaAttention_x = self.convert_deg_to_px(self.sigmaAttention, "x")
        sigmaAttention_y = self.convert_deg_to_px(self.sigmaAttention, "y")
        fix_x = self.convert_deg_to_px(fix_x, "x", fix=True)
        fix_y = self.convert_deg_to_px(fix_y, "y", fix=True)
        # equation 5
        rad = self.radial_distance(fix_x, sigmaAttention_x, fix_y, sigmaAttention_y)
        gaussAttention = self.gaussian_pdf(sigmaAttention_x, sigmaAttention_y, rad)

        return gaussAttention

    def make_attention_gauss_post_shift(self, fixs_x, fixs_y, get_loc=False):
        """
        make gaussian window at fixation point for attention during the
        post-saccadic shift

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's y
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        array
            gaussian attention map (usually 128x128)
        """
        fix_x_prev = fixs_x[0]
        fix_y_prev = fixs_y[0]
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        # post saccadic attention shift
        sigmaShift_x = self.convert_deg_to_px(self.sigmaShift, "x")
        sigmaShift_y = self.convert_deg_to_px(self.sigmaShift, "y")
        # get unit vector in degrees
        u, mag = self.get_unit_vector([fix_x_prev, fix_y_prev], [fix_x, fix_y])

        shift_by = mag * self.shift_size if self.saclen_shift else self.shift_size
        shift_loc_x = fix_x + (u[0] * shift_by)
        shift_loc_y = fix_y + (u[1] * shift_by)
        shift_loc_x_px = self.convert_deg_to_px(shift_loc_x, "x", fix=True, cutoff=False)
        shift_loc_y_px = self.convert_deg_to_px(shift_loc_y, "y", fix=True, cutoff=False)

        rad = self.radial_distance(shift_loc_x_px, sigmaShift_x, shift_loc_y_px, sigmaShift_y)
        gaussAttention_shift = self.gaussian_pdf(sigmaShift_x, sigmaShift_y, rad)

        gaussAttention_shift = jnp.where(
            jnp.isclose(jnp.sum(gaussAttention_shift), 0),
            jnp.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=PRECISION),
            gaussAttention_shift,
        )

        if self.warn_me:
            warning_msg = str(
                [
                    "shift gauss 0. tried to put a gaussian at",
                    shift_loc_x_px,
                    shift_loc_y_px,
                    "with ",
                    sigmaShift_x,
                    sigmaShift_y,
                ]
            )
            warnings.warn(warning_msg)

        if get_loc:
            return gaussAttention_shift, shift_loc_x_px, shift_loc_y_px
        return gaussAttention_shift

    @conditional_jit
    def combine_att_fixdens(self, gaussAttention, fix_density_map):
        """
        add empirical density information to gaussian attention mask

        Parameters
        ----------
        gaussAttention : array
            attention mask (usually 128x128)
        fix_density_map : array
            fixation density map (usually 128x128)

        Returns
        -------
        array
            combined attention map (usually 128x128)
        """

        salFixation = jnp.multiply(fix_density_map, gaussAttention)

        salFixation = salFixation / (jnp.sum(salFixation) + ADD_EPS)
        return salFixation

    def make_inhib_gauss(self, fixs_x, fixs_y):
        """
        make gaussian window at fixation point for inhibition

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's y
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y

        Returns
        -------
        array
            gaussian inhibition map
        """
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        sigmaInhib_x = self.convert_deg_to_px(self.sigmaInhib, "x")
        sigmaInhib_y = self.convert_deg_to_px(self.sigmaInhib, "y")
        fix_x = self.convert_deg_to_px(fix_x, "x", fix=True)
        fix_y = self.convert_deg_to_px(fix_y, "y", fix=True)
        # equation 5
        # T fixes weird meshgrid thing
        rad = (((self._xx - fix_x) ** 2) / (sigmaInhib_x**2) + ((self._yy - fix_y) ** 2) / (sigmaInhib_y**2)).T
        return (1 / (2 * jnp.pi * sigmaInhib_x * sigmaInhib_y + ADD_EPS)) * jnp.exp(-(rad / (2 * (1**2))))

    @conditional_jit
    def combine_subtractive(self, mapAtt, mapInhib, nth):
        """
        Combine attention and inhibition in a subtractive way

        Parameters
        ----------
        mapAtt : array
            attention map (128x128)
        mapInhib : array
            inhibition map (128x128)
        nth : int
            fixation number, used for early fixation exponent scaling

        Returns
        -------
        array
            combined map (128x128)
        """
        # Get exponents based on fixation index if early fixation exponent scaling is enabled
        if self.early_fix_exponents_scaling:
            # Calculate decayed exponents using exponential decay
            # The decay is relative to the base value (gamma/lamb)
            gamma = self.gamma_base + (self.gamma - self.gamma_base) * jnp.exp(-self.gamma_decay * (nth - 1))

            # Get lamb - either from decayed value or use gamma if exponents=1
            lamb = (
                self.lamb_base + (self.lamb - self.lamb_base) * jnp.exp(-self.lamb_decay * (nth - 1))
                if self.exponents == 2
                else gamma
            )
        else:
            # Use the standard exponents
            gamma = self.gamma
            lamb = self.lamb

        # equation 10
        mapAttPower = mapAtt**lamb
        mapAttPowerNorm = mapAttPower / (jnp.sum(mapAttPower) + ADD_EPS)

        mapInhibPower = mapInhib**gamma
        mapInhibPowerNorm = mapInhibPower / (jnp.sum(mapInhibPower) + ADD_EPS)

        u = mapAttPowerNorm - self.inhibStrength * mapInhibPowerNorm
        return u

    @conditional_jit
    def combine_divisive(self, mapAtt, mapInhib, nth):
        """
        Combine attention and inhibition in a divisive way

        Parameters
        ----------
        mapAtt : array
            attention map (128x128)
        mapInhib : array
            inhibition map (128x128)
        nth : int
            fixation number, used for early fixation exponent scaling

        Returns
        -------
        array
            combined map (128x128)
        """
        # Get exponents based on fixation index if early fixation exponent scaling is enabled
        if self.early_fix_exponents_scaling:
            # Calculate decayed exponents using exponential decay
            # The decay is relative to the base value (gamma/lamb)
            gamma = self.gamma_base + (self.gamma - self.gamma_base) * jnp.exp(-self.gamma_decay * (nth - 1))

            # Get lamb - either from decayed value or use gamma if exponents=1
            lamb = (
                self.lamb_base + (self.lamb - self.lamb_base) * jnp.exp(-self.lamb_decay * (nth - 1))
                if self.exponents == 2
                else gamma
            )
        else:
            # Use the standard exponents
            gamma = self.gamma
            lamb = self.lamb

        mapAttPower = mapAtt**lamb
        mapInhibPower = mapInhib**gamma

        # Normalizes ihib strength parameter to refer to how strong inhib is
        # compared to random activation
        inhibStrengthNorm = self.inhibStrength / jnp.array((self.MAP_SIZE, self.MAP_SIZE)).prod()

        inhibStrength_power = jnp.array((inhibStrengthNorm**gamma), dtype=PRECISION)
        weighted_mapInhib = inhibStrength_power + mapInhibPower
        return jnp.divide(mapAttPower, weighted_mapInhib)

    @conditional_jit
    def differential_time_basic(self, duration, current_map, prev_map, omega):
        """
        evolve the maps over time, given a decay paramter

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention or inhibition map
        prev_map : float
            previous attention or inhibition map
        omega : float
            decay parameter to use

        Returns
        -------
        array
            evolved map
        """
        current_map = current_map / (jnp.sum(current_map) + ADD_EPS)
        return current_map + jnp.exp(-duration * omega) * (prev_map - current_map)

    @conditional_jit
    def differential_time_att_org(self, duration, current_map, prev_map, fixs_x=None, fixs_y=None):
        """
        evolve the maps over time using the *attention* decay parameter

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention map
        prev_map : float
            previous attention map

        Returns
        -------
        array
            evolved map
        """
        current_map = current_map / (jnp.sum(current_map) + ADD_EPS)
        return current_map + jnp.exp(-duration * self.omegaAttention) * (prev_map - current_map)

    def differential_time_att_locdep_round(
        self, duration, current_map, prev_map, fixs_x: Optional[tuple] = None, fixs_y: Optional[tuple] = None
    ):
        """
        SLOWER version of differential_time_att_locdep
        evolve the maps over time with the normal attention decay parameter
        everywhere except around the previous fixation location. A circular
        aperture around the previous location defines the area of reduced decay.

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention map
        prev_map : float
            previous attention map
        fixs_x : tuple
            x fixation locations (prev, current, next)
        fixs_y: tuple
            y fixation location (prev, current, next)

        Returns
        -------
        array
            evolved map
        """
        # assert fixs_x[0] is not None
        # assert fixs_y[0] is not None
        omega_locdep = jnp.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=PRECISION) * self.omegaAttention
        j_prev = self.convert_deg_to_px(fixs_x[0], "x", fix=True)
        i_prev = self.convert_deg_to_px(fixs_y[0], "y", fix=True)

        r_px_x = self.convert_deg_to_px(self.foR_size / 2, "x", fix=False)
        r_px_y = self.convert_deg_to_px(self.foR_size / 2, "y", fix=False)

        mask = (((self._xx - j_prev) ** 2) / (r_px_x**2) + ((self._yy - i_prev) ** 2) / (r_px_y**2)).T < 1
        omega_locdep = jnp.where(mask, self.omega_prevloc, omega_locdep)

        current_map = current_map / (jnp.sum(current_map) + ADD_EPS)
        return current_map + jnp.exp(-duration * omega_locdep) * (prev_map - current_map)

    def make_positive(self, u):
        """
        cuts off negative components

        Parameters
        ----------
        u : array
            map to cut

        Returns
        -------
        array :
            ustar
        """
        # equation 12
        ustar = u
        ustar = jnp.where(ustar <= 0, 0, ustar)

        # handle numerical problem 3:
        # if inhibStrength is too large, whole map can be negative,
        # whole map is set to 0, cannot divide by sum 0
        # solution: make uniform
        ustar = jnp.where(
            jnp.sum(ustar) == 0,
            jnp.ones_like(ustar, dtype=PRECISION) / (ustar.shape[0] * ustar.shape[1]),
            ustar / jnp.sum(ustar),
        )

        return ustar

    @conditional_jit
    def add_noise(self, ustar):
        """
        adds zeta noise

        Parameters
        ----------
            ustar : array
                map without noise

        Returns
        -------
        array
            uFinal; map with noise
        """
        zeta = jnp.clip(self.zeta, self.PARAMETER_BOUNDS["zeta"][0], self.PARAMETER_BOUNDS["zeta"][1])
        return (1 - zeta) * ustar + zeta / (jnp.prod(jnp.array(self._xx.shape)) + ADD_EPS)

    @conditional_jit
    def get_phase_times_both(self, nth, durations):
        """
        Helper function that computes the durations of each phase
        1. You always have the post phase, no matter what
        2. the main phase can be skipped if the post phase is already too long
        3. from the time that is left for the main phase we then subtract the \\
        pre phase. If that leaves the main phase less than 10 ms short, \\
        we skip the pre phase.
        4. The post phase is always skipped for the first fixation

        Parameters
        ----------
        nth : int
            the how many'th fixation in the sequence are we dealing with
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        float
            duration of post phase (secs)
        float
            duration of main phase (secs)
        float
            duration of pre phase (secs)
        """
        duration_post_ph = jnp.where(nth > 1, jnp.minimum(self.tau_post, durations[1]), 0.0)
        duration_pre_ph = jnp.where(
            jnp.logical_and(nth > 1, durations[1] - duration_post_ph > self.tau_pre + 0.01),
            self.tau_pre,
            jnp.where(jnp.logical_and(nth == 1, durations[1] > self.tau_pre + 0.01), self.tau_pre, 0.0),
        )
        duration_main_ph = jnp.where(
            nth > 1,
            durations[1] - duration_pre_ph - duration_post_ph,
            jnp.where(nth == 1, durations[1] - duration_pre_ph, 0.0),
        )
        return duration_post_ph, duration_main_ph, duration_pre_ph

    def cardinal_attention_bias(self, fix_x, fix_y, nth, attention_map):
        """
        Makes a cardinal attention bias map with two components:
        1. A smooth bias favoring cardinal directions over oblique ones
        2. A leftward bias for early fixations (nth=1,2)

        The potential combines:
        - A cardinal direction bias using cosine of angle
        - A distance-dependent effect that decays with distance from current fixation
        - A leftward bias for early fixations with learnable strength and decay rate
        - Different strengths for horizontal and vertical biases with learnable growth rate

        Parameters
        ----------
        fix_x : float
            x coordinate on which to center the cross
        fix_y : float
            y coordinate on which to center the cross
        nth : int
            Current fixation number. Used to enable leftward bias for early fixations.
        attention_map : array

        Returns
        -------
        array
            map with cardinal attention bias

        Notes
        -----
        The distance decay is controlled by the chi parameter:
        - Higher chi values (e.g., 0.5) make the potential decay more slowly with distance
        - Lower chi values (e.g., 0.1) make it decay more quickly
        - The default value of 0.3 gives a moderate decay rate

        The leftward bias is controlled by:
        - alpha_left_bias: Base strength of leftward bias (starting value)
        - left_bias_decay_rate: How quickly the left bias decays with increasing fixation number
            (higher values = faster decay)
        - The bias decays from alpha_left_bias toward 1.0 (neutral)

        The horizontal/vertical bias is controlled by:
        - alpha_vert_bias: Relative strength of vertical vs horizontal bias when fully developed
        - dir_bias_growth_rate: How quickly the directional bias grows with increasing fixation number
            (higher values = faster growth)
        - The bias grows from 1.0 (neutral) toward alpha_vert_bias
        """
        x_px = self.convert_deg_to_px(fix_x, "x", fix=True)
        y_px = self.convert_deg_to_px(fix_y, "y", fix=True)

        # Calculate distances from current fixation
        dx = self._xx - x_px
        dy = self._yy - y_px

        # Calculate distance from current fixation
        dist = jnp.sqrt(dx**2 + dy**2)

        # Create smooth cardinal bias using cosine of angle
        angle = jnp.arctan2(dy, dx)
        cardinal_bias = jnp.cos(4 * angle)  # 4*angle gives us 4 peaks (cardinal directions)

        # Normalize and scale the cardinal bias
        cardinal_bias = (cardinal_bias + 1) / 2
        cardinal_bias = cardinal_bias / (jnp.sum(cardinal_bias) + ADD_EPS)

        # Add distance-dependent effect - stronger for shorter saccades
        # chi controls how quickly the potential decays with distance
        dist_factor = jnp.exp(-dist / (self.MAP_SIZE / (4 * self.chi)))  # Higher chi = slower decay
        q = cardinal_bias * dist_factor

        # Calculate directional bias strength that grows exponentially from 1.0 (neutral) toward alpha_vert_bias
        # As nth increases, dir_bias_strength approaches alpha_vert_bias
        dir_bias_strength = 1.0 + (self.alpha_vert_bias - 1.0) * (1.0 - jnp.exp(-self.dir_bias_growth_rate * (nth)))

        # Apply different strengths to horizontal and vertical biases based on current growth factor
        # Create masks for horizontal and vertical directions
        is_horizontal = jnp.abs(dx) > jnp.abs(dy)

        # Scale the potential based on direction with gradually increasing strength
        q = jnp.where(is_horizontal, q, q * dir_bias_strength)

        def apply_leftward_bias(q):
            # Calculate decaying leftward bias strength:
            # - Starts at alpha_left_bias
            # - Exponentially approaches 1.0 (neutral value) as nth increases
            left_bias_strength = 1.0 + (self.alpha_left_bias - 1.0) * jnp.exp(-self.left_bias_decay_rate * (nth - 1))

            # Create leftward bias mask
            left_mask = dx < 0
            right_mask = dx > 0

            # Apply stronger bias to horizontal saccades
            horizontal_mask = jnp.abs(dy) < jnp.abs(dx)

            # Combine masks
            left_bias = left_mask & horizontal_mask
            right_bias = right_mask & horizontal_mask

            # Apply bias with current strength
            q = jnp.where(left_bias, q * left_bias_strength, q)
            q = jnp.where(right_bias, q / left_bias_strength, q)
            return q

        # Apply leftward bias with decay
        # Only apply if bias hasn't decayed to very close to neutral (using threshold)
        left_bias_active = (
            jnp.abs(1.0 + (self.alpha_left_bias - 1.0) * jnp.exp(-self.left_bias_decay_rate * (nth - 1)) - 1.0) > 0.05
        )
        q = lax.cond(left_bias_active, apply_leftward_bias, lambda x: x, q)

        # Transpose to match grid coordinates
        q = q.T

        # Normalize final result
        q = q / (jnp.sum(q) + ADD_EPS)

        # Apply to attention map
        q = q * attention_map

        q = q / (jnp.sum(q) + ADD_EPS)

        return q

    def plot_om_potential(self, fix_x, fix_y, nth):
        """
        Debugging function.
        """
        x_px = self.convert_deg_to_px(fix_x, "x", fix=True)
        y_px = self.convert_deg_to_px(fix_y, "y", fix=True)

        # Calculate distances from current fixation
        dx = self._xx - x_px
        dy = self._yy - y_px

        # Calculate distance from current fixation
        dist = jnp.sqrt(dx**2 + dy**2)

        # Create smooth cardinal bias using cosine of angle
        angle = jnp.arctan2(dy, dx)
        cardinal_bias = jnp.cos(4 * angle)  # 4*angle gives us 4 peaks (cardinal directions)

        # Normalize and scale the cardinal bias
        cardinal_bias = (cardinal_bias + 1) / 2
        cardinal_bias = cardinal_bias / (jnp.sum(cardinal_bias) + ADD_EPS)

        # Add distance-dependent effect - stronger for shorter saccades
        # chi controls how quickly the potential decays with distance
        dist_factor = jnp.exp(-dist / (self.MAP_SIZE / (4 * self.chi)))  # Higher chi = slower decay
        q = cardinal_bias * dist_factor

        # Calculate directional bias strength that grows exponentially from 1.0 (neutral) toward alpha_vert_bias
        # As nth increases, dir_bias_strength approaches alpha_vert_bias
        dir_bias_strength = 1.0 + (self.alpha_vert_bias - 1.0) * (1.0 - jnp.exp(-self.dir_bias_growth_rate * (nth)))

        # Apply different strengths to horizontal and vertical biases based on current growth factor
        # Create masks for horizontal and vertical directions
        is_horizontal = jnp.abs(dx) > jnp.abs(dy)

        # Scale the potential based on direction with gradually increasing strength
        q = jnp.where(is_horizontal, q, q * dir_bias_strength)

        def apply_leftward_bias(q):
            # Calculate decaying leftward bias strength:
            # - Starts at alpha_left_bias
            # - Exponentially approaches 1.0 (neutral value) as nth increases
            left_bias_strength = 1.0 + (self.alpha_left_bias - 1.0) * jnp.exp(-self.left_bias_decay_rate * (nth - 1))

            # Create leftward bias mask
            left_mask = dx < 0
            right_mask = dx > 0

            # Apply stronger bias to horizontal saccades
            horizontal_mask = jnp.abs(dy) < jnp.abs(dx)

            # Combine masks
            left_bias = left_mask & horizontal_mask
            right_bias = right_mask & horizontal_mask

            # Apply bias with current strength
            q = jnp.where(left_bias, q * left_bias_strength, q)
            q = jnp.where(right_bias, q / left_bias_strength, q)
            return q

        # Apply leftward bias with decay
        # Only apply if bias hasn't decayed to very close to neutral (using threshold)
        left_bias_active = (
            jnp.abs(1.0 + (self.alpha_left_bias - 1.0) * jnp.exp(-self.left_bias_decay_rate * (nth - 1)) - 1.0) > 0.05
        )
        q = lax.cond(left_bias_active, apply_leftward_bias, lambda x: x, q)

        # Transpose to match grid coordinates
        q = q.T

        # Normalize final result
        q = q / (jnp.sum(q) + ADD_EPS)

        return q

    def make_om_potential_neg(self, fix_x, fix_y):
        """
        makes an occulomotor potential map where the cardinal directions have
        lower activation than the oblique.

        Parameters
        ----------
        fix_x : float
            x coordinate on which to center the cross
        fix_y : float
            y coordinate on which to center the cross

        Returns
        -------
        array
            map with oculomotor potential
        """
        x_px = self.convert_deg_to_px(fix_x, "x", fix=True)
        y_px = self.convert_deg_to_px(fix_y, "y", fix=True)
        q1 = (self._xx - x_px) ** 2
        q2 = (self._yy - y_px) ** 2
        q = ((q1 * q2) ** self.chi).T
        q = q / (jnp.max(q) + ADD_EPS)
        return q

    def make_om_potential(self, fix_x, fix_y):
        """
        makes an occulomotor potential map where the cardinal directions have
        higher activation than the oblique.

        Parameters
        ----------
        fix_x : float
            x coordinate on which to center the cross
        fix_y : float
            y coordinate on which to center the cross

        Returns
        -------
        array
            map with oculomotor potential
        """
        x_px = self.convert_deg_to_px(fix_x, "x", fix=True)
        y_px = self.convert_deg_to_px(fix_y, "y", fix=True)
        q1 = (self._xx - x_px) ** 2
        q2 = (self._yy - y_px) ** 2
        q = ((q1 * q2) ** self.chi).T
        q = q / (jnp.max(q) + ADD_EPS)
        q = jnp.abs(q - 1)
        return q

    # --------------------------------------------------------------------------
    # MECHANISMS
    # --------------------------------------------------------------------------
    @conditional_jit
    def compute_saccadic_momentum(self, fixs_x, fixs_y, nth):
        """
        Computes an oculomotor preference map that assigns behavioral costs to each possible
        gaze location, with strength depending on fixation index and previous saccade length.

        This implementation handles both:
        1. Saccadic momentum: tendency to continue in same direction after long saccades
        2. Return saccades: tendency to return in the opposite direction (180°) after
           short saccades and during early fixations

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y
        nth : int, optional
            current fixation index in the scanpath

        Returns
        -------
        array
            Oculomotor preference map with the same shape as the internal model maps
        """
        # Extract fixation coordinates
        prev_x, curr_x = fixs_x[0], fixs_x[1]
        prev_y, curr_y = fixs_y[0], fixs_y[1]

        # Convert to internal map coordinates
        prev_x_px = self.convert_deg_to_px(prev_x, "x", fix=True)
        prev_y_px = self.convert_deg_to_px(prev_y, "y", fix=True)
        curr_x_px = self.convert_deg_to_px(curr_x, "x", fix=True)
        curr_y_px = self.convert_deg_to_px(curr_y, "y", fix=True)

        # Calculate previous saccade vector and length
        prev_sac_x = curr_x_px - prev_x_px
        prev_sac_y = curr_y_px - prev_y_px
        prev_sac_len = jnp.sqrt(prev_sac_x**2 + prev_sac_y**2)

        # Convert saccade length to degrees for threshold comparison
        prev_sac_len_deg = jnp.sqrt(
            (self.convert_px_to_deg(prev_sac_x, "x")) ** 2 + (self.convert_px_to_deg(prev_sac_y, "y")) ** 2
        )

        # Create coordinate grid for all potential next fixation points
        grid_x, grid_y = self._xx, self._yy

        # Handle case where there's no previous saccade (first fixation)
        has_prev_saccade = ~(jnp.isnan(prev_x) | jnp.isnan(prev_y) | (prev_sac_len < EPS))

        # Calculate directional preferences for all possible next fixations
        def calc_direction_preference():
            # Calculate angle of previous saccade
            prev_angle = jnp.arctan2(prev_sac_y, prev_sac_x)

            # Calculate angles for all potential next saccades
            next_sac_x = grid_x - curr_x_px
            next_sac_y = grid_y - curr_y_px
            next_angles = jnp.arctan2(next_sac_y, next_sac_x)

            # Calculate the difference in angles and normalize to [-pi, pi]
            angle_diff = (next_angles - prev_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

            # MOMENTUM CASE: Apply Gaussian preference for continuing in the same direction (0°)
            # Transpose so that meshgrid coordinates match image coordinates (always done in SW meshgrid methods)
            momentum_preference = jnp.exp(-(angle_diff**2) / (2 * self.momentum_sigma**2)).T

            # RETURN SACCADE CASE: Apply Gaussian preference for returning in the opposite direction (180°)
            # Center the Gaussian at ±pi (opposite direction)
            return_preference = (
                jnp.exp(-((angle_diff - jnp.pi) ** 2) / (2 * self.return_sigma**2))
                + jnp.exp(-((angle_diff + jnp.pi) ** 2) / (2 * self.return_sigma**2))
            ).T

            # Fixation index effect: smooth transition from return to momentum behavior
            # Using a sigmoid function to transition from early to late fixations
            # At nth=1, this will be close to 0 (favoring returns)
            # As nth increases, approaches 1 (favoring momentum)
            fixation_weight = 1.0 / (
                1.0 + jnp.exp(-(nth - self.fixation_transition_midpoint) / self.fixation_transition_rate)
            )

            # Calculate saccade length weight (0 for short, 1 for long, gradual transition)
            saccade_length_weight = jnp.clip(
                (prev_sac_len_deg - self.short_saccade_threshold)
                / (self.long_saccade_threshold - self.short_saccade_threshold),
                0.0,
                1.0,
            )

            # Combine fixation count and saccade length effects
            # Lower values favor return saccades, higher values favor momentum
            combined_weight = fixation_weight * saccade_length_weight

            # Smoothly interpolate between return and momentum preferences
            direction_map = (1.0 - combined_weight) * return_preference + combined_weight * momentum_preference

            # Normalize to ensure valid probability distribution
            direction_map = direction_map / (jnp.sum(direction_map) + ADD_EPS)

            return direction_map

        # Apply the scaling factor only if we have a previous saccade
        oculomotor_pref_map = jax.lax.cond(
            has_prev_saccade,
            calc_direction_preference,
            lambda: jnp.ones_like(self._xx)
            / (self.MAP_SIZE * self.MAP_SIZE),  # Uniform distribution for first fixation
        )

        oculomotor_pref_map = oculomotor_pref_map / (jnp.sum(oculomotor_pref_map) + ADD_EPS)

        return oculomotor_pref_map

    def evolve_maps_basic(
        self, durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=False, rng_key=None
    ):
        # no locdep decay on first fix
        map_att = self.make_attention_gauss(fixs_x, fixs_y)
        map_att = self.combine_att_fixdens(map_att, fix_density_map)

        fixs_x, fixs_y = (
            jnp.asarray(none_to_nan(fixs_x), dtype=PRECISION),
            jnp.asarray(none_to_nan(fixs_y), dtype=PRECISION),
        )

        def first_fixation_diff_time_att():
            # on the first fixation use center bias decay or regular, depending on model configuration
            return lax.cond(
                self.att_map_init_type == "cb",
                partial(
                    self.differential_time_basic, durations[1], map_att, map_att_prev, self.first_fix_OmegaAttention
                ),
                partial(self.differential_time_basic, durations[1], map_att, map_att_prev, self.omegaAttention),
            )

        def diff_time_att():
            return self.differential_time_att(durations[1], map_att, map_att_prev, fixs_x=fixs_x, fixs_y=fixs_y)

        # Switch between the two functions based on nth
        map_att = lax.switch(jnp.array(nth == 1).astype(int), [diff_time_att, first_fixation_diff_time_att])

        if self.omp == "attention":
            omp_map = self.cardinal_attention_bias(fixs_x[1], fixs_y[1], nth, map_att)
            map_att = map_att + (self.ompfactor * omp_map)
            map_att = map_att / (jnp.sum(map_att) + ADD_EPS)

        map_inhib = self.make_inhib_gauss(fixs_x, fixs_y)
        map_inhib = self.differential_time_basic(durations[1], map_inhib, map_inhib_prev, self.omegaInhib)
        u = self.combine(map_att, map_inhib, nth)

        if self.omp == "add":
            omp_map = self.make_om_potential(fixs_x[1], fixs_y[1])
            omp_map = omp_map / (jnp.sum(omp_map) + ADD_EPS)
            u = u + (self.ompfactor * omp_map)
        if self.omp == "mult":
            # multiplicative OMP
            u = u * (self.ompfactor * self.make_om_potential(fixs_x[1], fixs_y[1]))

        ustar = self.make_positive(u)
        uFinal = jnp.squeeze(self.add_noise(ustar))

        # get likelihood for next fixations
        fixs_x_2_isnan = jnp.isnan(fixs_x[2])

        if self.saccadic_momentum:
            saccade_momentum_map = self.compute_saccadic_momentum(fixs_x, fixs_y, nth)
            uFinal = uFinal * (self.saccadic_bias_weigth * saccade_momentum_map)
            uFinal = uFinal / (jnp.sum(uFinal) + ADD_EPS)

        def sample_next_fix(uFinal, rng_key):
            x, y, LL = self.fixation_picker_stoch(uFinal, get_lik=True, rng_key=rng_key)
            return x, y, LL

        def compute_fix_ll(fixs_x, fixs_y, uFinal):
            idx_j_next = self.convert_deg_to_px(fixs_x[2], "x", fix=True).astype(int)
            idx_i_next = self.convert_deg_to_px(fixs_y[2], "y", fix=True).astype(int)

            LL = jnp.log2(jnp.array(uFinal[idx_i_next, idx_j_next], dtype=PRECISION) + ADD_EPS)

            return fixs_x[2], fixs_y[2], LL

        # Sample only if the next fixation is not already defined and if we are in simulation mode
        # otherwise compute the likelihood of the next fixation
        condition = fixs_x_2_isnan & sim
        x, y, LL = lax.cond(
            condition,
            lambda _: sample_next_fix(uFinal, rng_key),
            lambda _: compute_fix_ll(fixs_x, fixs_y, uFinal),
            None,
        )
        next_fix = (x, y)

        return map_att, map_inhib, uFinal, next_fix, LL

    def evolve_maps_main(
        self, durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=False, rng_key=None
    ):
        """
        Evolve maps with basic model

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        if sim or self.dynamic_duration:
            idx_j = self.convert_deg_to_px(fixs_x[1], "x", fix=True).astype(int)
            idx_i = self.convert_deg_to_px(fixs_y[1], "y", fix=True).astype(int)
            logact = jnp.log2(jnp.array(fix_density_map[idx_i, idx_j], dtype=PRECISION) + ADD_EPS)
        if sim:
            # this main evolve function is called sometimes im sim mode but we
            # dont want to sample a new duration
            assert (durations[1] is None) or jnp.isnan(durations[1])
            if (durations[1] is None) or jnp.isnan(durations[1]):
                durations = (durations[0], self.duration_picker(logact, rng_key), durations[2])
        # compute likelihood of duration
        if self.dynamic_duration:
            durlik = self.get_duration_likelihood(durations[1], logact)
        else:
            durlik = 1
        map_att, map_inhib, uFinal, next_fix, LL = self.evolve_maps_basic(
            durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=sim, rng_key=rng_key
        )

        spat_temp_LL = LL + jnp.log2(jnp.array(durlik, dtype=PRECISION))

        if self.detail_mode:
            # Here uFinal and the main map coincide so we return uFinal twice
            return map_att, map_inhib, uFinal, uFinal, next_fix, durations[1], spat_temp_LL, LL, jnp.log2(durlik)

        return map_att, map_inhib, uFinal, next_fix, durations[1], spat_temp_LL

    def evolve_maps_postsac(
        self,
        durations,
        fixs_x,
        fixs_y,
        map_att_prev,
        map_inhib_prev,
        fix_density_map,
        nth,
        sim=False,
        rng_key=None,
    ):
        """
        Evolve maps with postsaccadic shift.

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        raise NotImplementedError("Postsaccadic shift only is not implemented for the Jax model.")

    def evolve_maps_presac(
        self, durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=False, rng_key=None
    ):
        """
        Evolve maps with presaccadic shift.

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        raise NotImplementedError("Presaccadic shift only is not implemented for the Jax model.")

    def evolve_maps_both(
        self, durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=False, rng_key=None
    ):
        if sim or self.dynamic_duration:
            idx_j = self.convert_deg_to_px(fixs_x[1], "x", fix=True).astype(int)
            idx_i = self.convert_deg_to_px(fixs_y[1], "y", fix=True).astype(int)
            logact = jnp.log2(jnp.array(fix_density_map[idx_i, idx_j], dtype=PRECISION) + ADD_EPS)
        if sim:
            assert (durations[1] is None) or jnp.isnan(durations[1])
            if (durations[1] is None) or jnp.isnan(durations[1]):
                durations = (durations[0], self.duration_picker(logact, rng_key), durations[2])
        # compute likelihood of duration
        if self.dynamic_duration:
            durlik = self.get_duration_likelihood(durations[1], logact)
        else:
            durlik = 1

        duration_post_ph, duration_main_ph, duration_pre_ph = self.get_phase_times_both(nth, durations)
        skip_post = duration_post_ph == 0
        skip_pre = duration_pre_ph == 0

        map_att_prev, map_inhib_prev = jax.lax.cond(
            skip_post,
            lambda _: (map_att_prev, map_inhib_prev),
            lambda _: self.post_phase(fixs_x, fixs_y, duration_post_ph, map_att_prev, map_inhib_prev, fix_density_map),
            None,
        )

        durations_dummy = (None, duration_main_ph, None)
        map_att_main, map_inhib_main, uFinal_main, next_fix, LL = self.evolve_maps_basic(
            durations_dummy,
            fixs_x,
            fixs_y,
            map_att_prev,
            map_inhib_prev,
            fix_density_map,
            nth,
            sim=sim,
            rng_key=rng_key,
        )

        # pick the next location
        if sim:
            fixs_x = (*fixs_x[0:2], next_fix[0])
            fixs_y = (*fixs_y[0:2], next_fix[1])

        map_att_pre, map_inhib_pre, uFinal_pre = jax.lax.cond(
            skip_pre,
            lambda _: (map_att_main, map_inhib_main, uFinal_main),
            lambda _: self.pre_phase(
                fixs_x, fixs_y, duration_pre_ph, map_att_main, map_inhib_main, fix_density_map, nth
            ),
            None,
        )

        spat_temp_LL = LL + jnp.log2(jnp.array(durlik, dtype=PRECISION))

        if self.detail_mode:
            return (
                map_att_pre,
                map_inhib_pre,
                uFinal_pre,
                uFinal_main,
                tuple(next_fix),
                durations[1],
                spat_temp_LL,
                LL,
                jnp.log2(durlik),
            )

        return map_att_pre, map_inhib_pre, uFinal_pre, tuple(next_fix), durations[1], spat_temp_LL

    def post_phase(self, fixs_x, fixs_y, duration_post_ph, map_att_prev, map_inhib_prev, fix_density_map):
        if fixs_x[0] is None:
            return map_att_prev, map_inhib_prev

        map_att = self.make_attention_gauss_post_shift(fixs_x, fixs_y)
        map_att = self.combine_att_fixdens(map_att, fix_density_map)
        map_att = self.differential_time_att(duration_post_ph, map_att, map_att_prev, fixs_x=fixs_x, fixs_y=fixs_y)

        map_inhib = self.make_inhib_gauss(fixs_x, fixs_y)
        map_inhib = self.differential_time_basic(duration_post_ph, map_inhib, map_inhib_prev, self.omegaInhib)

        return jnp.squeeze(map_att), map_inhib

    def pre_phase(self, fixs_x, fixs_y, duration_pre_ph, map_att_main, map_inhib_main, fix_density_map, nth):
        map_att_shift = self.make_attention_gauss(fixs_x[1:3], fixs_y[1:3])
        map_att_shift = self.combine_att_fixdens(map_att_shift, fix_density_map)

        fixs_x, fixs_y = (
            jnp.asarray(none_to_nan(fixs_x), dtype=PRECISION),
            jnp.asarray(none_to_nan(fixs_y), dtype=PRECISION),
        )

        def following_fixations_case():
            return self.differential_time_att(
                duration_pre_ph, map_att_shift, map_att_main, fixs_x=fixs_x, fixs_y=fixs_y
            )

        def first_fixation_case():
            return self.differential_time_basic(duration_pre_ph, map_att_shift, map_att_main, self.omegaAttention)

        map_att_pre = lax.cond(nth != 1, following_fixations_case, first_fixation_case)

        map_inhib_pre = self.make_inhib_gauss(fixs_x, fixs_y)
        map_inhib_pre = self.differential_time_basic(duration_pre_ph, map_inhib_pre, map_inhib_main, self.omegaInhib)

        u = self.combine(map_att_pre, map_inhib_pre, nth)
        ustar = self.make_positive(u)
        uFinal_pre = self.add_noise(ustar)

        return map_att_pre, map_inhib_pre, jnp.squeeze(uFinal_pre)

    # --------------------------------------------------------------------------
    # Main Interface Functions
    # --------------------------------------------------------------------------

    @conditional_jit
    def get_scanpath_likelihood(self, x_path, y_path, dur_path, fix_dens):
        """
        calculate likelihood of one scanpath under scenewalk model with params in scene_walk_params.

        Parameters
        ----------
        x_path : array
            array with a datapoint for each fixation's x coordinate
        y_path : array
            array with a datapoint for each fixation's y coordinate
        dur_path : array
            array with a datapoint for each fixation's duration
        fix_dens : array
            empirical fixation density of the viewed image

        Returns
        -------
        array
            log likelihood of the scanpath, one value for each fixation

        Notes
        -----
        Beware: Unlike the original Numpy implementation, this function returns per-fixation log likelihoods,
                not their sum. This is for added flexibility.
        """
        x_path = jnp.asarray(x_path, dtype=PRECISION)
        y_path = jnp.asarray(y_path, dtype=PRECISION)
        dur_path = jnp.asarray(dur_path, dtype=PRECISION)
        fix_dens = jnp.asarray(fix_dens, dtype=PRECISION)

        def scan_fun(carry, inputs):
            mapAtt, mapInhib, i_fix, rng_key = carry
            fixs_x, fixs_y, durs = inputs

            def continue_scan(carry):
                mapAtt, mapInhib, i_fix, rng_key = carry
                key_to_use, rng_key = jax.random.split(rng_key)
                mapAtt, mapInhib, _, _, _, LL = self.evolve_maps(
                    durs, fixs_x, fixs_y, mapAtt, mapInhib, fix_dens, i_fix, rng_key=key_to_use
                )
                return (jnp.squeeze(mapAtt), mapInhib, i_fix + 1, rng_key), LL

            def stop_scan(carry):
                return carry, jnp.nan

            # if there are NaNs at the end of durations, it means we are entering the padding (if there is any)
            # and we should stop the scan
            carry, LL = jax.lax.cond(
                jnp.isnan(durs[-1]),
                stop_scan,
                continue_scan,
                carry,
            )
            return carry, LL

        # initializations
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()
        rng_key = RNG_KEY
        x_iter = self.window(x_path)
        y_iter = self.window(y_path)
        dur_iter = self.window(dur_path)

        # Zip fixation data together
        fixation_data = jnp.array(list(zip(none_to_nan(x_iter), none_to_nan(y_iter), none_to_nan(dur_iter))))

        # Run scan over all fixation data, including the first fixation
        (_, _, _, _), log_ll = jax.lax.scan(scan_fun, (mapAtt, mapInhib, 1, rng_key), fixation_data)

        return log_ll[:-1]

    @conditional_jit
    def get_scanpath_likelihood_detail_core(self, x_path, y_path, dur_path, fix_dens):
        """
        JIT-compatible function to compute scanpath likelihood with detailed information.
        """
        x_path = jnp.asarray(x_path, dtype=PRECISION)
        y_path = jnp.asarray(y_path, dtype=PRECISION)
        dur_path = jnp.asarray(dur_path, dtype=PRECISION)
        fix_dens = jnp.asarray(fix_dens, dtype=PRECISION)

        # Initial maps and rng setup
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()
        rng_key = RNG_KEY

        x_iter = self.window(x_path)
        y_iter = self.window(y_path)
        dur_iter = self.window(dur_path)

        # Zip fixation data together
        fixation_data = jnp.array(list(zip(none_to_nan(x_iter), none_to_nan(y_iter), none_to_nan(dur_iter))))

        def scan_fun(carry, inputs):
            mapAtt, mapInhib, i_fix, rng_key = carry
            fixs_x, fixs_y, durs = inputs

            key_to_use, rng_key = jax.random.split(rng_key)
            mapAtt, mapInhib, mapFinal, map_main, next_fix, fd, LL, spat, temp = self.evolve_maps(
                durs, fixs_x, fixs_y, mapAtt, mapInhib, fix_dens, i_fix, rng_key=key_to_use
            )

            # Pack all necessary data into arrays (JAX-compatible)
            fixation_data = {
                "n": i_fix,
                "pos": jnp.array([fixs_x[1], fixs_y[1]]),
                "next_pos": jnp.array(next_fix),
                "dur": fd,
                "ll_temp": temp,
                "ll_spat": spat,
                "ll": LL,
                "att_map": mapAtt,
                "inhib_map": mapInhib,
                "final_map": mapFinal,
                "main_map": map_main,
                "dens": fix_dens,
            }

            # Return updated carry (map states and fixation index) and the fixation data
            return (mapAtt, mapInhib, i_fix + 1, rng_key), fixation_data

        # Initial carry values
        carry_init = (mapAtt, mapInhib, 1, rng_key)

        # Run scan over all fixation data, excluding the last fixation
        (_, _, _, _), scanpath_dict = jax.lax.scan(scan_fun, carry_init, fixation_data[:-1])

        # Return the full scanpath as a dictionary (arrays for each property)
        return scanpath_dict

    def get_scanpath_likelihood_detail(self, x_path, y_path, dur_path, fix_dens):
        """
        This function is the detail-saving version of the regular
        get_scanpath_likelihood() function. It is meant to be used for analyses, not estimation.
        It makes the additional assumption that values are not padded with nans.
        """
        return dict_to_fixation(self.get_scanpath_likelihood_detail_core(x_path, y_path, dur_path, fix_dens))

    def get_batch_scanpath_likelihood_detail_core(self, x_paths, y_paths, dur_paths, fix_dens):
        """
        Calculate detailed likelihood information for a batch of scanpaths in parallel.

        Parameters
        ----------
        x_paths : Array[batch_size, max_scanpath_len]
            Batch of x coordinates for scanpaths (padded with NaNs)
        y_paths : Array[batch_size, max_scanpath_len]
            Batch of y coordinates for scanpaths (padded with NaNs)
        dur_paths : Array[batch_size, max_scanpath_len]
            Batch of durations for scanpaths (padded with NaNs)
        fix_dens : Array[batch_size, map_size, map_size]
            Batch of fixation density maps

        Returns
        -------
        dict:
            Dictionary with arrays for each scanpath detail field
        """

        # Single scanpath detail calculation
        def single_scanpath_fn(x_path, y_path, dur_path, fix_den):
            return self.get_scanpath_likelihood_detail_core(x_path, y_path, dur_path, fix_den)

        # Vectorize across the batch dimension
        return jax.vmap(single_scanpath_fn)(x_paths, y_paths, dur_paths, fix_dens)

    def simulate_scanpath(self, fix_dens, startpos, n_fixes, get_LL=False, dur_params=None):
        """
        Simulates a scanpath given durations.

        Parameters
        ----------
        fix_dens : array
            empirical fixation density of the viewed image
        startpos : tuple
            beginning location for the scanpath in degrees (x, y)
        nfixes : int
            number of fixations to simulate
        get_LL : bool
            whether to return the LL of the path
        dur_params : None or array
            when using a non-dyndur model, you may want to pass parameters for
            a gamma distribution from which fixation durations will be sampled.
            This should be a list of 2, (t_p, t_alpha), which correspond in
            scipy.stats to (a, 1/scale)

        Returns
        -------
        array
            x of the simulated scanpath
        array
            y of the simulated scanpath
        array
            durations of the simulated scanpath
        array (optional)
            likelihood of the simulated scanpath
        """
        if dur_params is not None:
            if self.dynamic_duration:
                warnings.warn("Your model is defined with dynamic fds, ignoring the passed dur_params")
            else:
                self.t_alpha = dur_params[1]
                self.t_p = dur_params[0]
                self.t_beta = 0

        ll_path = []
        x_path = []
        y_path = []
        dur_path = []
        # initializations
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()

        prev_x = None
        prev_y = None
        prev_dur = None
        curr_x = startpos[0]
        curr_y = startpos[1]
        # we're always evaluating the next fixation, so the last fixation's map
        # is useless
        for i_fix in range(1, n_fixes + 1):
            fixs_x = (prev_x, curr_x, None)
            fixs_y = (prev_y, curr_y, None)
            fixs_dur = (prev_dur, None, None)
            # evolve map given fixation
            mapAtt, mapInhib, _, next_fix, curr_dur, LL = self.evolve_maps(
                fixs_dur,
                fixs_x,
                fixs_y,
                mapAtt,
                mapInhib,
                fix_dens,
                i_fix,
                sim=True,
                rng_key=...,  # TODO
            )
            x_path.append(curr_x)
            y_path.append(curr_y)
            dur_path.append(curr_dur)
            prev_x, prev_y, prev_dur = curr_x, curr_y, curr_dur
            curr_x, curr_y = next_fix
            ll_path.append(LL)

        if get_LL:
            return jnp.array(x_path), jnp.array(y_path), jnp.array(dur_path), jnp.sum(ll_path)
        else:
            return jnp.array(x_path), jnp.array(y_path), jnp.array(dur_path)

    def get_scanpath_likelihood_detail_nojit(self, x_path, y_path, dur_path, fix_dens):
        """
        This function is the slow und detail-saving cousin of the regular
        get_scanpath_likelihood() function, and the unjitted version of "get_scanpath_likelihood_detail".
        Here for debugging and testing purposes only.
        """
        scanpath = []
        assert self.detail_mode is True, "You need to initialise a model with detail_mode = True to use this function"

        # initializations
        x_path = jnp.asarray(x_path, dtype=PRECISION)
        y_path = jnp.asarray(y_path, dtype=PRECISION)
        dur_path = jnp.asarray(dur_path, dtype=PRECISION)
        fix_dens = jnp.asarray(fix_dens, dtype=PRECISION)
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()

        # iterate over all fixations
        x_iter = none_to_nan(self.window(x_path))
        y_iter = none_to_nan(self.window(y_path))
        dur_iter = none_to_nan(self.window(dur_path))
        rng_key = RNG_KEY
        # we're always evaluating the next fixation, so the last fixation's
        # map is useless
        for i_fix, (fixs_x, fixs_y, durs) in enumerate(list(zip(x_iter, y_iter, dur_iter))[:-1], start=1):
            # evolve map given fixation
            key_to_use, rng_key = jax.random.split(rng_key)
            mapAtt, mapInhib, mapFinal, map_main, next_fix, fd, LL, spat, temp = self.evolve_maps(
                durs, fixs_x, fixs_y, mapAtt, mapInhib, fix_dens, i_fix, rng_key=key_to_use
            )

            scanpath.append(
                Fixation(
                    n=i_fix,
                    pos=(fixs_x[1], fixs_y[1]),
                    next_pos=next_fix,
                    dur=fd,
                    ll_temp=temp,
                    ll_spat=spat,
                    ll=LL,
                    att_map=mapAtt,
                    inhib_map=mapInhib,
                    main_map=map_main,
                    final_map=mapFinal,
                    dens=fix_dens,
                )
            )
        return scanpath

    def window(self, iterable, n_before=1, n_after=1):
        """
        Sliding window function.
        Returns a list of lists, where each inner list contains an element and its neighbors.
        None elements are returned as neighbors at the beginning and end.

        Parameters
        ----------
        iterable : list-ish
            List or other kind of iterable
        n_before : int
            Number of elements to return before each element
        n_after : int
            Number of elements to return after each element

        Returns
        -------
        list of lists
            Each inner list contains an element and its surrounding neighbors
        """
        size = n_before + n_after + 1
        iterable_length = len(iterable)

        # Create a list with None values at the beginning and end
        padded_iterable = [jnp.nan] * n_before + list(iterable) + [jnp.nan] * n_after

        # Use list comprehension to generate the sliding windows
        windows = [padded_iterable[i : i + size] for i in range(iterable_length)]

        return windows

    # ------------------------------
    # Dynamical Fixation Likelihood
    # ------------------------------
    def make_duration_distribution(self, logact):
        """
        Get gamma distribution of durations given parameters and fixation
        There are two parametrizations of the Gamma distribution.

        ger Wiki | Scipy   | eng Wiki | Here
        -------------------------------------------
        b        | 1/scale | 1/theta  | ~1/t_alpha (+beta*logact)
        p        | a       | k        | t_p

        in order to simulate with the parameters from scipy fitting set
        parameters to t_beta=0, t_alpha = 1/scale, t_p = a

        """
        # if we have dyndur switched off, beta has to be zero, so we dont have
        # a dynamical influence
        if not self.dynamic_duration:
            eqx.error_if(self.t_beta, self.t_beta != 0, "t_beta has to be zero if dynamic durations is off")

        # To avoid numerical issues during optimisation we need to respect the bounds.
        t_p = jnp.clip(self.t_p, EPS, 1000)

        b = jnp.clip(self.t_alpha + self.t_beta * logact, EPS)

        distr = dist.Gamma(concentration=t_p, rate=b.item())

        return distr

    @conditional_jit
    def duration_picker(self, logact, rng_key):
        distr = self.make_duration_distribution(logact)
        dur = distr.sample(key=rng_key)

        return dur

    @conditional_jit
    def get_duration_likelihood(self, dur, logact):
        distr = self.make_duration_distribution(logact.astype(PRECISION))
        all_durs = jnp.linspace(1 / self.sample_rate, self.trial_length, self.sample_rate * self.trial_length)
        log_lik_samples = distr.log_prob(all_durs.astype(PRECISION))
        lik_samples = jnp.exp(log_lik_samples)
        norm_lik_samples = lik_samples / jnp.sum(lik_samples)
        lik = norm_lik_samples[jnp.floor(dur * self.sample_rate).astype(int)]
        lik = jnp.where(lik == 0, self.EPS, lik)
        return lik

    @property
    def omegaInhib(self):
        if self.coupled_oms:
            return self.omegaAttention / self.omfrac
        return self._omegaInhib

    @property
    def zeta(self):
        if self.logged_z:
            return jnp.power(10, self._zeta)
        return self._zeta

    @property
    def inhibStrength(self):
        if self.logged_cf:
            return jnp.power(10, self._inhibStrength)
        return self._inhibStrength

    @property
    def ompfactor(self):
        if self.logged_ompf:
            return jnp.power(10, self._ompfactor)
        return self._ompfactor

    @property
    def omega_prevloc(self):
        if self.coupled_facil:
            return self.omegaAttention / self.omega_prevloc_frac
        return self._omega_prevloc

    @property
    def _xx(self):
        ii = jj = self.MAP_SIZE
        _xx, _ = jnp.mgrid[0:ii, 0:jj]
        return jax.lax.stop_gradient(_xx.astype(PRECISION))

    @property
    def _yy(self):
        ii = jj = self.MAP_SIZE
        _, _yy = jnp.mgrid[0:ii, 0:jj]
        return jax.lax.stop_gradient(_yy.astype(PRECISION))

    @property
    def lamb(self):
        if self.exponents == 1:
            return self.gamma
        return self._lamb


def load_model_parameters(scenewalk_model: SceneWalk, scene_walk_parameters) -> SceneWalk:
    """
    Loads the specified model parameters into a SceneWalk model object.
    Note that a SceneWalk model object is immutable, so this function
    returns a new SceneWalk model instance with the updated parameters.

    Args:
        scenewalk_model (SceneWalk): The SceneWalk model object to update.
        scene_walk_parameters (dict): A dictionary containing the scene walk parameters.

    Returns:
        SceneWalk: a new model instance with the specified parameters.

    """

    def allow_none_leaf(x):
        return x is None

    def update_model(model, param_name, value, logged=False) -> SceneWalk:
        if logged:
            value = 10**value
        new_model = eqx.tree_at(
            lambda m: getattr(m, param_name), model, jnp.asarray(value, dtype=PRECISION), is_leaf=allow_none_leaf
        )
        return new_model

    model = scenewalk_model

    model = update_model(model, "omegaAttention", scene_walk_parameters["omegaAttention"])

    if model.coupled_oms:
        model = update_model(model, "omfrac", scene_walk_parameters["omfrac"])
    else:
        assert "omegaInhib" in scene_walk_parameters, "omegaInhib must be specified if coupled_oms is False"
        model = update_model(model, "_omegaInhib", scene_walk_parameters["omegaInhib"])

    model = update_model(model, "sigmaAttention", scene_walk_parameters["sigmaAttention"])
    model = update_model(
        model,
        "sigmaInhib",
        scene_walk_parameters["sigmaAttention"] if model.coupled_sigmas else scene_walk_parameters["sigmaInhib"],
    )
    model = update_model(model, "gamma", scene_walk_parameters["gamma"])
    model = update_model(model, "_lamb", model.gamma if model.exponents == 1 else scene_walk_parameters["lamb"])
    model = update_model(model, "_inhibStrength", scene_walk_parameters["inhibStrength"])
    model = update_model(model, "_zeta", scene_walk_parameters["zeta"])

    if model.shifts in ["post", "both"]:
        model = update_model(model, "sigmaShift", scene_walk_parameters["sigmaShift"])
        model = update_model(model, "shift_size", scene_walk_parameters["shift_size"])

    if model.att_map_init_type == "cb":
        model = update_model(model, "first_fix_OmegaAttention", scene_walk_parameters["first_fix_OmegaAttention"])
        model = update_model(model, "cb_sd", (scene_walk_parameters["cb_sd_x"], scene_walk_parameters["cb_sd_y"]))

    if model.locdep_decay_switch == "on":
        if model.coupled_facil:
            model = update_model(model, "omega_prevloc_frac", scene_walk_parameters["omega_prevloc_frac"])
        else:
            assert "omega_prevloc" in scene_walk_parameters, "omega_prevloc must be specified if coupled_facil is False"
            model = update_model(model, "_omega_prevloc", scene_walk_parameters["omega_prevloc"])

    if model.estimate_times:
        model = update_model(model, "tau_pre", scene_walk_parameters["tau_pre"])
        model = update_model(model, "tau_post", scene_walk_parameters["tau_post"])

    if model.omp != "off":
        model = update_model(model, "chi", scene_walk_parameters["chi"])
        model = update_model(model, "_ompfactor", scene_walk_parameters["ompfactor"])
        if model.omp == "attention":
            try:
                model = update_model(model, "alpha_left_bias", scene_walk_parameters["alpha_left_bias"])
                model = update_model(model, "alpha_vert_bias", scene_walk_parameters["alpha_vert_bias"])
                # Add new decay/growth parameters if they exist
                if "left_bias_decay_rate" in scene_walk_parameters:
                    model = update_model(model, "left_bias_decay_rate", scene_walk_parameters["left_bias_decay_rate"])
                if "dir_bias_growth_rate" in scene_walk_parameters:
                    model = update_model(model, "dir_bias_growth_rate", scene_walk_parameters["dir_bias_growth_rate"])
            except KeyError:
                warnings.warn(
                    "alpha_left_bias, alpha_vert_bias not found in parameters. Using default values.",
                    UserWarning,
                    2,
                )
                model = update_model(model, "alpha_left_bias", 1.0)
                model = update_model(model, "alpha_vert_bias", 1.0)
                model = update_model(model, "left_bias_decay_rate", 0.5)
                model = update_model(model, "dir_bias_growth_rate", 0.5)

    if model.dynamic_duration:
        model = update_model(model, "t_alpha", scene_walk_parameters["t_alpha"])
        model = update_model(model, "t_beta", scene_walk_parameters["t_beta"])
        model = update_model(model, "t_p", scene_walk_parameters["t_p"])

    if model.saccadic_momentum:
        model = update_model(model, "saccadic_bias_weigth", scene_walk_parameters["saccadic_bias_weigth"])
        model = update_model(model, "momentum_sigma", scene_walk_parameters["momentum_sigma"])
        model = update_model(model, "fixation_transition_rate", scene_walk_parameters["fixation_transition_rate"])
        model = update_model(model, "long_saccade_threshold", scene_walk_parameters["long_saccade_threshold"])
        if "return_sigma" in scene_walk_parameters:
            model = update_model(model, "return_sigma", scene_walk_parameters["return_sigma"])
        if "fixation_transition_midpoint" in scene_walk_parameters:
            model = update_model(
                model, "fixation_transition_midpoint", scene_walk_parameters["fixation_transition_midpoint"]
            )
        if "short_saccade_threshold" in scene_walk_parameters:
            model = update_model(model, "short_saccade_threshold", scene_walk_parameters["short_saccade_threshold"])
    if model.early_fix_exponents_scaling:
        # Set up decay parameters for gamma
        if "gamma_base" in scene_walk_parameters:
            model = update_model(model, "gamma_base", scene_walk_parameters["gamma_base"])
        else:
            warnings.warn(
                "gamma_base not found in parameters. Using default value of 1.0. Please check your init parameter dictionary.",
                UserWarning,
                2,
            )
            model = update_model(model, "gamma_base", 1.0)

        if "gamma_decay" in scene_walk_parameters:
            model = update_model(model, "gamma_decay", scene_walk_parameters["gamma_decay"])
        else:
            warnings.warn(
                "gamma_decay not found in parameters. Using default value of 0.5. Please check your init parameter dictionary.",
                UserWarning,
                2,
            )
            model = update_model(model, "gamma_decay", 0.5)

        # Only load lamb decay parameters if exponents=2
        if model.exponents == 2:
            if "lamb_base" in scene_walk_parameters:
                model = update_model(model, "lamb_base", scene_walk_parameters["lamb_base"])
            else:
                warnings.warn(
                    "lamb_base not found in parameters. Using default value of 1.0. Please check your init parameter dictionary.",
                    UserWarning,
                    2,
                )
                model = update_model(model, "lamb_base", 1.0)

            if "lamb_decay" in scene_walk_parameters:
                model = update_model(model, "lamb_decay", scene_walk_parameters["lamb_decay"])
            else:
                warnings.warn(
                    "lamb_decay not found in parameters. Using default value of 0.5. Please check your init parameter dictionary.",
                    UserWarning,
                    2,
                )
                model = update_model(model, "lamb_decay", 0.5)
        else:
            # If exponents=1, just set lamb decay parameters to match gamma
            model = update_model(model, "lamb_base", model.gamma_base)
            model = update_model(model, "lamb_decay", model.gamma_decay)

    model.__check_init__()
    return model


def none_to_nan(values):
    return [jnp.nan if x is None else x for x in values]


def initialize_model_parameters(
    scenewalk_model: SceneWalk, rng_key=None, overwrite_defaults: bool = False
) -> SceneWalk:
    """
    Initializes the model parameters within the correct bounds.

    Args:
        scenewalk_model (SceneWalk): The SceneWalk model object to initialize.
        rng_key (jax.random.PRNGKey, optional): The random key for parameter initialization. Defaults to None.
        overwrite_defaults (bool, optional): Whether to overwrite the default parameters with random values. Defaults to False.

    Returns:
        SceneWalk: A new model instance with randomly initialized parameters within the allowed bounds.
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    scene_walk_parameters = {}

    for param, (lower, upper) in scenewalk_model.PARAMETER_BOUNDS.items():
        if hasattr(scenewalk_model, param):
            if (
                getattr(scenewalk_model, param) is None
                or jnp.isnan(getattr(scenewalk_model, param))
                or overwrite_defaults
            ):
                # Check if the parameter has a prior we can use to sample.
                if param in scenewalk_model.PARAMETER_PRIORS:
                    mean, std, lower, upper = (
                        scenewalk_model.PARAMETER_PRIORS[param][0],
                        scenewalk_model.PARAMETER_PRIORS[param][1],
                        scenewalk_model.PARAMETER_PRIORS[param][2],
                        scenewalk_model.PARAMETER_PRIORS[param][3],
                    )
                else:
                    mean, std = 0, 1

                if param in LOGGABLE_PARAMETERS and (
                    (param == "zeta" and scenewalk_model.logged_z)
                    or (param == "inhibStrength" and scenewalk_model.logged_cf)
                    or (param == "ompfactor" and scenewalk_model.logged_ompf)
                ):
                    if param in scenewalk_model.PARAMETER_PRIORS:
                        # priors for loggable parameters are expressed in log directly.
                        pass
                    else:
                        # Bounds, in turn, are expressed in terms of the final computed parameter value.
                        lower = jnp.log10(lower)
                        upper = jnp.log10(upper)

                rng_key, subkey = jax.random.split(rng_key)
                scene_walk_parameters[param] = dist.TruncatedNormal(loc=mean, scale=std, low=lower, high=upper).sample(
                    subkey
                )
            else:
                scene_walk_parameters[param] = getattr(scenewalk_model, param)

    # Deal with the way center bias parameters are provided
    if scenewalk_model.att_map_init_type == "cb":
        rng_key, *subkeys = jax.random.split(rng_key, 3)
        lower_x, upper_x = scenewalk_model.PARAMETER_BOUNDS["cb_sd_x"]
        lower_y, upper_y = scenewalk_model.PARAMETER_BOUNDS["cb_sd_y"]
        scene_walk_parameters["cb_sd_x"] = jax.random.truncated_normal(subkeys[0], lower_x, upper_x) * 10
        scene_walk_parameters["cb_sd_y"] = jax.random.truncated_normal(subkeys[1], lower_y, upper_y) * 10

    return load_model_parameters(scenewalk_model, scene_walk_parameters)


def force_parameters_in_bounds(scenewalk_model: SceneWalk) -> SceneWalk:
    """
    Forces the model parameters to be within the allowed bounds.

    Args:
        scenewalk_model (SceneWalk): The SceneWalk model object to update.

    Returns:
        SceneWalk: A new model instance with the parameters forced within the allowed bounds.
    """
    scene_walk_parameters = {}

    for param, (lower, upper) in scenewalk_model.PARAMETER_BOUNDS.items():
        if hasattr(scenewalk_model, param):
            value = getattr(scenewalk_model, param)
            if value is not None and not jnp.isnan(value):
                scene_walk_parameters[param] = jnp.clip(value, lower, upper)
            elif value is None or jnp.isnan(value):
                scene_walk_parameters[param] = lower

    return load_model_parameters(scenewalk_model, scene_walk_parameters)


# Wrapper to convert back to Fixation dataclass
def dict_to_fixation(scanpath_dict):
    """
    Convert arrays stored in the dictionary into Fixation dataclass instances.
    """
    n_vals = scanpath_dict["n"]
    pos_vals = scanpath_dict["pos"]
    next_pos_vals = scanpath_dict["next_pos"]
    dur_vals = scanpath_dict["dur"]
    ll_temp_vals = scanpath_dict["ll_temp"]
    ll_spat_vals = scanpath_dict["ll_spat"]
    ll_vals = scanpath_dict["ll"]
    att_map_vals = scanpath_dict["att_map"]
    inhib_map_vals = scanpath_dict["inhib_map"]
    final_map_vals = scanpath_dict["final_map"]
    main_map_vals = scanpath_dict["main_map"]
    dens_vals = scanpath_dict["dens"]

    # Construct a list of Fixation dataclass instances
    return [
        Fixation(
            n=n_vals[i],
            pos=pos_vals[i],
            next_pos=next_pos_vals[i],
            dur=dur_vals[i],
            ll_temp=ll_temp_vals[i],
            ll_spat=ll_spat_vals[i],
            ll=ll_vals[i],
            att_map=att_map_vals[i],
            inhib_map=inhib_map_vals[i],
            main_map=main_map_vals[i],
            final_map=final_map_vals[i],
            dens=dens_vals[i],
        )
        for i in range(len(n_vals))
    ]
