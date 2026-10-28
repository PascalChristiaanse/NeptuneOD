"""Construction of TudatPy estimatable parameters from the experiment config.

The configuration in ``cfg.estimation.parameters_to_estimate`` is keyed by
*short* parameter names (e.g. ``neptune_GM``). This module is the single place
that translates those keys into TudatPy
:class:`~tudatpy.dynamics.parameters_setup.EstimatableParameterSettings` objects,
and therefore owns the mapping from a config key to a parameter identifier.

Two important design points:

* **Block sizes are never hard-coded.** The number of scalars each parameter
  contributes is read from the consolidated
  :class:`~tudatpy.dynamics.parameters.EstimatableParameterSet` via
  :attr:`~tudatpy.dynamics.parameters.EstimatableParameter.parameter_size`
  (see :func:`get_parameter_block_layout`). This keeps the a priori builder in
  sync with whatever TudatPy actually builds, including future parameters.

* **Initial values can be set from the configuration.** Using the mapping form
  for a parameter allows overriding its initial value::

      estimation:
        parameters_to_estimate:
          neptune_GM:
            value: 6.809e6
          neptune_inverse_tidal_quality_factor:
            value: 0.0125
          initial_state:
            value: [x, y, z, vx, vy, vz]

  Values are applied to the built parameter set by
  :func:`apply_configured_parameter_values`, which calls
  :meth:`~tudatpy.dynamics.parameters.EstimatableParameter.set_parameter_value`.
  A ``value`` for ``initial_state`` overrides the state the propagator was built
  with (which otherwise comes from the SPICE ephemeris at the arc's initial
  epoch), so the estimation config is the single source of truth for the initial
  estimate.
"""

import logging
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import parameters as param
from tudatpy.dynamics import parameters_setup as param_setup
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)

#: Angular frequencies (rad/s) of the two Neptune pole libration terms estimated
#: when ``iau_rotation_model_pole_librations`` is enabled. Derived from the
#: dominant period in the IAU pole model (``omega_dot`` and its second harmonic).
_NEPTUNE_POLE_LIBRATION_FREQUENCIES: list[float] = [
    np.deg2rad(52.3836218446110 / 36525 / 24 / 3600),
    2 * np.deg2rad(52.3836218446110 / 36525 / 24 / 3600),
]

#: Human-readable description for each additional (non-state) parameter key,
#: used for logging. The set of keys is the authoritative list of parameters that
#: can be requested via ``cfg.estimation.parameters_to_estimate``.
_ADDITIONAL_PARAMETER_DESCRIPTIONS: dict[str, str] = {
    "iau_rotation_model_pole": "IAU rotation model pole",
    "iau_rotation_model_pole_rate": "IAU rotation model pole rate",
    "iau_rotation_model_pole_librations": "IAU rotation model pole librations",
    "neptune_GM": "Neptune gravitational parameter",
    "triton_GM": "Triton gravitational parameter",
    "neptune_j2_j4": "Neptune J2 and J4 spherical harmonics coefficients",
    "neptune_inverse_tidal_quality_factor": (
        "Neptune inverse tidal quality factor (1/Q), tide raised by Triton"
    ),
    "neptune_tidal_time_lag": "Neptune tidal time lag (tau), tide raised by Triton",
}

#: Parameter identifiers that mark an initial-state (propagated/rotational) block.
_STATE_PARAMETER_TYPES = frozenset(
    {
        param_setup.EstimatableParameterTypes.initial_body_state_type,
        param_setup.EstimatableParameterTypes.arc_wise_initial_body_state_type,
        param_setup.EstimatableParameterTypes.initial_rotational_body_state_type,
    }
)


@dataclass(frozen=True)
class ParameterBlock:
    """A single contiguous block of scalars inside the estimation parameter set.

    Attributes
    ----------
    key : str
        Configuration key this block was requested under
        (``"initial_state"`` for the propagated state, otherwise one of
        :data:`_ADDITIONAL_PARAMETER_DESCRIPTIONS`).
    start : int
        Start index of the block in the parameter vector.
    size : int
        Number of scalar parameters in the block.
    """

    key: str
    start: int
    size: int


def _build_additional_parameter_settings(
    param_name: str,
) -> param_setup.EstimatableParameterSettings:
    """Build the TudatPy settings object for a single additional parameter key.

    Parameters
    ----------
    param_name : str
        Configuration key, e.g. ``"neptune_GM"``.

    Returns
    -------
    EstimatableParameterSettings
        Settings whose ``parameter_identifier`` maps the key to the identifier
        TudatPy will use once the parameter set is created.

    Raises
    ------
    ValueError
        If ``param_name`` is not a recognised additional parameter key.
    """
    match param_name:
        case "iau_rotation_model_pole":
            return param_setup.iau_rotation_model_pole("Neptune")
        case "iau_rotation_model_pole_rate":
            return param_setup.iau_rotation_model_pole_rate("Neptune")
        case "iau_rotation_model_pole_librations":
            return param_setup.iau_rotation_model_pole_librations(
                "Neptune", _NEPTUNE_POLE_LIBRATION_FREQUENCIES
            )
        case "neptune_GM":
            return param_setup.gravitational_parameter("Neptune")
        case "triton_GM":
            return param_setup.gravitational_parameter("Triton")
        case "neptune_j2_j4":
            block_indices = [
                (2, 0),  # C20 (J2)
                (4, 0),  # C40 (J4)
            ]
            return param_setup.spherical_harmonics_c_coefficients_block(
                body="Neptune", block_indices=block_indices
            )
        case "neptune_inverse_tidal_quality_factor":
            # Inverse tidal quality factor of Neptune (tide raised on Neptune,
            # by Triton). Requires a direct_tidal_dissipation acceleration on
            # Triton parameterised by (k2, 1/Q, tidal_period); see
            # orbitdet.simulation.dynamics.
            #
            # NOTE: the deforming_body MUST be passed as a list. The single-string
            # overload of tudatpy's inverse_tidal_quality_factor has a bug and
            # returns a *time-lag* parameter instead.
            return param_setup.inverse_tidal_quality_factor("Neptune", ["Triton"])
        case "neptune_tidal_time_lag":
            # Tidal time lag of Neptune (tide raised on Neptune by Triton).
            # Mutually exclusive with neptune_inverse_tidal_quality_factor for the
            # same (body, deforming_body) pair.
            return param_setup.direct_tidal_dissipation_time_lag("Neptune", ["Triton"])
        case _:
            raise ValueError(f"Unknown parameter {param_name} specified for estimation")


def _is_sequence(value) -> bool:
    """Return whether ``value`` is a non-string sequence (list / OmegaConf list)."""
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))
    )


def iter_parameter_entries(cfg: DictConfig) -> list[tuple[str, bool]]:
    """Return ``(key, enabled)`` pairs from ``cfg.estimation.parameters_to_estimate``.

    Supports both the plain form (``neptune_GM: true``) and the mapping form
    (``neptune_GM: {value: ..., apriori: ...}``). In the mapping form the
    parameter is enabled unless ``enabled: false`` is set explicitly.
    """
    parameters_to_estimate = cfg.estimation.parameters_to_estimate
    entries: list[tuple[str, bool]] = []
    for key, value in parameters_to_estimate.items():
        if isinstance(value, DictConfig):
            enabled = bool(value.get("enabled", True))
        else:
            enabled = bool(value)
        entries.append((key, enabled))
    return entries


def get_additional_parameter_identifiers(cfg: DictConfig) -> dict[str, tuple]:
    """Map every enabled additional parameter key to its parameter identifier.

    The identifier is read off the settings object built for the key, so it stays
    correct even for parameter types whose enum member is not exposed by name
    (e.g. the IAU pole parameters). Building these settings requires no bodies or
    propagator settings, so this is usable from the configuration alone.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.

    Returns
    -------
    dict[str, tuple]
        Mapping from configuration key to the corresponding
        ``EstimatableParameterSet`` identifier tuple.
    """
    identifiers: dict[str, tuple] = {}
    for key, enabled in iter_parameter_entries(cfg):
        if not enabled or key == "initial_state":
            continue
        identifiers[key] = _build_additional_parameter_settings(key).parameter_identifier
    return identifiers


def _is_state_identifier(identifier: tuple) -> bool:
    """Return whether ``identifier`` refers to an initial-state parameter block."""
    return identifier[0] in _STATE_PARAMETER_TYPES


def get_parameter_block_layout(
    cfg: DictConfig, parameter_set: param.EstimatableParameterSet
) -> list[ParameterBlock]:
    """Describe how the requested parameters are laid out in ``parameter_set``.

    TudatPy reorders parameter blocks (initial states first, then scalar
    parameters, then vector parameters), so the order in the configuration is
    *not* the order in the parameter vector. This function reconciles the two by
    matching each block of the built set to the configuration key that requested
    it, and reporting the block's real start index and size.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.
    parameter_set : EstimatableParameterSet
        The consolidated parameter set to estimate.

    Returns
    -------
    list[ParameterBlock]
        One block per parameter in the set, in parameter-vector order. The
        ``initial_state`` key labels the propagated/rotational state block(s).

    Raises
    ------
    ValueError
        If the parameter set contains a block that no configured parameter key
        maps to.
    """
    identifiers = get_additional_parameter_identifiers(cfg)
    identifier_to_key = {identifier: key for key, identifier in identifiers.items()}

    layout: list[ParameterBlock] = []
    for identifier in parameter_set.get_parameter_identifiers():
        matches = parameter_set.indices_for_parameter_type(identifier)
        if not matches:
            continue
        start, size = matches[0]
        if _is_state_identifier(identifier):
            key = "initial_state"
        else:
            key = identifier_to_key.get(identifier)
            if key is None:
                raise ValueError(
                    f"Parameter set contains a block with identifier {identifier!r} that no "
                    "configured parameter key maps to. Update "
                    "orbitdet.estimation.estimators._ADDITIONAL_PARAMETER_DESCRIPTIONS and "
                    "_build_additional_parameter_settings to cover it."
                )
        layout.append(ParameterBlock(key=key, start=int(start), size=int(size)))
    return layout


def _configured_parameter_values(cfg: DictConfig) -> dict[str, list[float]]:
    """Extract ``value`` overrides from the mapping-form parameter entries."""
    values: dict[str, list[float]] = {}
    for key, enabled in iter_parameter_entries(cfg):
        if not enabled:
            continue
        entry = cfg.estimation.parameters_to_estimate[key]
        if isinstance(entry, DictConfig) and "value" in entry:
            raw = entry.value
            values[key] = [float(v) for v in raw] if _is_sequence(raw) else [float(raw)]
    return values


def _find_state_parameter_object(
    parameter_set: param.EstimatableParameterSet,
) -> param.EstimatableParameter | None:
    """Return the single initial-state parameter object in ``parameter_set``."""
    for identifier in parameter_set.get_parameter_identifiers():
        if _is_state_identifier(identifier):
            objects = parameter_set.parameters_for_parameter_type(identifier)
            if objects:
                return objects[0]
    return None


def apply_configured_parameter_values(
    cfg: DictConfig, parameter_set: param.EstimatableParameterSet
) -> list[str]:
    """Apply initial parameter values configured under ``parameters_to_estimate``.

    Any parameter entry given in mapping form with a ``value`` key has its value
    written into ``parameter_set`` through
    :meth:`~tudatpy.dynamics.parameters.EstimatableParameter.set_parameter_value`.
    This includes ``initial_state`` (which overrides the SPICE-ephemeris state the
    propagator was built with) and the additional parameters such as
    ``neptune_inverse_tidal_quality_factor``.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.
    parameter_set : EstimatableParameterSet
        The consolidated parameter set whose values should be updated in place.

    Returns
    -------
    list[str]
        The configuration keys whose values were applied.

    Raises
    ------
    ValueError
        If a configured value does not match the size of its parameter block, if
        a key does not resolve to exactly one parameter block, or if an
        ``initial_state`` value is requested for a set with no state block.
    """
    values = _configured_parameter_values(cfg)
    if not values:
        return []

    layout = {block.key: block for block in get_parameter_block_layout(cfg, parameter_set)}
    identifiers = get_additional_parameter_identifiers(cfg)

    applied: list[str] = []
    for key, value in values.items():
        new_value = np.asarray(value, dtype=float).reshape(-1, 1)

        if key == "initial_state":
            block = layout.get("initial_state")
            parameter_object = _find_state_parameter_object(parameter_set)
            if block is None or parameter_object is None:
                raise ValueError("Configuring an `initial_state` value requires a state block.")
        else:
            if key not in identifiers:
                raise ValueError(
                    f"Configured value for '{key}' but '{key}' is not an enabled additional "
                    "parameter."
                )
            objects = parameter_set.parameters_for_parameter_type(identifiers[key])
            if len(objects) != 1:
                raise ValueError(
                    f"Expected exactly one parameter block for '{key}', found {len(objects)}."
                )
            parameter_object = objects[0]
            block = layout.get(key)
            if block is None:
                raise ValueError(f"Configured value for '{key}' but no matching block found.")

        if new_value.size != block.size:
            raise ValueError(
                f"Configured value for '{key}' has {new_value.size} entries but its parameter "
                f"block has size {block.size}."
            )

        previous = np.asarray(parameter_object.get_parameter_value()).ravel()
        parameter_object.set_parameter_value(new_value)
        logger.info("Set '%s' from config: %s -> %s", key, previous, new_value.ravel())
        applied.append(key)

    return applied


def get_estimatable_parameter_settings(
    cfg: DictConfig,
    ctx: RuntimeContext,
    prop_settings: prop_setup.propagator.PropagatorSettings,
    bodies: env.SystemOfBodies,
) -> list[param.EstimatableParameter]:
    """Build the list of estimatable parameter settings from the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.
    ctx : RuntimeContext
        Runtime context (unused for parameter construction; kept for API symmetry).
    prop_settings : PropagatorSettings
        Propagator settings, used to derive the initial-state parameters.
    bodies : SystemOfBodies
        The system of bodies (unused for parameter construction; kept for API
        symmetry).

    Returns
    -------
    list[EstimatableParameter]
        The parameter settings in configuration order (the consolidated set may
        reorder them).
    """
    estimated_parameters: list[param.EstimatableParameter] = []
    if cfg.estimation.parameters_to_estimate.get("initial_state", False):
        logger.warning(
            """Initial state must be estimated for the propagator to work correctly."""
            """ Adding it anyway..."""
        )
    estimated_parameters.extend(param_setup.initial_states(prop_settings, bodies))

    logger.info(f"Parameters to estimate ({len(estimated_parameters)})")
    for param_name, should_estimate in iter_parameter_entries(cfg):
        if not should_estimate or param_name == "initial_state":
            continue
        estimated_parameters.append(_build_additional_parameter_settings(param_name))
        logger.info("\t - %s", _ADDITIONAL_PARAMETER_DESCRIPTIONS.get(param_name, param_name))

    return estimated_parameters


def get_estimatable_parameters(
    cfg: DictConfig,
    ctx: RuntimeContext,
    prop_settings: prop_setup.propagator.PropagatorSettings,
    bodies: env.SystemOfBodies,
) -> param.EstimatableParameterSet:
    """Create the consolidated parameter set and apply any configured values.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.
    ctx : RuntimeContext
        Runtime context (passed through to :func:`get_estimatable_parameter_settings`).
    prop_settings : PropagatorSettings
        Propagator settings.
    bodies : SystemOfBodies
        The system of bodies.

    Returns
    -------
    EstimatableParameterSet
        The parameter set, with any values configured under
        ``estimation.parameters_to_estimate.<key>.value`` applied.
    """
    estimated_parameters = get_estimatable_parameter_settings(cfg, ctx, prop_settings, bodies)
    parameter_set = param_setup.create_parameter_set(estimated_parameters, bodies, prop_settings)
    apply_configured_parameter_values(cfg, parameter_set)
    return parameter_set
