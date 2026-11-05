import logging

import numpy as np
from omegaconf import DictConfig
from tudatpy.dynamics import parameters as param

from orbitdet.estimation.estimators import (
    get_parameter_block_layout,
    iter_parameter_entries,
)

logger = logging.getLogger(__name__)

#: Sigma used when a parameter block has no a priori constraint. Squared and
#: inverted this gives an (almost) zero weight, i.e. "no constraint".
_UNCONSTRAINED_SIGMA = 1e30

#: Configuration keys inside ``cfg.estimation.apriori`` that constrain the
#: propagated initial state, as opposed to a named additional parameter.
_STATE_APRIORI_KEYS = ("position", "velocity")


def _extract_per_parameter_apriori(cfg: DictConfig) -> dict[str, list[float]]:
    """Collect per-parameter ``apriori`` lists from mapping-form entries.

    Handles the mapping form, e.g.
    ``initial_state: {apriori: [1.0e7, 1.0e2]}`` or
    ``neptune_j2_j4: {apriori: [1e-8, 1e-9]}``.
    """
    per_param: dict[str, list[float]] = {}
    for key, _enabled in iter_parameter_entries(cfg):
        entry = cfg.estimation.parameters_to_estimate[key]
        if isinstance(entry, DictConfig) and "apriori" in entry and entry.apriori is not None:
            per_param[key] = [float(v) for v in entry.apriori]
    return per_param


def _has_global_apriori(global_apriori) -> bool:
    """Return whether the global ``estimation.apriori`` block constrains anything."""
    if not global_apriori:
        return False
    return (
        any(hasattr(global_apriori, key) for key in _STATE_APRIORI_KEYS) or len(global_apriori) > 0
    )


def _block_inverse_variances(
    key: str,
    size: int,
    per_param_apriori: dict[str, list[float]],
    global_apriori,
) -> list[float]:
    """Return the inverse-variance entries for one parameter block.

    The sigma for each block is taken from (in priority order):

    1. The per-parameter ``apriori`` list of the mapping-form config entry. A list
       whose length equals the block size sets each scalar individually; a
       single-element list is broadcast across the block; an ``initial_state``
       block interprets ``[sigma_pos, sigma_vel]`` as position and velocity
       uncertainties.
    2. The global ``cfg.estimation.apriori`` entry ``<key>: sigma`` (broadcast).
       For the initial state the global ``position``/``velocity`` pair is used.
    3. :data:`_UNCONSTRAINED_SIGMA` (no constraint).
    """
    if key == "initial_state":
        return _state_inverse_variances(size, per_param_apriori, global_apriori)

    sigmas: list[float] | None = None
    if key in per_param_apriori and per_param_apriori[key]:
        sigmas = per_param_apriori[key]
    elif hasattr(global_apriori, key):
        sigmas = [float(global_apriori[key])]

    if not sigmas:
        return [1.0 / _UNCONSTRAINED_SIGMA**2] * size

    if len(sigmas) == 1:
        sigma = float(sigmas[0])
        return [1.0 / sigma**2] * size
    if len(sigmas) != size:
        raise ValueError(
            f"A priori for '{key}' provides {len(sigmas)} sigma values but its parameter block "
            f"has size {size}."
        )
    return [1.0 / float(sigma) ** 2 for sigma in sigmas]


def _state_inverse_variances(
    size: int,
    per_param_apriori: dict[str, list[float]],
    global_apriori,
) -> list[float]:
    """Return inverse variances for an initial-state block of ``size`` scalars.

    Each propagated body contributes a contiguous ``[x, y, z, vx, vy, vz]`` group,
    so the position and velocity sigmas alternate every three entries within each
    six-entry group.
    """
    if size <= 0:
        return []

    state_apriori = per_param_apriori.get("initial_state")
    if state_apriori and len(state_apriori) >= 2:
        sigma_pos, sigma_vel = float(state_apriori[0]), float(state_apriori[1])
    elif hasattr(global_apriori, "position") and hasattr(global_apriori, "velocity"):
        sigma_pos = float(global_apriori.position)
        sigma_vel = float(global_apriori.velocity)
    else:
        sigma_pos = sigma_vel = _UNCONSTRAINED_SIGMA

    inv_var_pos = 1.0 / sigma_pos**2
    inv_var_vel = 1.0 / sigma_vel**2

    if size % 6 != 0:
        # Not a translational state (e.g. a rotational state): apply the
        # position sigma to the whole block.
        return [inv_var_pos] * size

    group = [inv_var_pos] * 3 + [inv_var_vel] * 3
    return group * (size // 6)


def get_apriori_covariance_matrix(
    cfg: DictConfig, parameter_set: param.EstimatableParameterSet
) -> np.ndarray | None:
    """Construct the inverse a priori covariance matrix from the configuration.

    The matrix is diagonal and is built **in the order of the consolidated
    parameter set**, so the weight of every block lines up with the parameter it
    constrains. The block sizes are read from TudatPy itself via
    :func:`orbitdet.estimation.estimators.get_parameter_block_layout`
    (backed by :attr:`~tudatpy.dynamics.parameters.EstimatableParameter.parameter_size`),
    so no per-parameter size table needs to be maintained here.

    For each parameter block the a priori sigma is taken from (in priority order):

      - The per-parameter ``apriori`` list of the mapping-form config entry, e.g.
        ``initial_state: {apriori: [sigma_pos, sigma_vel]}`` or
        ``neptune_j2_j4: {apriori: [sigma_c20, sigma_c40]}``.
      - The global ``cfg.estimation.apriori`` entry ``<parameter_key>: sigma``
        (broadcast over the block). For the initial state the global
        ``position`` / ``velocity`` entries are used.
      - :data:`_UNCONSTRAINED_SIGMA` (effectively no constraint).

    If **no** a priori constraints are specified anywhere in the configuration
    (neither per-parameter nor global), the function returns ``None`` to indicate
    that no a priori covariance should be applied.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra / OmegaConf configuration object.
    parameter_set : EstimatableParameterSet
        The consolidated parameter set the prior will be applied to. Its block
        layout determines the size and ordering of the returned matrix.

    Returns
    -------
    np.ndarray | None
        Diagonal **inverse** a priori covariance matrix (shape ``N x N``),
        ready to be passed to
        ``est_an.EstimationInput(…, inverse_apriori_covariance=…)``,
        or ``None`` if no a priori constraints are configured.
    """
    global_apriori = cfg.estimation.get("apriori", {})
    per_param_apriori = _extract_per_parameter_apriori(cfg)

    if not per_param_apriori and not _has_global_apriori(global_apriori):
        logger.info("No a priori constraints configured — returning None.")
        return None

    layout = get_parameter_block_layout(cfg, parameter_set)

    inv_var_list: list[float] = []
    for block in layout:
        inv_var_list.extend(
            _block_inverse_variances(block.key, block.size, per_param_apriori, global_apriori)
        )

    inverse_apriori_covariance = np.diag(inv_var_list)

    expected = int(parameter_set.parameter_set_size)
    actual = inverse_apriori_covariance.shape[0]
    if actual != expected:
        raise ValueError(
            f"Constructed inverse a priori covariance has size {actual}, but the parameter "
            f"set has {expected} parameters."
        )

    logger.info(
        "Constructed inverse a priori covariance matrix "
        f"(size {inverse_apriori_covariance.shape[0]}×"
        f"{inverse_apriori_covariance.shape[1]})."
    )
    logger.debug(f"Inverse a priori covariance diagonal:\n{np.diag(inverse_apriori_covariance)}")

    return inverse_apriori_covariance
