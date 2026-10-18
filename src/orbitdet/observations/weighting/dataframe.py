"""Shared per-observation weight DataFrame construction.

Every weighting strategy must emit the **same** DataFrame schema, because the
engine concatenates the per-set frames into one CSV and the visualization layer
groups on ``group_id``.  A strategy that omits a column (e.g. ``group_id``)
would produce ``NaN`` for its rows and break downstream sorting/grouping.

Keeping the schema in one place guarantees consistency across strategies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from ..grouping import GroupList

logger = logging.getLogger(__name__)

#: Columns every weights DataFrame must contain, in order.
WEIGHTS_DF_COLUMNS: tuple[str, ...] = (
    "set_id",
    "strategy",
    "group_level",
    "observable_type",
    "group_id",
    "parent_id",
    "parent_level",
    "time",
    "ra_residual_rad",
    "dec_residual_rad",
    "ra_residual_arcsec",
    "dec_residual_arcsec",
    "group_ra_rms_arcsec",
    "group_dec_rms_arcsec",
    "weight_ra",
    "weight_dec",
)


def rms(values: np.ndarray) -> float:
    """Root-mean-square of finite values; 0 if there are no finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(finite))))


def residuals_array(observation_set: obs.SingleObservationSet) -> np.ndarray:
    """Extract residuals as an ``(n_obs, 2)`` array in radians.

    Guards against 1D empty arrays and single-observation ``(2,)`` shapes.
    """
    residuals = np.array(observation_set.residuals)
    if residuals.ndim == 1:
        if residuals.shape[0] == 0:
            return np.empty((0, 2))
        return residuals.reshape(1, 2)
    return residuals


def interleave_weights(weights_ra: np.ndarray, weights_dec: np.ndarray) -> np.ndarray:
    """Interleave RA and DEC weights into a flat array.

    Tudat's ``set_tabulated_weights`` expects RA at even indices and DEC at odd
    indices.
    """
    n = len(weights_ra)
    result = np.empty(2 * n, dtype=float)
    result[0::2] = weights_ra
    result[1::2] = weights_dec
    return result


def observable_type_str(observation_set: obs.SingleObservationSet) -> str:
    """Return a human-readable label for the observation's observable type.

    Currently supported:
        - ``angular_position`` (type 1) -> ``"absolute"``
        - ``relative_angular_position`` (type 9) -> ``"relative"``
    """
    from tudatpy.estimation.observable_models_setup import model_settings as obs_model_settings

    labels: dict[int, str] = {
        1: "absolute",  # angular_position_type
        9: "relative",  # relative_angular_position_type
    }
    obs_type = observation_set.observable_type
    for type_id, label in labels.items():
        if obs_type == obs_model_settings.ObservableType(type_id):
            return label
    logger.debug("Unrecognised observable type %s, falling back to 'absolute'", obs_type)
    return "absolute"


def build_weights_df(
    n_obs: int,
    times: np.ndarray,
    residuals_rad: np.ndarray,
    groups: GroupList,
    weights_ra: np.ndarray,
    weights_dec: np.ndarray,
    strategy_name: str,
    set_id: str,
    level: str,
    observable_type: str = "astrometric",
) -> pd.DataFrame:
    """Build the standard per-observation weight metadata DataFrame.

    Each row is one observation and carries its group assignment at ``level``,
    the RA/DEC residuals, and the assigned weights.

    Parameters
    ----------
    level : str
        The grouping level whose ``group_id`` is reported (e.g. ``"timeframe"``).
        If no group at that level covers an observation, ``group_id`` is ``""``.
    observable_type : str
        Human-readable observable type (e.g. ``"absolute"``, ``"relative"``).
    """
    group_ids = np.full(n_obs, "", dtype=object)
    group_ra_rms = np.full(n_obs, np.nan)
    group_dec_rms = np.full(n_obs, np.nan)
    parent_ids = np.full(n_obs, "", dtype=object)
    parent_levels = np.full(n_obs, "", dtype=object)

    for g in groups:
        if g.level != level:
            continue
        ra_rms = rms(residuals_rad[g.indices, 0])
        dec_rms = rms(residuals_rad[g.indices, 1])
        for idx in g.indices:
            group_ids[idx] = g.group_id
            group_ra_rms[idx] = ra_rms
            group_dec_rms[idx] = dec_rms
            if g.parent_id:
                parent_ids[idx] = g.parent_id
            if g.parent_level:
                parent_levels[idx] = g.parent_level

    return pd.DataFrame(
        {
            "set_id": set_id,
            "strategy": strategy_name,
            "group_level": level,
            "observable_type": observable_type,
            "group_id": group_ids,
            "parent_id": parent_ids,
            "parent_level": parent_levels,
            "time": times,
            "ra_residual_rad": residuals_rad[:, 0],
            "dec_residual_rad": residuals_rad[:, 1],
            "ra_residual_arcsec": np.rad2deg(residuals_rad[:, 0]) * 3600.0,
            "dec_residual_arcsec": np.rad2deg(residuals_rad[:, 1]) * 3600.0,
            "group_ra_rms_arcsec": np.rad2deg(group_ra_rms) * 3600.0,
            "group_dec_rms_arcsec": np.rad2deg(group_dec_rms) * 3600.0,
            "weight_ra": weights_ra,
            "weight_dec": weights_dec,
        }
    )
