"""Concrete weighting strategies.

Each strategy is a class inheriting from :class:`WeightStrategy` and registered
via the :func:`register_weight_strategy` decorator.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from .base import WeightStrategy
from .grouping import GroupList
from .registry import register_weight_strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rms(values: np.ndarray) -> float:
    """Root-mean-square of finite values; returns 0 if no finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(finite))))


def _residuals_array(observation_set: obs.SingleObservationSet) -> np.ndarray:
    """Extract residuals as (n_obs, 2) array in radians.

    Guards against 1D empty arrays.
    """
    residuals = np.array(observation_set.residuals)
    if residuals.ndim == 1:
        if residuals.shape[0] == 0:
            return np.empty((0, 2))
        # Single observation: shape (2,) -> (1, 2)
        return residuals.reshape(1, 2)
    return residuals


def _build_weights_df(
    n_obs: int,
    times: np.ndarray,
    residuals_rad: np.ndarray,
    groups: GroupList,
    weights_ra: np.ndarray,
    weights_dec: np.ndarray,
    strategy_name: str,
    set_id: str,
    level: str,
) -> pd.DataFrame:
    """Build a per-observation metadata DataFrame.

    Each row corresponds to one observation and carries the group assignment
    at the specified level, the RA/DEC residuals, and the assigned weights.
    """
    # Build a group-id array at the finest granularity
    group_ids = np.full(n_obs, "", dtype=object)
    group_ra_rms = np.full(n_obs, np.nan)
    group_dec_rms = np.full(n_obs, np.nan)
    parent_ids = np.full(n_obs, "", dtype=object)
    parent_levels = np.full(n_obs, "", dtype=object)

    for g in groups:
        if g.level == level:
            ra_rms = _rms(residuals_rad[g.indices, 0])
            dec_rms = _rms(residuals_rad[g.indices, 1])
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


# ---------------------------------------------------------------------------
# IDv2 strategy
# ---------------------------------------------------------------------------


@register_weight_strategy("id_v2")
class IDv2Weight(WeightStrategy):
    """ID v2 weighting: per-file RMSE from descaled per-timeframe RMSEs.

    The ID v2 RMSE is the RMS of the **descaled** per-timeframe RMSEs:

    1. Split the set into timeframes.
    2. For each timeframe :math:`t`:
       :math:`\\sigma_{\\text{RA},t} = \\text{RMS}(r_{\\text{RA},i\\in t})`,
       :math:`\\sigma_{\\text{DEC},t} = \\text{RMS}(r_{\\text{DEC},i\\in t})`
    3. Descale: :math:`\\sigma_{\\text{RA},t,\\text{descaled}} = \\sigma_{\\text{RA},t} \\cdot \\sqrt{n_t}`
    4. ID v2 RMSE: :math:`\\sigma_{\\text{RA,id\\_v2}} = \\text{RMS}(\\sigma_{\\text{RA},t,\\text{descaled}})`
    5. Weight: :math:`w_{\\text{RA}} = 1 / \\sigma_{\\text{RA,id\\_v2}}^2`

    This accounts for the number of observations per night, unlike the plain
    ID RMSE.
    """

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        # Collect timeframe groups
        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            # Fallback: no timeframes defined — use the set as one group
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_array = _interleave_weights(
                np.full(n_obs, w_ra), np.full(n_obs, w_dec)
            )
            df = _build_weights_df(
                n_obs, times, residuals_rad, groups, np.full(n_obs, w_ra),
                np.full(n_obs, w_dec), "id_v2", set_id, "set",
            )
            return weights_array, df

        # Per-timeframe descaled RMSEs
        descaled_ra: list[float] = []
        descaled_dec: list[float] = []
        for g in tf_groups:
            ra_rms = _rms(residuals_rad[g.indices, 0])
            dec_rms = _rms(residuals_rad[g.indices, 1])
            scale = np.sqrt(g.size)
            descaled_ra.append(max(ra_rms, min_sigma_rad) * scale)
            descaled_dec.append(max(dec_rms, min_sigma_rad) * scale)

        # ID v2 RMSE = RMS of descaled per-timeframe RMSEs
        id_v2_ra_sigma = max(_rms(np.array(descaled_ra)), min_sigma_rad)
        id_v2_dec_sigma = max(_rms(np.array(descaled_dec)), min_sigma_rad)

        w_ra = 1.0 / id_v2_ra_sigma**2
        w_dec = 1.0 / id_v2_dec_sigma**2

        # Constant weight per set
        weights_ra_arr = np.full(n_obs, w_ra)
        weights_dec_arr = np.full(n_obs, w_dec)
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs, times, residuals_rad, groups,
            weights_ra_arr, weights_dec_arr,
            "id_v2", set_id, "timeframe",
        )
        logger.debug(
            "IDv2Weight [%s]: RA σ=%.2e rad → w=%.2e, DEC σ=%.2e rad → w=%.2e across %d timeframe(s)",
            set_id, id_v2_ra_sigma, w_ra, id_v2_dec_sigma, w_dec, len(tf_groups),
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Hybrid Geometric strategy
# ---------------------------------------------------------------------------


@register_weight_strategy("hybrid_geometric")
class HybridGeometricWeight(WeightStrategy):
    r"""Hybrid geometric mean of set-level and timeframe-level weights.

    .. math::

        w_{\text{RA},i} = \sqrt{w_{\text{RA,set}} \cdot w_{\text{RA,tf}(i)}}

    where :math:`w_{\text{RA,set}} = 1 / \sigma_{\text{RA,set}}^2` is computed
    from all residuals in the set (file-level RMSE), and
    :math:`w_{\text{RA,tf}(i)} = 1 / \sigma_{\text{RA,tf}(i)}^2` is computed
    from the residuals of the timeframe containing observation :math:`i`.
    The same applies independently for DEC.

    This blends a global (per-file) weight with a local (per-night) weight
    via the geometric mean, which is scale-invariant and stays between the
    two values.
    """

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        # --- Set-level (global) RMSE ---
        ra_set_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
        dec_set_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
        w_ra_set = 1.0 / ra_set_sigma**2
        w_dec_set = 1.0 / dec_set_sigma**2

        # --- Timeframe-level (local) RMSE ---
        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            # Fallback: no timeframes — use set-level only
            w_ra = w_ra_set
            w_dec = w_dec_set
            weights_ra_arr = np.full(n_obs, w_ra)
            weights_dec_arr = np.full(n_obs, w_dec)
            weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)
            df = _build_weights_df(
                n_obs, times, residuals_rad, groups,
                weights_ra_arr, weights_dec_arr,
                "hybrid_geometric", set_id, "set",
            )
            logger.debug(
                "HybridGeoWeight [%s]: no timeframes, using set-level only "
                "(RA σ=%.2e, DEC σ=%.2e)",
                set_id, ra_set_sigma, dec_set_sigma,
            )
            return weights_array, df

        # Build per-observation weight arrays by looking up each obs's timeframe
        weights_ra_arr = np.empty(n_obs)
        weights_dec_arr = np.empty(n_obs)

        # Initialize with set-level weights as default
        weights_ra_arr[:] = w_ra_set
        weights_dec_arr[:] = w_dec_set

        for g in tf_groups:
            ra_tf_sigma = max(_rms(residuals_rad[g.indices, 0]), min_sigma_rad)
            dec_tf_sigma = max(_rms(residuals_rad[g.indices, 1]), min_sigma_rad)
            w_ra_tf = 1.0 / ra_tf_sigma**2
            w_dec_tf = 1.0 / dec_tf_sigma**2

            # Geometric mean with set-level weight
            for idx in g.indices:
                weights_ra_arr[idx] = np.sqrt(w_ra_set * w_ra_tf)
                weights_dec_arr[idx] = np.sqrt(w_dec_set * w_dec_tf)

        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs, times, residuals_rad, groups,
            weights_ra_arr, weights_dec_arr,
            "hybrid_geometric", set_id, "timeframe",
        )
        logger.debug(
            "HybridGeoWeight [%s]: %d timeframe(s), "
            "set RA σ=%.2e w=%.2e, DEC σ=%.2e w=%.2e",
            set_id, len(tf_groups),
            ra_set_sigma, w_ra_set, dec_set_sigma, w_dec_set,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------


def _interleave_weights(
    weights_ra: np.ndarray,
    weights_dec: np.ndarray,
) -> np.ndarray:
    """Interleave RA and DEC weights into a flat array.

    Tudat's ``set_tabulated_weights`` expects RA at even indices and DEC at
    odd indices.
    """
    n = len(weights_ra)
    result = np.empty(2 * n, dtype=float)
    result[0::2] = weights_ra
    result[1::2] = weights_dec
    return result