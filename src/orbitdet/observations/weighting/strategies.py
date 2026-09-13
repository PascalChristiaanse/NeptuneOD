"""Concrete weighting strategies.

Each strategy is a class inheriting from :class:`WeightStrategy` and registered
via the :func:`register_weight_strategy` decorator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from .base import WeightStrategy
from .grouping import Group, GroupList
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
    observable_type: str = "astrometric",
) -> pd.DataFrame:
    """Build a per-observation metadata DataFrame.

    Each row corresponds to one observation and carries the group assignment
    at the specified level, the RA/DEC residuals, and the assigned weights.

    Parameters
    ----------
    observable_type : str
        Human-readable type (e.g. ``"absolute"``, ``"relative"``).
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
    3. Descale:
       :math:`\\sigma_{\\text{RA},t,\\text{descaled}} = \\sigma_{\\text{RA},t} \\cdot \\sqrt{n_t}`
    4. ID v2 RMSE:
       :math:`\\sigma_{\\text{RA,id\\_v2}} = \\text{RMS}(\\sigma_{\\text{RA},t,\\text{descaled}})`
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
            weights_array = _interleave_weights(np.full(n_obs, w_ra), np.full(n_obs, w_dec))
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                groups,
                np.full(n_obs, w_ra),
                np.full(n_obs, w_dec),
                "id_v2",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
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
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "id_v2",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "IDv2Weight [%s]: RA σ=%.2e rad → w=%.2e, "
            "DEC σ=%.2e rad → w=%.2e across %d timeframe(s)",
            set_id,
            id_v2_ra_sigma,
            w_ra,
            id_v2_dec_sigma,
            w_dec,
            len(tf_groups),
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
                n_obs,
                times,
                residuals_rad,
                groups,
                weights_ra_arr,
                weights_dec_arr,
                "hybrid_geometric",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
            )
            logger.debug(
                "HybridGeoWeight [%s]: no timeframes, using set-level only (RA σ=%.2e, DEC σ=%.2e)",
                set_id,
                ra_set_sigma,
                dec_set_sigma,
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
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "hybrid_geometric",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "HybridGeoWeight [%s]: %d timeframe(s), set RA σ=%.2e w=%.2e, DEC σ=%.2e w=%.2e",
            set_id,
            len(tf_groups),
            ra_set_sigma,
            w_ra_set,
            dec_set_sigma,
            w_dec_set,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# ID strategy (plain per-file)
# ---------------------------------------------------------------------------


@register_weight_strategy("id")
class IDWeight(WeightStrategy):
    """Plain per-file RMSE weighting — single constant weight per file.

    .. math::

        \\epsilon^\\alpha_f = \\sqrt{\\frac{1}{N_f}\\sum_i (r^\\alpha_{i,f})^2},
        \\quad w^\\alpha_f = \\frac{1}{(\\epsilon^\\alpha_f)^2}

    This is the most commonly used scheme in the literature.  Each observation
    file (set) gets one weight.  Does **not** account for temporal clustering
    or within-file quality variations.
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

        ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
        dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
        w_ra = 1.0 / ra_sigma**2
        w_dec = 1.0 / dec_sigma**2

        weights_ra_arr = np.full(n_obs, w_ra)
        weights_dec_arr = np.full(n_obs, w_dec)
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "id",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "IDWeight [%s]: RA σ=%.2e rad → w=%.2e, DEC σ=%.2e rad → w=%.2e",
            set_id,
            ra_sigma,
            w_ra,
            dec_sigma,
            w_dec,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# TF strategy (per-timeframe, descaled)
# ---------------------------------------------------------------------------


def _descaled_tf_rmse_and_weights(
    residuals_rad: np.ndarray,
    tf_groups: list,
    min_sigma_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute descaled per-timeframe RMSE and weights.

    For each timeframe :math:`t`:
    :math:`\\epsilon_t = \\text{RMS}(r_{i \\in t})`
    :math:`\\tilde{\\epsilon}_t = \\epsilon_t \\cdot \\sqrt{n_t}` (descaled)
    :math:`w_t = 1 / \\tilde{\\epsilon}_t^2`

    Parameters
    ----------
    residuals_rad : np.ndarray, shape (n_obs, 2)
        RA and DEC residuals in radians.
    tf_groups : list of Group
        Timeframe groups.
    min_sigma_rad : float
        Floor on sigma before descaling.

    Returns
    -------
    tuple
        weights_ra_arr, weights_dec_arr, tf_ra_sigmas, tf_dec_sigmas
        All per-observation arrays.
    """
    n_obs = residuals_rad.shape[0]
    weights_ra_arr = np.zeros(n_obs)
    weights_dec_arr = np.zeros(n_obs)
    for g in tf_groups:
        ra_rms = _rms(residuals_rad[g.indices, 0])
        dec_rms = _rms(residuals_rad[g.indices, 1])
        scale = np.sqrt(g.size)
        # Descaled RMSE
        ra_descaled = max(ra_rms, min_sigma_rad) * scale
        dec_descaled = max(dec_rms, min_sigma_rad) * scale
        w_ra = 1.0 / ra_descaled**2
        w_dec = 1.0 / dec_descaled**2
        for idx in g.indices:
            weights_ra_arr[idx] = w_ra
            weights_dec_arr[idx] = w_dec
    return weights_ra_arr, weights_dec_arr


@register_weight_strategy("timeframe")
class TFWeight(WeightStrategy):
    r"""Per-timeframe weighting with :math:`\sqrt{n_t}` descalement.

    Splits each file into timeframes (nights).  For each timeframe :math:`t`:

    .. math::

        \epsilon^\alpha_t = \sqrt{\frac{1}{n_t}\sum_i (r^\alpha_{t,i})^2},
        \quad
        \tilde{\epsilon}^\alpha_t = \epsilon^\alpha_t \cdot \sqrt{n_t},
        \quad
        w^\alpha_t = \frac{1}{(\tilde{\epsilon}^\alpha_t)^2}

    The descalement accounts for correlated errors within a night.
    A lower bound :math:`\epsilon_t \ge \upsilon_{\min}` is enforced.
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

        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            # Fallback: no timeframes — use set-level as a single timeframe
            tf_groups = [
                Group(indices=np.arange(n_obs, dtype=int), group_id=set_id, level="timeframe")
            ]

        weights_ra_arr, weights_dec_arr = _descaled_tf_rmse_and_weights(
            residuals_rad,
            tf_groups,
            min_sigma_rad,
        )
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "timeframe",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "TFWeight [%s]: %d timeframe(s), min_sigma=%.2e arcsec",
            set_id,
            len(tf_groups),
            min_sigma_arcsec,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# TF free strategy (per-timeframe, no sigma cap)
# ---------------------------------------------------------------------------


@register_weight_strategy("timeframe_free")
class TFFreeWeight(WeightStrategy):
    r"""Per-timeframe weighting with **no** upper limit on weight magnitude.

    Identical to :class:`TFWeight` but with :math:`\upsilon_{\min} = 0`,
    so no lower bound is imposed on the per-timeframe RMSE.  This allows
    certain timeframes to receive extremely high weights, which can skew
    the solution toward those observations and cause formal errors to
    collapse (as noted in the thesis).
    """

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        # Ignore min_sigma_arcsec — effectively min_sigma = 0
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            tf_groups = [
                Group(indices=np.arange(n_obs, dtype=int), group_id=set_id, level="timeframe")
            ]

        weights_ra_arr, weights_dec_arr = _descaled_tf_rmse_and_weights(
            residuals_rad,
            tf_groups,
            min_sigma_rad=0.0,
        )
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "timeframe_free",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "TFFreeWeight [%s]: %d timeframe(s), no sigma cap",
            set_id,
            len(tf_groups),
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# ID v2 helper — compute ID v2 (scaled per-file) sigma from timeframe groups
# ---------------------------------------------------------------------------


def _id_v2_sigmas(
    residuals_rad: np.ndarray,
    tf_groups: list,
    min_sigma_rad: float,
) -> tuple[float, float]:
    """Compute ID v2 per-file sigmas from descaled per-timeframe RMSEs.

    .. math::

        \\sigma_{\\text{RA,id_v2}} =
        \\text{RMS}\\left(\\{\\sigma_{\\text{RA},t} \\cdot \\sqrt{n_t}\\}_{t=1}^{T_f}\\right)
    """
    descaled_ra: list[float] = []
    descaled_dec: list[float] = []
    for g in tf_groups:
        ra_rms = _rms(residuals_rad[g.indices, 0])
        dec_rms = _rms(residuals_rad[g.indices, 1])
        scale = np.sqrt(g.size)
        descaled_ra.append(max(ra_rms, min_sigma_rad) * scale)
        descaled_dec.append(max(dec_rms, min_sigma_rad) * scale)
    ra_sigma = max(_rms(np.array(descaled_ra)), min_sigma_rad)
    dec_sigma = max(_rms(np.array(descaled_dec)), min_sigma_rad)
    return ra_sigma, dec_sigma


# ---------------------------------------------------------------------------
# Hybrid Geometric v2 strategy (thesis-faithful: ID v2 + descaled TF)
# ---------------------------------------------------------------------------


@register_weight_strategy("hybrid_geometric_v2")
class HybridGeometricV2Weight(WeightStrategy):
    r"""Hybrid geometric mean — thesis-faithful ``scaled hybrid geom.``.

    Combines the **ID v2** file-level weight with the **descaled TF**
    per-timeframe weight via the geometric mean:

    .. math::

        w^{(g)}_{t,f} = \sqrt{w_f^{\text{v2}} \cdot w_t^{\text{descaled}}}

    where :math:`w_f^{\text{v2}}` is the ID v2 scaled per-file weight (RMS
    of descaled per-timeframe RMSEs) and :math:`w_t^{\text{descaled}}` is
    the descaled per-timeframe weight :math:`1/(\epsilon_t \sqrt{n_t})^2`.

    This is the ``scaled hybrid geom.`` scheme from the thesis (Table 6.2)
    and ``hybrid_new_id`` in the reference codebase.
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

        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            # Fallback: no timeframes — use ID v2 only
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_array = _interleave_weights(np.full(n_obs, w_ra), np.full(n_obs, w_dec))
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                groups,
                np.full(n_obs, w_ra),
                np.full(n_obs, w_dec),
                "hybrid_geometric_v2",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
            )
            return weights_array, df

        # ID v2 file-level weight
        ra_id_v2_sigma, dec_id_v2_sigma = _id_v2_sigmas(
            residuals_rad,
            tf_groups,
            min_sigma_rad,
        )
        w_ra_file = 1.0 / ra_id_v2_sigma**2
        w_dec_file = 1.0 / dec_id_v2_sigma**2

        # Descaled TF weight per timeframe
        weights_ra_arr = np.empty(n_obs)
        weights_dec_arr = np.empty(n_obs)
        for g in tf_groups:
            ra_rms = _rms(residuals_rad[g.indices, 0])
            dec_rms = _rms(residuals_rad[g.indices, 1])
            scale = np.sqrt(g.size)
            ra_descaled = max(ra_rms, min_sigma_rad) * scale
            dec_descaled = max(dec_rms, min_sigma_rad) * scale
            w_ra_tf = 1.0 / ra_descaled**2
            w_dec_tf = 1.0 / dec_descaled**2
            # Geometric mean
            for idx in g.indices:
                weights_ra_arr[idx] = np.sqrt(w_ra_file * w_ra_tf)
                weights_dec_arr[idx] = np.sqrt(w_dec_file * w_dec_tf)

        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "hybrid_geometric_v2",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "HybridGeometricV2Weight [%s]: %d timeframe(s), IDv2 RA σ=%.2e DEC σ=%.2e",
            set_id,
            len(tf_groups),
            ra_id_v2_sigma,
            dec_id_v2_sigma,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Hybrid Arithmetic strategy (thesis-faithful: ID v2 + descaled TF)
# ---------------------------------------------------------------------------


@register_weight_strategy("hybrid_arithmetic")
class HybridArithmeticWeight(WeightStrategy):
    r"""Hybrid arithmetic mean — thesis-faithful ``scaled hybrid arith.``.

    Combines the **ID v2** file-level weight with the **descaled TF**
    per-timeframe weight via the arithmetic mean:

    .. math::

        w^{(a)}_{t,f} = \\frac{w_f^{\text{v2}} + w_t^{\text{descaled}}}{2}

    This is the ``scaled hybrid arith.`` scheme from the thesis (Table 6.2).
    In practice the arithmetic mean is dominated by the per-timeframe
    component (which is typically orders of magnitude larger), making it
    less balanced than the geometric variant.
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

        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            # Fallback: no timeframes — use ID v2 only
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_array = _interleave_weights(np.full(n_obs, w_ra), np.full(n_obs, w_dec))
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                groups,
                np.full(n_obs, w_ra),
                np.full(n_obs, w_dec),
                "hybrid_arithmetic",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
            )
            return weights_array, df

        # ID v2 file-level weight
        ra_id_v2_sigma, dec_id_v2_sigma = _id_v2_sigmas(
            residuals_rad,
            tf_groups,
            min_sigma_rad,
        )
        w_ra_file = 1.0 / ra_id_v2_sigma**2
        w_dec_file = 1.0 / dec_id_v2_sigma**2

        # Descaled TF weight per timeframe
        weights_ra_arr = np.empty(n_obs)
        weights_dec_arr = np.empty(n_obs)
        for g in tf_groups:
            ra_rms = _rms(residuals_rad[g.indices, 0])
            dec_rms = _rms(residuals_rad[g.indices, 1])
            scale = np.sqrt(g.size)
            ra_descaled = max(ra_rms, min_sigma_rad) * scale
            dec_descaled = max(dec_rms, min_sigma_rad) * scale
            w_ra_tf = 1.0 / ra_descaled**2
            w_dec_tf = 1.0 / dec_descaled**2
            # Arithmetic mean
            for idx in g.indices:
                weights_ra_arr[idx] = (w_ra_file + w_ra_tf) / 2.0
                weights_dec_arr[idx] = (w_dec_file + w_dec_tf) / 2.0

        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "hybrid_arithmetic",
            set_id,
            "timeframe",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "HybridArithmeticWeight [%s]: %d timeframe(s), IDv2 RA σ=%.2e DEC σ=%.2e",
            set_id,
            len(tf_groups),
            ra_id_v2_sigma,
            dec_id_v2_sigma,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Fixed sigma strategy
# ---------------------------------------------------------------------------


@register_weight_strategy("fixed")
class FixedWeight(WeightStrategy):
    """Fixed-sigma weighting from user-specified sigmas in the grouping config.

    Assigns weights from known sigmas (``fixed_sigmas`` in the grouping config)
    rather than computing them from residuals.  Useful when the precision is
    known a priori (e.g. Gaia) or when reproducing published results.

    Sigmas are specified per group-level identifier (section name or set id).
    Observers not listed fall back to the set-level RMSE.
    """

    def __init__(self, fixed_sigmas: dict[str, dict[str, float]] | None = None):
        self._fixed_sigmas: dict[str, dict[str, float]] | None = fixed_sigmas or None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigma_rad(value_arcsec: float) -> float:
        """Convert arcseconds to radians."""
        return value_arcsec * _ARCSEC_TO_RAD

    @staticmethod
    def _resolve_entry(
        entry: dict[str, float],
    ) -> tuple[float, float] | None:
        """Extract (ra_sigma_arcsec, dec_sigma_arcsec) from a fixed_sigmas entry.

        Supports both per-component (``ra``, ``dec``) and single-value
        (``sigma``) syntax.
        """
        single = entry.get("sigma")
        if single is not None and single > 0:
            return single, single
        ra = entry.get("ra", 0.0)
        dec = entry.get("dec", 0.0)
        if ra > 0 and dec > 0:
            return ra, dec
        return None

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

        fixed_sigmas = self._fixed_sigmas or {}

        # Pre-compute per-group fixed sigmas if any groups match in fixed_sigmas
        group_sigmas: dict[str, tuple[float, float]] = {}
        for g in groups:
            entry = fixed_sigmas.get(g.group_id)
            if entry is not None:
                resolved = self._resolve_entry(entry)
                if resolved is not None:
                    group_sigmas[g.group_id] = resolved

        # Per-set fixed sigma
        set_entry = fixed_sigmas.get(set_id)
        set_resolved = self._resolve_entry(set_entry) if set_entry else None

        # Build per-observation weights
        weights_ra_arr = np.zeros(n_obs)
        weights_dec_arr = np.zeros(n_obs)
        used_fixed = False

        if group_sigmas or set_resolved is not None:
            used_fixed = True
            for i in range(n_obs):
                # Check if this observation belongs to a group with a fixed sigma
                ra_sigma_asc = None
                dec_sigma_asc = None
                for g in groups:
                    if i in g.indices and g.group_id in group_sigmas:
                        ra_sigma_asc, dec_sigma_asc = group_sigmas[g.group_id]
                        break
                if ra_sigma_asc is None and set_resolved is not None:
                    ra_sigma_asc, dec_sigma_asc = set_resolved
                if ra_sigma_asc is not None:
                    ra_sigma_rad = max(ra_sigma_asc * _ARCSEC_TO_RAD, min_sigma_rad)
                    dec_sigma_rad = max(dec_sigma_asc * _ARCSEC_TO_RAD, min_sigma_rad)
                    weights_ra_arr[i] = 1.0 / ra_sigma_rad**2
                    weights_dec_arr[i] = 1.0 / dec_sigma_rad**2
                else:
                    # Fallback: residual-driven RMSE
                    pass  # will fill below

        # Fill any zeros (fallback to residual-driven RMSE)
        zero_mask = weights_ra_arr == 0.0
        if np.any(zero_mask):
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            weights_ra_arr[zero_mask] = 1.0 / ra_sigma**2
            weights_dec_arr[zero_mask] = 1.0 / dec_sigma**2

        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            groups,
            weights_ra_arr,
            weights_dec_arr,
            "fixed",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "FixedWeight [%s]: used_fixed=%s",
            set_id,
            used_fixed,
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


def _observable_type_str(observation_set: obs.SingleObservationSet) -> str:
    """Return a human-readable string for the observable type.

    Maps Tudat's ``ObservableType`` enum to ``"absolute"`` or ``"relative"``.
    """
    from tudatpy.estimation.observable_models_setup import model_settings as obs_model_settings

    obs_type = observation_set.observable_type
    # relative_angular_position_type = 9 in TudatPy
    if obs_type == obs_model_settings.ObservableType(9):
        return "relative"
    # angular_position_type = 1 (absolute)
    return "absolute"
