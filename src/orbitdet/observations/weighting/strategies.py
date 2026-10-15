"""Concrete weighting strategies.

Each strategy is a class inheriting from :class:`WeightStrategy` and registered
via the :func:`register_weight_strategy` decorator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from ..grouping import Group, GroupList
from .base import WeightStrategy
from .dataframe import (
    build_weights_df as _build_weights_df,
)
from .dataframe import (
    interleave_weights as _interleave_weights,
)
from .dataframe import (
    observable_type_str as _observable_type_str,
)
from .dataframe import (
    residuals_array as _residuals_array,
)
from .dataframe import (
    rms as _rms,
)
from .gaia_cache import get_gaia_weights
from .layered import LayeredWeight  # noqa: F401  (registers the "layered" strategy)
from .registry import register_weight_strategy
from .router import RouterWeight  # noqa: F401  (registers the "router" strategy)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


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
# Per-set strategy dispatch
# ---------------------------------------------------------------------------


@register_weight_strategy("per_set")
class PerSetWeight(WeightStrategy):
    """Per-set strategy dispatch.

    Routes each observation set to a sub-strategy based on the set identifier.
    Sets not listed in the ``per_set`` mapping fall back to ID v2 residual-driven
    weighting.

    Configuration example::

        weighting:
          enabled: true
          strategy: per_set
          min_sigma_arcsec: 0.001
          grouping:
            levels:
              - type: section
                sections:
                  - {name: "pre_encounter", start: -1e30, end: 618600000.0}
                  - {name: "post_encounter", start: 618600000.0, end: 1e30}
              - type: timeframe
                gap_threshold_hours: 4.0
          per_set:
            "Voyager 2":
              strategy: fixed
              fixed_sigmas:
                "Voyager 2_pre_encounter": {ra: 1.0, dec: 1.0}
                "Voyager 2_post_encounter": {ra: 4.0, dec: 4.0}
            "Gaia":
              strategy: gaia_archive
    """

    def __init__(self, per_set_config: dict[str, dict] | None = None):
        self._per_set: dict[str, dict] = per_set_config or {}
        self._strategies: dict[str, WeightStrategy] = {}
        # Mirror sub-strategy preservation so the engine honours it for routed sets.
        self._preserve_full_weight_matrix = False

    @property
    def preserve_full_weight_matrix(self):
        """True when any routed sub-strategy preserves the full archive matrix.

        :class:`WeightEngine` checks this on the top-level strategy, so it must
        reflect the sub-strategies (e.g. a ``gaia_archive`` route).  If *any*
        routed sub-strategy preserves the full matrix, every set carrying one is
        left untouched by the engine; the non-preserving sub-strategies still
        assign base weights to sets without a full matrix.
        """
        return self._preserve_full_weight_matrix

    def _get_strategy(self, set_id: str) -> WeightStrategy | None:
        """Return (or create) the sub-strategy for *set_id*."""
        if set_id in self._strategies:
            return self._strategies[set_id]

        entry = self._per_set.get(set_id)
        if entry is None:
            return None

        sub_name = entry.get("strategy", "id_v2")
        sub_cls = get_strategy_class(sub_name)

        if sub_name == "fixed":
            fixed_sigmas = {k: dict(v) for k, v in entry.get("fixed_sigmas", {}).items()}
            sub_instance = sub_cls(fixed_sigmas=fixed_sigmas)
        elif sub_name == "gaia_archive":
            gaia_cfg = entry.get("gaia_archive", {}) or {}
            sub_instance = sub_cls(
                covariance_mode=gaia_cfg.get("covariance_mode", "full_matrix"),
                scale=gaia_cfg.get("scale", 1.0),
            )
        else:
            sub_instance = sub_cls()

        if getattr(sub_instance, "preserve_full_weight_matrix", False):
            self._preserve_full_weight_matrix = True

        self._strategies[set_id] = sub_instance
        return sub_instance

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        sub = self._get_strategy(set_id)
        if sub is not None:
            return sub.compute_weights(observation_set, groups, set_id, min_sigma_arcsec)

        # Fallback: ID v2 residual-driven weighting
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_ra_arr = np.full(n_obs, w_ra)
            weights_dec_arr = np.full(n_obs, w_dec)
        else:
            ra_id_v2_sigma, dec_id_v2_sigma = _id_v2_sigmas(residuals_rad, tf_groups, min_sigma_rad)
            w_ra = 1.0 / ra_id_v2_sigma**2
            w_dec = 1.0 / dec_id_v2_sigma**2
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
            "per_set",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "PerSetWeight [%s]: no per_set entry, using ID v2 fallback (RA σ=%.2e, DEC σ=%.2e)",
            set_id,
            ra_sigma if not tf_groups else ra_id_v2_sigma,
            dec_sigma if not tf_groups else dec_id_v2_sigma,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Gaia archive strategy (off-diagonal weights from Gaia covariance)
# ---------------------------------------------------------------------------


@register_weight_strategy("gaia_archive")
class GaiaArchiveWeight(WeightStrategy):
    """Gaia archive weighting from the Gaia transit covariance.

    The Gaia archive covariance (random + systematic, with RA/DEC correlations)
    is turned into a weight matrix by the tudatpy ``GaiaAstrometry`` interface
    (via :class:`~orbitdet.data.gaia.GaiaQuery`) and carried by the observation
    sets.  Within a transit this couples every RA/DEC sample to every other
    sample (a ``2N x 2N`` block for ``N`` CCD crossings), so the full
    off-diagonal weight matrix is the only scheme here that accounts for the
    Gaia intra-transit systematic correlation.

    This strategy is configurable through ``weighting.gaia_archive``:

    ``covariance_mode``
        * ``"full_matrix"`` (default) - use the archive's full transit weight
          matrix as-is.  The :class:`WeightEngine` leaves the set untouched.
        * ``"diagonal"`` - use only the per-observation diagonal of the archive
          weight matrix, i.e. ignore the intra-transit correlations.  The
          cached diagonal is applied as base (per-observation) weights.
    ``scale``
        Multiplicative factor applied to the archive weights (default ``1.0``).
        Use ``> 1`` to down-weight Gaia relative to other datasets, or ``< 1``
        to up-weight it.  In ``full_matrix`` mode the archived transit matrix is
        rescaled; in ``diagonal`` mode the returned base weights are rescaled.

    If no cached archive weights are found for a set, the strategy falls back to
    the ID v2 residual-driven weighting.
    """

    def __init__(
        self,
        covariance_mode: str = "full_matrix",
        scale: float = 1.0,
    ):
        """Configure the Gaia archive weighting.

        Parameters
        ----------
        covariance_mode : str
            ``"full_matrix"`` (default) to use the full off-diagonal transit
            weight matrix, or ``"diagonal"`` to use only its per-observation
            diagonal.
        scale : float
            Positive multiplicative factor applied to the archive weights.
        """
        if covariance_mode not in ("full_matrix", "diagonal"):
            raise ValueError(
                "GaiaArchiveWeight: 'covariance_mode' must be 'full_matrix' or "
                f"'diagonal', got {covariance_mode!r}."
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"GaiaArchiveWeight: 'scale' must be a positive finite number, got {scale!r}."
            )
        self.covariance_mode = covariance_mode
        self.scale = float(scale)

    @property
    def preserve_full_weight_matrix(self) -> bool:
        """Whether the engine should leave the set's full weight matrix untouched.

        True in ``full_matrix`` mode (the archive matrix IS the weighting); False
        in ``diagonal`` mode (the engine clears the full matrix and applies the
        cached diagonal as base weights).
        """
        return self.covariance_mode == "full_matrix"

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

        # Try to retrieve cached Gaia archive weights
        cached = get_gaia_weights(set_id)
        if cached is not None and len(cached) == 2 * n_obs:
            weights_array = np.asarray(cached, dtype=float).copy()
            if self.scale != 1.0:
                weights_array = weights_array * self.scale
                if self.covariance_mode == "full_matrix" and getattr(
                    observation_set, "has_full_weight_matrix_contribution", False
                ):
                    # Rescale the archived transit matrix in place; the engine preserves it.
                    observation_set.set_full_weight_matrix(
                        np.asarray(observation_set.full_weight_matrix, dtype=float) * self.scale
                    )
            # Extract RA/DEC weights for the metadata DataFrame
            weights_ra_arr = weights_array[0::2]
            weights_dec_arr = weights_array[1::2]
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                groups,
                weights_ra_arr,
                weights_dec_arr,
                "gaia_archive",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
            )
            logger.debug(
                "GaiaArchiveWeight [%s]: mode=%s, scale=%.3g, using cached archive weights "
                "(RA w range=[%.2e, %.2e], DEC w range=[%.2e, %.2e])",
                set_id,
                self.covariance_mode,
                self.scale,
                weights_ra_arr.min(),
                weights_ra_arr.max(),
                weights_dec_arr.min(),
                weights_dec_arr.max(),
            )
            return weights_array, df

        # Fallback: no cached weights — use ID v2 residual-driven weighting
        logger.warning(
            "GaiaArchiveWeight [%s]: no cached archive weights found "
            "(set_id='%s', n_obs=%d). Falling back to ID v2 residual-driven weighting.",
            set_id,
            set_id,
            n_obs,
        )
        tf_groups = groups.by_level("timeframe")
        if not tf_groups:
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_ra_arr = np.full(n_obs, w_ra)
            weights_dec_arr = np.full(n_obs, w_dec)
        else:
            ra_id_v2_sigma, dec_id_v2_sigma = _id_v2_sigmas(residuals_rad, tf_groups, min_sigma_rad)
            w_ra = 1.0 / ra_id_v2_sigma**2
            w_dec = 1.0 / dec_id_v2_sigma**2
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
            "gaia_archive",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "GaiaArchiveWeight [%s]: fallback ID v2 (RA σ=%.2e, DEC σ=%.2e)",
            set_id,
            ra_id_v2_sigma if tf_groups else ra_sigma,
            dec_id_v2_sigma if tf_groups else dec_sigma,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Voyager fixed (pre/post encounter) + Gaia archive strategy
# ---------------------------------------------------------------------------


@register_weight_strategy("voyager_fixed_gaia_archive")
class VoyagerFixedGaiaArchiveWeight(WeightStrategy):
    """Voyager fixed pre/post-encounter sigmas combined with Gaia archive weights.

    This is a purpose-built combination of the two thesis-motivated presets:

    * **Voyager 2** sets receive *fixed* sigmas split at the Triton encounter
      epoch — :math:`1.0''` per axis before encounter and :math:`4.0''` after
      (see ``voyager_fixed.yaml``).
    * **Gaia** sets keep the *full* off-diagonal archive transit weight matrix
      (random + systematic, with RA/DEC correlations) as delivered by the
      tudatpy ``GaiaAstrometry`` interface (see ``gaia_archive.yaml``).
    * **Every other set** (e.g. ground-based astrometry) falls back to the
      residual-driven **ID v2** weighting.

    Unlike the generic ``per_set`` dispatch, this strategy exposes
    :attr:`preserve_full_weight_matrix` = ``True`` whenever Gaia is routed to
    the full archive matrix, so :class:`WeightEngine` leaves a Gaia set that
    already carries a full weight-matrix contribution untouched instead of
    clearing it and applying diagonal weights.  This is the property a
    ``per_set`` configuration cannot express on its own.

    Configuration (``weighting:`` node)::

        weighting:
          enabled: true
          strategy: voyager_fixed_gaia_archive
          min_sigma_arcsec: 0.005
          voyager_fixed_gaia_archive:
            encounter_epoch: 618600000.0        # s past J2000 (optional)
            voyager_set_ids: ["Voyager 2"]      # optional
            gaia_set_ids: ["Gaia"]              # optional
            gaia_covariance_mode: full_matrix   # optional
            gaia_scale: 1.0                     # optional
          grouping:
            levels:
              - type: section
                sections:
                  - {name: "pre_encounter", start: -1e30, end: 618600000.0}
                  - {name: "post_encounter", start: 618600000.0, end: 1e30}
              - type: timeframe
                gap_threshold_hours: 8.0
                min_obs_per_frame: 1
    """

    #: Default section bounds for the Voyager 2 encounter split (seconds past J2000).
    DEFAULT_SECTION_BOUNDS: tuple[tuple[str, float, float], ...] = (
        ("pre_encounter", -1e30, 618600000.0),
        ("post_encounter", 618600000.0, 1e30),
    )
    #: Default set-identifier substrings.
    DEFAULT_VOYAGER_SET_IDS: tuple[str, ...] = ("Voyager 2",)
    DEFAULT_GAIA_SET_IDS: tuple[str, ...] = ("Gaia",)
    #: Default fixed sigmas (arcsec) applied to the pre/post encounter sections.
    DEFAULT_PRE_SIGMA_ARCSEC: float = 1.0
    DEFAULT_POST_SIGMA_ARCSEC: float = 4.0

    def __init__(
        self,
        voyager_set_ids: list[str] | None = None,
        gaia_set_ids: list[str] | None = None,
        voyager_fixed_sigmas: dict[str, dict[str, float]] | None = None,
        section_bounds: list[tuple[str, float, float]] | None = None,
        gaia_covariance_mode: str = "full_matrix",
        gaia_scale: float = 1.0,
    ):
        """Configure the combined Voyager-fixed + Gaia-archive weighting.

        Parameters
        ----------
        voyager_set_ids : list[str], optional
            Set identifiers treated as Voyager 2.  Defaults to ``["Voyager 2"]``.
        gaia_set_ids : list[str], optional
            Set identifiers treated as Gaia.  Defaults to ``["Gaia"]``.
        voyager_fixed_sigmas : dict[str, dict[str, float]], optional
            Explicit ``fixed_sigmas`` mapping (group-id -> ``{ra, dec}``).  When
            omitted, the sigmas are derived from the section configuration.
        section_bounds : list[tuple[str, float, float]], optional
            Explicit ``(name, start, end)`` section bounds.  When omitted, the
            bounds are read from the grouping config; failing that, the
            :attr:`DEFAULT_SECTION_BOUNDS` (encounter at ``618600000.0``) are used.
        gaia_covariance_mode : str
            ``"full_matrix"`` (default) keeps the archive transit matrix;
            ``"diagonal"`` applies only its per-observation diagonal.
        gaia_scale : float
            Positive multiplicative factor applied to the archive weights.
        """
        self._voyager_set_ids = (
            list(voyager_set_ids) if voyager_set_ids else list(self.DEFAULT_VOYAGER_SET_IDS)
        )
        self._gaia_set_ids = (
            list(gaia_set_ids) if gaia_set_ids else list(self.DEFAULT_GAIA_SET_IDS)
        )
        self._voyager_fixed_sigmas = voyager_fixed_sigmas
        self._section_bounds = section_bounds
        self._grouping = None
        self._voyager_strategy = FixedWeight(
            fixed_sigmas=voyager_fixed_sigmas if voyager_fixed_sigmas else None
        )
        self._gaia_strategy = GaiaArchiveWeight(
            covariance_mode=gaia_covariance_mode,
            scale=gaia_scale,
        )

    def set_grouping(self, grouping) -> None:
        """Attach the Hydra grouping config so section sigmas can be derived.

        Called by :meth:`WeightEngine.from_config`; harmless if never called.
        """
        self._grouping = grouping

    @property
    def preserve_full_weight_matrix(self):
        """True unless Gaia is routed to the diagonal-only archive weighting.

        In ``full_matrix`` mode (the default) the Gaia archive matrix *is* the
        Gaia weighting, so :class:`WeightEngine` must leave Gaia sets untouched.
        In ``diagonal`` mode the archive matrix is cleared and the cached
        diagonal is applied as base weights instead.
        """
        return self._gaia_strategy.preserve_full_weight_matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_bounds_from_grouping(grouping) -> list[tuple[str, float, float]] | None:
        """Extract ``(name, start, end)`` section bounds from a grouping config."""
        if grouping is None:
            return None
        try:
            from omegaconf import OmegaConf
        except Exception:  # pragma: no cover - omegaconf is a hard dependency
            return None
        levels = OmegaConf.select(grouping, "levels")
        if not levels:
            return None
        for level_entry in levels:
            if OmegaConf.select(level_entry, "type") != "section":
                continue
            sections = OmegaConf.select(level_entry, "sections")
            if not sections:
                continue
            bounds: list[tuple[str, float, float]] = []
            for sc in sections:
                name = OmegaConf.select(sc, "name")
                start = OmegaConf.select(sc, "start")
                end = OmegaConf.select(sc, "end")
                if name is None or start is None or end is None:
                    continue
                bounds.append((str(name), float(start), float(end)))
            if bounds:
                return bounds
        return None

    @staticmethod
    def _label_to_ra_dec(label: str) -> tuple[float, float] | None:
        """Parse ``"pre_encounter_1"`` / ``"post_encounter_4-5"`` style labels.

        Returns ``(ra_sigma_arcsec, dec_sigma_arcsec)`` or ``None`` if the label
        does not encode a sigma.
        """
        if "encounter" not in label.lower():
            return None
        # Grab the value(s) following the final underscore
        tail = label.rsplit("_", 1)[-1] if "_" in label else ""
        if not tail:
            return None
        parts = tail.split("-")
        try:
            values = [float(p) for p in parts if p != ""]
        except ValueError:
            return None
        if not values:
            return None
        if len(values) == 1:
            return values[0], values[0]
        return values[0], values[1]

    def _build_section_bounds(self) -> list[tuple[str, float, float]]:
        """Resolve the section bounds (explicit -> grouping -> defaults)."""
        if self._section_bounds:
            return list(self._section_bounds)
        from_grouping = self._section_bounds_from_grouping(self._grouping)
        if from_grouping:
            return from_grouping
        return list(self.DEFAULT_SECTION_BOUNDS)

    def _derive_fixed_sigmas(self, set_id: str) -> dict[str, dict[str, float]]:
        """Derive per-section ``fixed_sigmas`` from the resolved section bounds.

        Keys match :func:`partition_by_sections` group ids (``"<set>_<name>"``).
        When a section name encodes the sigmas (e.g. ``pre_encounter_1``) those
        win; otherwise the thesis defaults are used (1" pre, 4" post).
        """
        sigmas: dict[str, dict[str, float]] = {}
        for name, _start, _end in self._build_section_bounds():
            parsed = self._label_to_ra_dec(name)
            if parsed is not None:
                ra, dec = parsed
            elif "pre" in name.lower():
                ra, dec = self.DEFAULT_PRE_SIGMA_ARCSEC, self.DEFAULT_PRE_SIGMA_ARCSEC
            elif "post" in name.lower():
                ra, dec = self.DEFAULT_POST_SIGMA_ARCSEC, self.DEFAULT_POST_SIGMA_ARCSEC
            else:
                continue
            sigmas[f"{set_id}_{name}"] = {"ra": ra, "dec": dec}
        return sigmas

    def _sub_voyager(self, set_id: str) -> FixedWeight:
        """Return a Voyager FixedWeight strategy, building sigmas lazily if needed."""
        if self._voyager_fixed_sigmas:
            return self._voyager_strategy
        # Derive from the section configuration the first time we see a Voyager set.
        return FixedWeight(fixed_sigmas=self._derive_fixed_sigmas(set_id))

    @staticmethod
    def _is_gaia(set_id: str, gaia_ids: list[str]) -> bool:
        """Match a Gaia set id (exact, substring, or the generic ``"Gaia"``)."""
        low = set_id.lower()
        for g in gaia_ids:
            gl = g.lower()
            if low == gl or gl in low or "gaia" in low:
                return True
        return False

    def _is_voyager(self, set_id: str, voyager_ids: list[str]) -> bool:
        """Match a Voyager set id (exact, substring, or a ``"Voyager"`` prefix)."""
        low = set_id.lower()
        for v in voyager_ids:
            vl = v.lower()
            if low == vl or vl in low:
                return True
        return "voyager" in low

    # ------------------------------------------------------------------
    # WeightStrategy API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if self._is_gaia(set_id, self._gaia_set_ids):
            return self._gaia_strategy.compute_weights(
                observation_set, groups, set_id, min_sigma_arcsec
            )
        if self._is_voyager(set_id, self._voyager_set_ids):
            return self._sub_voyager(set_id).compute_weights(
                observation_set, groups, set_id, min_sigma_arcsec
            )
        # Everything else: residual-driven ID v2 weighting.
        return IDv2Weight().compute_weights(observation_set, groups, set_id, min_sigma_arcsec)

