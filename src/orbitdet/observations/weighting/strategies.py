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
