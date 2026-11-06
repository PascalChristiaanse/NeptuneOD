"""Concrete weighting strategies.

Each strategy is a class inheriting from :class:`WeightStrategy` and registered
via the :func:`register_weight_strategy` decorator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from ..clustering import Cluster, ClusterList
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
from .gaia_cache import get_gaia_covariance_components, get_gaia_weights
from .layered import LayeredWeight  # noqa: F401  (registers the "layered" strategy)
from .registry import register_weight_strategy

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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        # Collect timeframe clusters
        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
            # Fallback: no timeframes defined — use the set as one cluster
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_array = _interleave_weights(np.full(n_obs, w_ra), np.full(n_obs, w_dec))
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                clusters,
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
        for g in tf_clusters:
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
            clusters,
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
            len(tf_clusters),
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
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
        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
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
                clusters,
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

        for g in tf_clusters:
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
            clusters,
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
            len(tf_clusters),
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
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
            clusters,
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
    tf_clusters: list,
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
    tf_clusters : list of Cluster
        Timeframe clusters.
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
    for g in tf_clusters:
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
            # Fallback: no timeframes — use set-level as a single timeframe
            tf_clusters = [
                Cluster(indices=np.arange(n_obs, dtype=int), cluster_id=set_id, level="timeframe")
            ]

        weights_ra_arr, weights_dec_arr = _descaled_tf_rmse_and_weights(
            residuals_rad,
            tf_clusters,
            min_sigma_rad,
        )
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
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
            len(tf_clusters),
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        # Ignore min_sigma_arcsec — effectively min_sigma = 0
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
            tf_clusters = [
                Cluster(indices=np.arange(n_obs, dtype=int), cluster_id=set_id, level="timeframe")
            ]

        weights_ra_arr, weights_dec_arr = _descaled_tf_rmse_and_weights(
            residuals_rad,
            tf_clusters,
            min_sigma_rad=0.0,
        )
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)

        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
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
            len(tf_clusters),
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# ID v2 helper — compute ID v2 (scaled per-file) sigma from timeframe clusters
# ---------------------------------------------------------------------------


def _id_v2_sigmas(
    residuals_rad: np.ndarray,
    tf_clusters: list,
    min_sigma_rad: float,
) -> tuple[float, float]:
    """Compute ID v2 per-file sigmas from descaled per-timeframe RMSEs.

    .. math::

        \\sigma_{\\text{RA,id_v2}} =
        \\text{RMS}\\left(\\{\\sigma_{\\text{RA},t} \\cdot \\sqrt{n_t}\\}_{t=1}^{T_f}\\right)
    """
    descaled_ra: list[float] = []
    descaled_dec: list[float] = []
    for g in tf_clusters:
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
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
                clusters,
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
            tf_clusters,
            min_sigma_rad,
        )
        w_ra_file = 1.0 / ra_id_v2_sigma**2
        w_dec_file = 1.0 / dec_id_v2_sigma**2

        # Descaled TF weight per timeframe
        weights_ra_arr = np.empty(n_obs)
        weights_dec_arr = np.empty(n_obs)
        for g in tf_clusters:
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
            clusters,
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
            len(tf_clusters),
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
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
                clusters,
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
            tf_clusters,
            min_sigma_rad,
        )
        w_ra_file = 1.0 / ra_id_v2_sigma**2
        w_dec_file = 1.0 / dec_id_v2_sigma**2

        # Descaled TF weight per timeframe
        weights_ra_arr = np.empty(n_obs)
        weights_dec_arr = np.empty(n_obs)
        for g in tf_clusters:
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
            clusters,
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
            len(tf_clusters),
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
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
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
                clusters,
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
        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
            w_ra = 1.0 / ra_sigma**2
            w_dec = 1.0 / dec_sigma**2
            weights_ra_arr = np.full(n_obs, w_ra)
            weights_dec_arr = np.full(n_obs, w_dec)
        else:
            ra_id_v2_sigma, dec_id_v2_sigma = _id_v2_sigmas(residuals_rad, tf_clusters, 
                min_sigma_rad)
            w_ra = 1.0 / ra_id_v2_sigma**2
            w_dec = 1.0 / dec_id_v2_sigma**2
            weights_ra_arr = np.full(n_obs, w_ra)
            weights_dec_arr = np.full(n_obs, w_dec)

        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)
        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
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
            ra_id_v2_sigma if tf_clusters else ra_sigma,
            dec_id_v2_sigma if tf_clusters else dec_sigma,
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Gaia component (random/systematic) strategy
# ---------------------------------------------------------------------------


def _blocks_to_full_matrix(blocks: list) -> np.ndarray:
    """Concatenate per-transit covariance/weight blocks into one block-diagonal matrix."""
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=float)
    from scipy.linalg import block_diag

    return block_diag(*[np.asarray(block, dtype=float) for block in blocks])


def _scan_frame_rotation(scan_angles: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return per-observation ``(RA, Dec) -> (AL, AC)`` rotation matrices.

    The scan frame is rotated by the position angle of scan ``psi`` and scaled by
    ``cos(dec)`` because RA is a longitude::

        east  = dRA * cos(dec),   north = dDec
        AL    = east * sin(psi) + north * cos(psi)
        AC    = east * cos(psi) - north * sin(psi)

    so ``T`` maps a ``(RA, Dec)`` offset to ``(AL, AC)`` and a covariance
    transforms as ``C_scan = T C T^T``.  This is the same convention
    :class:`FixedScanWeight` uses.
    """
    cos_dec = np.cos(dec)
    sin_psi = np.sin(scan_angles)
    cos_psi = np.cos(scan_angles)
    rotation = np.empty((len(scan_angles), 2, 2))
    rotation[:, 0, 0] = sin_psi * cos_dec
    rotation[:, 0, 1] = cos_psi
    rotation[:, 1, 0] = cos_psi * cos_dec
    rotation[:, 1, 1] = -sin_psi
    return rotation


def _scale_in_scan_frame(
    matrix: np.ndarray,
    rotation: np.ndarray,
    scale_al: float,
    scale_ac: float,
) -> np.ndarray:
    """Scale a ``(RA, Dec)`` covariance along/across scan, then rotate back.

    With ``S = diag(scale_al, scale_ac)`` and ``A = blockdiag(T_k^{-1} S T_k)``
    the scaled covariance is the similarity transform ``A C A^T``.  Writing it
    this way handles the random component (block diagonal) and the systematic
    component (dense within a transit) with the same expression: the systematic
    part is one physical offset per transit, so its ``(i, j)`` entry is rotated
    by observation ``i`` on one side and observation ``j`` on the other.
    """
    if scale_al == 1.0 and scale_ac == 1.0:
        return matrix

    scale = np.diag([scale_al, scale_ac])
    n_obs = rotation.shape[0]
    transform = np.zeros((2 * n_obs, 2 * n_obs))
    for k in range(n_obs):
        rows = slice(2 * k, 2 * k + 2)
        transform[rows, rows] = np.linalg.inv(rotation[k]) @ scale @ rotation[k]
    return transform @ matrix @ transform.T


@register_weight_strategy("gaia_components")
class GaiaComponentsWeight(WeightStrategy):
    r"""Gaia archive weighting with independent random/systematic scaling.

    The Gaia archive covariance has two parts: a **random** component, which
    varies from CCD crossing to crossing, and a **systematic** component, which
    is constant across a whole transit and therefore correlates every pair of
    observations within it.  :class:`GaiaArchiveWeight` scales the *total*
    covariance with a single factor; this strategy instead scales the two parts
    independently and recombines them::

        C = alpha_random * C_random + alpha_systematic * C_systematic
        W = C^{-1}

    so the intra-transit systematic correlation can be tuned (down-weighted by
    ``systematic_scale < 1``, emphasised by ``> 1``) without touching the random
    noise level, or vice versa.

    Each component can be scaled **per scan axis** as well.  The archive blocks
    are stored in ``(RA, Dec)``, but Gaia's error budget is naturally expressed
    along and across the scan direction, so the four factors are applied in the
    scan frame: each block is rotated by the observation's position angle of
    scan, scaled by ``diag(scale_al, scale_ac)``, and rotated back.  The
    single-factor ``random_scale`` / ``systematic_scale`` remain as a shorthand
    that sets both axes at once.

    Like :class:`GaiaArchiveWeight` this is a full-matrix scheme — the resulting
    ``W`` has off-diagonal transit blocks — so it sets
    :attr:`preserve_full_weight_matrix` to True and rewrites the observation
    set's full weight matrix.  The component covariance blocks are read from the
    cache populated by the Gaia observation factory; if none are available the
    strategy falls back to the ID v2 residual-driven weighting.

    This strategy is configurable through ``weighting.gaia_components``:

    ``random_scale_al`` / ``random_scale_ac``
        Multiplicative factors applied to the random covariance component along
        and across the scan direction (default ``1.0``).  Use ``> 1`` to inflate
        the random noise (down-weight) or ``< 1`` to tighten it.
    ``systematic_scale_al`` / ``systematic_scale_ac``
        The same, for the systematic covariance component.
    ``random_scale`` / ``systematic_scale``
        Shorthand that sets both axes of the corresponding component.  Ignored
        for an axis that is given explicitly.
    """

    def __init__(
        self,
        random_scale: float | None = None,
        systematic_scale: float | None = None,
        random_scale_al: float | None = None,
        random_scale_ac: float | None = None,
        systematic_scale_al: float | None = None,
        systematic_scale_ac: float | None = None,
    ):
        """Configure the Gaia component weighting.

        Parameters
        ----------
        random_scale : float, optional
            Shorthand for setting both ``random_scale_al`` and
            ``random_scale_ac``.  Defaults to ``1.0`` where not overridden.
        systematic_scale : float, optional
            Shorthand for setting both ``systematic_scale_al`` and
            ``systematic_scale_ac``.
        random_scale_al : float, optional
            Along-scan factor for the random covariance component.
        random_scale_ac : float, optional
            Across-scan factor for the random covariance component.
        systematic_scale_al : float, optional
            Along-scan factor for the systematic covariance component.
        systematic_scale_ac : float, optional
            Across-scan factor for the systematic covariance component.
        """
        self.random_scale_al = self._resolve_scale(
            "random_scale_al", random_scale_al, random_scale
        )
        self.random_scale_ac = self._resolve_scale(
            "random_scale_ac", random_scale_ac, random_scale
        )
        self.systematic_scale_al = self._resolve_scale(
            "systematic_scale_al", systematic_scale_al, systematic_scale
        )
        self.systematic_scale_ac = self._resolve_scale(
            "systematic_scale_ac", systematic_scale_ac, systematic_scale
        )

    @staticmethod
    def _resolve_scale(name: str, explicit: float | None, shorthand: float | None) -> float:
        """Pick the per-axis factor, falling back to the shorthand, then to 1.0."""
        value = explicit if explicit is not None else shorthand
        if value is None:
            return 1.0
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"GaiaComponentsWeight: '{name}' must be a positive finite number, got {value!r}."
            )
        return float(value)

    @property
    def preserve_full_weight_matrix(self) -> bool:
        """The component weights *are* the (reweighted) full transit matrix."""
        return True

    def _component_weight_matrix(
        self,
        set_id: str,
        scan_angles: np.ndarray | None = None,
        dec: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Build ``W = (alpha_r C_r + alpha_s C_s)^{-1}`` from the cached components.

        The per-axis factors are applied in the scan frame, which needs the
        position angle of scan and the declination of every observation.  When
        those are unavailable the factors are applied directly in ``(RA, Dec)``
        — exact for the isotropic case (``scale_al == scale_ac``) and a
        documented approximation otherwise.

        Returns ``None`` if the components for *set_id* are not cached or do not
        describe a single, consistently-sized system.
        """
        cached = get_gaia_covariance_components(set_id)
        if cached is None:
            return None
        random_blocks, systematic_blocks = cached
        if not random_blocks or len(random_blocks) != len(systematic_blocks):
            return None

        random_matrix = _blocks_to_full_matrix(random_blocks)
        systematic_matrix = _blocks_to_full_matrix(systematic_blocks)

        rotation = self._rotation_for(random_matrix.shape[0], scan_angles, dec)
        if rotation is None:
            random_matrix = self._scale_isotropic(
                random_matrix, self.random_scale_al, self.random_scale_ac
            )
            systematic_matrix = self._scale_isotropic(
                systematic_matrix, self.systematic_scale_al, self.systematic_scale_ac
            )
        else:
            random_matrix = _scale_in_scan_frame(
                random_matrix, rotation, self.random_scale_al, self.random_scale_ac
            )
            systematic_matrix = _scale_in_scan_frame(
                systematic_matrix, rotation, self.systematic_scale_al, self.systematic_scale_ac
            )

        covariance = random_matrix + systematic_matrix

        weight_matrix = np.linalg.inv(covariance)
        # Force symmetry: the inversion introduces numerical asymmetry, and Tudat's
        # normal equations assume a symmetric weight matrix.
        return (weight_matrix + weight_matrix.T) / 2.0

    @staticmethod
    def _rotation_for(
        n_entries: int, scan_angles: np.ndarray | None, dec: np.ndarray | None
    ) -> np.ndarray | None:
        """Return the per-observation rotations, or ``None`` if they are unavailable."""
        if scan_angles is None or dec is None:
            return None
        n_obs = n_entries // 2
        if len(scan_angles) != n_obs or len(dec) != n_obs:
            return None
        return _scan_frame_rotation(
            np.asarray(scan_angles, dtype=float), np.asarray(dec, dtype=float)
        )

    @staticmethod
    def _scale_isotropic(matrix: np.ndarray, scale_al: float, scale_ac: float) -> np.ndarray:
        """Scale the ``(RA, Dec)`` axes directly, for when scan angles are missing."""
        if scale_al == 1.0 and scale_ac == 1.0:
            return matrix
        scaled = matrix.copy()
        scaled[0::2, :] *= scale_al
        scaled[:, 0::2] *= scale_al
        scaled[1::2, :] *= scale_ac
        scaled[:, 1::2] *= scale_ac
        return scaled

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        min_sigma_rad = min_sigma_arcsec * _ARCSEC_TO_RAD
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        # The per-axis factors are applied in the scan frame, which needs the
        # position angle of scan and the declination of each observation.  Both
        # are only read when the scan angles are actually available, so a set
        # without them never touches the observation array.
        scan_angles = view.scan_angles if view is not None else None
        dec = None
        if scan_angles is not None:
            observed = np.asarray(observation_set.concatenated_observations, dtype=float)
            dec = observed.reshape(-1, 2)[:, 1]

        weight_matrix = self._component_weight_matrix(set_id, scan_angles=scan_angles, dec=dec)
        if weight_matrix is not None and weight_matrix.shape[0] == 2 * n_obs:
            # Rewrite the full transit matrix in place; the engine preserves it.
            if getattr(observation_set, "has_full_weight_matrix_contribution", False):
                observation_set.set_full_weight_matrix(weight_matrix)
            weights_array = np.diag(weight_matrix).copy()
            weights_ra_arr = weights_array[0::2]
            weights_dec_arr = weights_array[1::2]
            df = _build_weights_df(
                n_obs,
                times,
                residuals_rad,
                clusters,
                weights_ra_arr,
                weights_dec_arr,
                "gaia_components",
                set_id,
                "set",
                observable_type=_observable_type_str(observation_set),
            )
            logger.debug(
                "GaiaComponentsWeight [%s]: random(AL=%.3g, AC=%.3g), "
                "systematic(AL=%.3g, AC=%.3g) [%s]; rewrote full transit matrix "
                "(RA w range=[%.2e, %.2e], DEC w range=[%.2e, %.2e])",
                set_id,
                self.random_scale_al,
                self.random_scale_ac,
                self.systematic_scale_al,
                self.systematic_scale_ac,
                "scan frame" if scan_angles is not None else "RA/Dec fallback",
                weights_ra_arr.min(),
                weights_ra_arr.max(),
                weights_dec_arr.min(),
                weights_dec_arr.max(),
            )
            return weights_array, df

        # Fallback: no cached components — use ID v2 residual-driven weighting.
        logger.warning(
            "GaiaComponentsWeight [%s]: no cached random/systematic covariance components found "
            "(set_id='%s', n_obs=%d). Falling back to ID v2 residual-driven weighting.",
            set_id,
            set_id,
            n_obs,
        )
        tf_clusters = clusters.by_level("timeframe")
        if not tf_clusters:
            ra_sigma = max(_rms(residuals_rad[:, 0]), min_sigma_rad)
            dec_sigma = max(_rms(residuals_rad[:, 1]), min_sigma_rad)
        else:
            ra_sigma, dec_sigma = _id_v2_sigmas(residuals_rad, tf_clusters, min_sigma_rad)
        w_ra = 1.0 / ra_sigma**2
        w_dec = 1.0 / dec_sigma**2
        weights_ra_arr = np.full(n_obs, w_ra)
        weights_dec_arr = np.full(n_obs, w_dec)
        weights_array = _interleave_weights(weights_ra_arr, weights_dec_arr)
        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
            weights_ra_arr,
            weights_dec_arr,
            "gaia_components",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        return weights_array, df


# ---------------------------------------------------------------------------
# Fixed along/across-scan weighting
# ---------------------------------------------------------------------------


@register_weight_strategy("fixed_scan")
class FixedScanWeight(WeightStrategy):
    r"""Fixed weights in the Gaia along-scan (AL) / across-scan (AC) frame.

    Assigns every observation the same 1σ uncertainty in the scan frame, with
    independent values for the along-scan and across-scan directions::

        sigma_AL = along_scan_sigma_mas  [mas]
        sigma_AC = across_scan_sigma_mas [mas]

    Because the scan frame is rotated relative to ``(RA, Dec)`` by the position
    angle of scan :math:`\psi`, a diagonal covariance in the scan frame becomes a
    **non-diagonal** 2x2 block in ``(RA, Dec)``::

        C_scan = diag(sigma_AL^2, sigma_AC^2)
        C_radec = T^T C_scan T,   T = [[sin(psi)cos(dec), cos(psi)],
                                       [cos(psi)cos(dec), -sin(psi)]]

    The weight block is ``W = C_radec^{-1}``.  This strategy therefore returns
    per-observation 2x2 blocks via :meth:`compute_block_weights` (applied with
    ``set_block_diagonal_weights``) rather than a diagonal weight vector.

    The scan angles are read from the standard frame through the
    :class:`~orbitdet.analysis.set_view.SetView` (the ``scan_angle`` column,
    populated on ingest from the Gaia archive); sets without a finite scan angle
    fall back to a diagonal ``(RA, Dec)`` weighting with the same sigmas.

    Parameters
    ----------
    along_scan_sigma_mas : float
        Fixed along-scan 1σ uncertainty in milliarcseconds.
    across_scan_sigma_mas : float
        Fixed across-scan 1σ uncertainty in milliarcseconds.
    """

    def __init__(
        self,
        along_scan_sigma_mas: float = 5.0,
        across_scan_sigma_mas: float = 200.0,
    ):
        if not np.isfinite(along_scan_sigma_mas) or along_scan_sigma_mas <= 0.0:
            raise ValueError(
                "FixedScanWeight: 'along_scan_sigma_mas' must be a positive finite number, "
                f"got {along_scan_sigma_mas!r}."
            )
        if not np.isfinite(across_scan_sigma_mas) or across_scan_sigma_mas <= 0.0:
            raise ValueError(
                "FixedScanWeight: 'across_scan_sigma_mas' must be a positive finite number, "
                f"got {across_scan_sigma_mas!r}."
            )
        self.along_scan_sigma_mas = float(along_scan_sigma_mas)
        self.across_scan_sigma_mas = float(across_scan_sigma_mas)

    def _sigmas_rad(self) -> tuple[float, float]:
        """Return the fixed ``(sigma_AL, sigma_AC)`` in radians."""
        mas_to_rad = _ARCSEC_TO_RAD / 1e3
        return (
            self.along_scan_sigma_mas * mas_to_rad,
            self.across_scan_sigma_mas * mas_to_rad,
        )

    def _scan_angles(self, view) -> np.ndarray | None:
        """Return the per-observation scan angles (radians), or None if unavailable.

        Reads the standard frame's ``scan_angle`` column through the set's
        :class:`~orbitdet.analysis.set_view.SetView`.  When no view is supplied
        (a strategy called directly, outside the engine) the angles are
        unavailable and the caller falls back to a diagonal weighting.
        """
        if view is None:
            return None
        return view.scan_angles

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Return the diagonal of the scan-frame weight blocks for reporting.

        The actual weighting is applied through :meth:`compute_block_weights`;
        this method only produces the metadata DataFrame and a diagonal weight
        vector (used for reporting and as a fallback).
        """
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        sigma_al, sigma_ac = self._sigmas_rad()
        psi = self._scan_angles(view)

        if psi is not None and len(psi) == n_obs:
            # Diagonal of the (RA, Dec) weight block, for reporting.
            observed = np.asarray(observation_set.concatenated_observations).reshape(-1, 2)
            cos_dec = np.cos(observed[:, 1])
            sin_psi = np.sin(psi)
            cos_psi = np.cos(psi)
            # W_radec = T^T diag(1/sigma_AL^2, 1/sigma_AC^2) T
            w_al = 1.0 / sigma_al**2
            w_ac = 1.0 / sigma_ac**2
            t_al_ra = sin_psi * cos_dec
            t_al_dec = cos_psi
            t_ac_ra = cos_psi * cos_dec
            t_ac_dec = -sin_psi
            w_ra = w_al * t_al_ra**2 + w_ac * t_ac_ra**2
            w_dec = w_al * t_al_dec**2 + w_ac * t_ac_dec**2
        else:
            # No scan angles: fall back to a diagonal (RA, Dec) weighting.
            w_ra = np.full(n_obs, 1.0 / sigma_al**2)
            w_dec = np.full(n_obs, 1.0 / sigma_ac**2)

        weights_array = _interleave_weights(w_ra, w_dec)
        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
            w_ra,
            w_dec,
            "fixed_scan",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "FixedScanWeight [%s]: sigma_AL=%.3g mas, sigma_AC=%.3g mas, scan angles %s",
            set_id,
            self.along_scan_sigma_mas,
            self.across_scan_sigma_mas,
            "available" if psi is not None else "MISSING (diagonal fallback)",
        )
        return weights_array, df

    def compute_block_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> list[np.ndarray] | None:
        """Return one 2x2 ``(RA, Dec)`` weight block per observation.

        Returns ``None`` when the scan angles are unavailable, so the engine
        falls back to the diagonal weights from :meth:`compute_weights`.
        """
        n_obs = len(observation_set.observation_times)
        if n_obs == 0:
            return None

        psi = self._scan_angles(view)
        if psi is None or len(psi) != n_obs:
            return None

        sigma_al, sigma_ac = self._sigmas_rad()
        w_al = 1.0 / sigma_al**2
        w_ac = 1.0 / sigma_ac**2

        observed = np.asarray(observation_set.concatenated_observations).reshape(-1, 2)
        cos_dec = np.cos(observed[:, 1])
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        # T maps (RA, Dec) -> (AL, AC); C_radec = T^T C_scan T, W = C_radec^{-1}.
        # Build W directly as T^T diag(w_AL, w_AC) T (2x2 per observation).
        t_al_ra = sin_psi * cos_dec
        t_al_dec = cos_psi
        t_ac_ra = cos_psi * cos_dec
        t_ac_dec = -sin_psi

        w_ra_ra = w_al * t_al_ra**2 + w_ac * t_ac_ra**2
        w_dec_dec = w_al * t_al_dec**2 + w_ac * t_ac_dec**2
        w_ra_dec = w_al * t_al_ra * t_al_dec + w_ac * t_ac_ra * t_ac_dec

        blocks = []
        for i in range(n_obs):
            blocks.append(
                np.array(
                    [
                        [w_ra_ra[i], w_ra_dec[i]],
                        [w_ra_dec[i], w_dec_dec[i]],
                    ]
                )
            )
        return blocks


# ---------------------------------------------------------------------------
# Fixed along/across-scan weighting with explicit random + systematic components
# ---------------------------------------------------------------------------

#: Epoch gap (seconds) above which a new Gaia transit starts.  Mirrors
#: ``orbitdet.visualization.gaia_transit_errors.DEFAULT_TRANSIT_GAP_SECONDS``:
#: intra-transit samples span tens of seconds, transits are >= ~106 min apart.
_FIXED_SCAN_COMPONENTS_TRANSIT_GAP_SECONDS = 1800.0


@register_weight_strategy("fixed_scan_components")
class FixedScanComponentsWeight(FixedScanWeight):
    r"""Fixed scan-frame weighting with random (block-diagonal) + systematic (rank-1) parts.

    A sibling of :class:`FixedScanWeight` for the case where the along/across-scan
    uncertainty is known as a **random** and a **systematic** component rather
    than a single sigma.  The two are *not* combined in quadrature into one
    per-observation sigma; they keep their distinct statistical structure, which
    is exactly what the Gaia archive does (see
    ``GaiaAstrometry.get_observation_covariance_components``):

    ``random`` (within transit)
        Independent from crossing to crossing, so it contributes a **block
        diagonal** scan-frame covariance: one ``2x2`` block per observation,
        ``diag(sigma_AL,r^2, sigma_AC,r^2)``.
    ``systematic`` (between transits)
        Constant across a whole transit, so it correlates *every* pair of
        observations within that transit.  For a transit with ``n`` observations
        it contributes a dense ``2n x 2n`` block built as a **Kronecker product**
        of an all-ones ``n x n`` matrix with the ``2x2`` systematic scan
        covariance::

            C_sys,transit = ones(n, n) ⊗ diag(sigma_AL,s^2, sigma_AC,s^2)

    The two are summed and inverted per transit to give the full (off-diagonal)
    ``(RA, Dec)`` weight matrix, rotated out of the scan frame.  Weighting each
    transit's observations jointly (rather than per observation) is the point:
    the systematic offset is degenerate across a transit, so its block carries
    off-diagonal terms the per-observation path cannot express.

    Because the result is a full transit matrix, this strategy sets
    :attr:`preserve_full_weight_matrix` to True and writes the set's full weight
    matrix (as :class:`GaiaComponentsWeight` does).  The ``random`` and
    ``systematic`` values are the ``random σ`` / ``systematic σ`` measured by
    :class:`orbitdet.visualization.gaia_transit_errors.GaiaTransitErrorComponents`.

    Configurable through ``weighting.fixed_scan_components`` (all in mas):

    ``random_along_scan_mas``
        Along-scan random 1σ (within-transit), in milliarcseconds.
    ``random_across_scan_mas``
        Across-scan random 1σ, in milliarcseconds.
    ``systematic_along_scan_mas``
        Along-scan systematic 1σ (transit-to-transit), in milliarcseconds.
    ``systematic_across_scan_mas``
        Across-scan systematic 1σ, in milliarcseconds.
    ``transit_gap_seconds``
        Epoch gap (seconds) above which a new transit starts (default ``1800``).
    """

    def __init__(
        self,
        random_along_scan_mas: float = 0.0,
        random_across_scan_mas: float = 0.0,
        systematic_along_scan_mas: float = 0.0,
        systematic_across_scan_mas: float = 0.0,
        transit_gap_seconds: float = _FIXED_SCAN_COMPONENTS_TRANSIT_GAP_SECONDS,
    ):
        for name, value in (
            ("random_along_scan_mas", random_along_scan_mas),
            ("random_across_scan_mas", random_across_scan_mas),
            ("systematic_along_scan_mas", systematic_along_scan_mas),
            ("systematic_across_scan_mas", systematic_across_scan_mas),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"FixedScanComponentsWeight: '{name}' must be a non-negative finite "
                    f"number, got {value!r}."
                )
        if float(random_along_scan_mas) <= 0.0 or float(random_across_scan_mas) <= 0.0:
            raise ValueError(
                "FixedScanComponentsWeight: the random along/across-scan sigma must be "
                "positive (it is the block-diagonal part that keeps the transit covariance "
                f"invertible); got random_AL={random_along_scan_mas!r}, "
                f"random_AC={random_across_scan_mas!r}."
            )
        if not np.isfinite(transit_gap_seconds) or transit_gap_seconds <= 0.0:
            raise ValueError(
                "FixedScanComponentsWeight: 'transit_gap_seconds' must be a positive finite "
                f"number, got {transit_gap_seconds!r}."
            )

        # The parent's per-observation sigma is only used for the reporting path
        # (compute_weights) and the diagonal fallback; the actual weighting is the
        # full transit matrix built in compute_block_weights.  Combine in
        # quadrature here purely for that reporting sigma.
        super().__init__(
            along_scan_sigma_mas=float(np.hypot(random_along_scan_mas, systematic_along_scan_mas)),
            across_scan_sigma_mas=float(
                np.hypot(random_across_scan_mas, systematic_across_scan_mas)
            ),
        )
        self.random_along_scan_mas = float(random_along_scan_mas)
        self.random_across_scan_mas = float(random_across_scan_mas)
        self.systematic_along_scan_mas = float(systematic_along_scan_mas)
        self.systematic_across_scan_mas = float(systematic_across_scan_mas)
        self.transit_gap_seconds = float(transit_gap_seconds)

    @property
    def preserve_full_weight_matrix(self) -> bool:
        """This strategy's weights *are* the full (per-transit) weight matrix."""
        return True

    def _component_sigmas_rad(self) -> tuple[float, float, float, float]:
        """Return ``(random_AL, random_AC, systematic_AL, systematic_AC)`` in radians."""
        mas_to_rad = _ARCSEC_TO_RAD / 1e3
        return (
            self.random_along_scan_mas * mas_to_rad,
            self.random_across_scan_mas * mas_to_rad,
            self.systematic_along_scan_mas * mas_to_rad,
            self.systematic_across_scan_mas * mas_to_rad,
        )

    @staticmethod
    def _transit_labels(epochs: np.ndarray, gap_seconds: float) -> np.ndarray:
        """Label sorted epochs with a transit index, splitting on epoch gaps."""
        if epochs.size == 0:
            return np.zeros(0, dtype=int)
        starts = np.concatenate([[False], np.diff(epochs) > gap_seconds])
        return np.cumsum(starts).astype(int)

    def _full_weight_matrix(
        self, observation_set: obs.SingleObservationSet, n_obs: int, view=None
    ) -> np.ndarray | None:
        """Build the full ``(2n, 2n)`` scan-frame weight matrix, or ``None``.

        ``None`` means the set cannot be handled as one transit matrix (no scan
        angles), so the caller should fall back to the per-observation path.
        """
        psi = self._scan_angles(view)
        if psi is None or len(psi) != n_obs:
            return None

        epochs = np.array([t.to_float() for t in observation_set.observation_times], dtype=float)
        order = np.argsort(epochs, kind="stable")
        sorted_epochs = epochs[order]
        transit_id = self._transit_labels(sorted_epochs, self.transit_gap_seconds)
        n_transits = int(transit_id[-1]) + 1

        observed = np.asarray(observation_set.concatenated_observations, dtype=float).reshape(-1, 2)
        dec_obs = observed[:, 1]

        sigma_al_r, sigma_ac_r, sigma_al_s, sigma_ac_s = self._component_sigmas_rad()
        random_cov_scan = np.diag([sigma_al_r**2, sigma_ac_r**2])
        systematic_cov_scan = np.diag([sigma_al_s**2, sigma_ac_s**2])

        blocks: list[np.ndarray] = []
        for j in range(n_transits):
            members = np.flatnonzero(transit_id == j)
            n = members.size
            # Rotation from (RA, Dec) to (AL, AC) for each member.
            #   east = ΔRA cos(dec), north = ΔDec
            #   AL = east sin psi + north cos psi
            #   AC = east cos psi - north sin psi
            t_blocks = np.empty((n, 2, 2))
            for k, idx in enumerate(members.tolist()):
                cos_dec = np.cos(dec_obs[idx])
                sin_psi = np.sin(psi[idx])
                cos_psi = np.cos(psi[idx])
                t_blocks[k] = np.array(
                    [
                        [sin_psi * cos_dec, cos_psi],
                        [cos_psi * cos_dec, -sin_psi],
                    ]
                )

            # Per-member random covariance in (RA, Dec): T_k^T C_rand_scan T_k.
            random_cov = np.zeros((2 * n, 2 * n))
            for k in range(n):
                rk = t_blocks[k].T @ random_cov_scan @ t_blocks[k]
                random_cov[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = rk

            # Systematic covariance in (RA, Dec), constant across the transit:
            # the systematic offset is one physical shift per transit, rotated by
            # each member's own scan angle, so its (RA, Dec) covariance is
            # C = mean_k(T_k^T C_sys_scan T_k).  The transit (Kronecker) structure
            # then tiles this 2x2 block with ones(n, n).
            c_sys_radec = np.zeros((2, 2))
            for k in range(n):
                c_sys_radec += t_blocks[k].T @ systematic_cov_scan @ t_blocks[k]
            c_sys_radec /= n
            systematic_cov = np.kron(np.ones((n, n)), c_sys_radec)

            covariance = random_cov + systematic_cov
            weight = np.linalg.inv(covariance)
            blocks.append((weight + weight.T) / 2.0)

        # Assemble block-diagonal over transits, then undo the epoch sort so the
        # matrix lines up with the set's original observation order.
        from scipy.linalg import block_diag

        weight_matrix = block_diag(*blocks) if len(blocks) > 1 else blocks[0]

        # ``order`` is the permutation that sorted the epochs; the matrix built
        # above is in sorted order, so permute its 2x2 rows/columns back.
        perm = np.argsort(order, kind="stable")  # inverse permutation
        idx = np.empty(2 * n_obs, dtype=int)
        idx[0::2] = 2 * perm
        idx[1::2] = 2 * perm + 1
        weight_matrix = weight_matrix[np.ix_(idx, idx)]
        return weight_matrix

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Write the full transit weight matrix and report its diagonal."""
        residuals_rad = _residuals_array(observation_set)
        n_obs = residuals_rad.shape[0]
        times = np.array([t.to_float() for t in observation_set.observation_times])

        if n_obs == 0:
            return np.array([], dtype=float), pd.DataFrame()

        weight_matrix = self._full_weight_matrix(observation_set, n_obs, view=view)
        if weight_matrix is None:
            # No scan angles: defer to FixedScanWeight's diagonal fallback (the
            # engine will not preserve the full matrix, since none was set).
            logger.warning(
                "FixedScanComponentsWeight [%s]: no scan angles; falling back to the "
                "diagonal (RA, Dec) weighting.",
                set_id,
            )
            return super().compute_weights(
                observation_set, clusters, set_id, min_sigma_arcsec, view=view
            )

        if getattr(observation_set, "has_full_weight_matrix_contribution", False):
            observation_set.set_full_weight_matrix(weight_matrix)

        weights_array = np.diag(weight_matrix).copy()
        weights_ra_arr = weights_array[0::2]
        weights_dec_arr = weights_array[1::2]
        df = _build_weights_df(
            n_obs,
            times,
            residuals_rad,
            clusters,
            weights_ra_arr,
            weights_dec_arr,
            "fixed_scan_components",
            set_id,
            "set",
            observable_type=_observable_type_str(observation_set),
        )
        logger.debug(
            "FixedScanComponentsWeight [%s]: random(AL=%.3g, AC=%.3g) mas, "
            "systematic(AL=%.3g, AC=%.3g) mas; wrote full %dx%d transit matrix",
            set_id,
            self.random_along_scan_mas,
            self.random_across_scan_mas,
            self.systematic_along_scan_mas,
            self.systematic_across_scan_mas,
            *weight_matrix.shape,
        )
        return weights_array, df


