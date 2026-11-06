"""Concrete outlier rejection strategies.

Each strategy is a class inheriting from :class:`OutlierStrategy` and registered
via the :func:`register_outlier_strategy` decorator.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from tudatpy.estimation.observations import observations_processing as obs_proc

from .base import OutlierStrategy
from .registry import register_outlier_strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

#: Default tolerance (seconds) for matching a configured epoch to an observation.
#:
#: Epochs are written into configs as decimal literals, and a value copied from a
#: log or a report is rarely bit-identical to the float Tudat holds: the two can
#: differ in the last digit or two, which is microseconds.  Tudat's
#: ``epochs_filtering`` matches exactly, so without a tolerance a configured epoch
#: silently removes nothing and the observation survives into the propagation.
DEFAULT_EPOCH_TOLERANCE_SECONDS = 1e-3


# ---------------------------------------------------------------------------
# Residual threshold outlier rejection
# ---------------------------------------------------------------------------


@register_outlier_strategy("residual_threshold")
class ResidualThresholdOutlier(OutlierStrategy):
    """Remove observations whose RA or DEC residual exceeds a fixed threshold.

    Uses Tudat's built-in ``residual_filtering`` observation filter.  Residuals
    must already have been computed on the observation set (via
    :func:`~tudatpy.estimation.observations.compute_residuals_and_dependent_variables`)
    before this strategy is applied.

    Parameters
    ----------
    threshold_arcsec : float
        Maximum absolute residual in arcseconds.  Observations with either
        RA *or* DEC residual exceeding this value are removed.
        Default: 1.5 arcsec.
    """

    def __init__(self, threshold_arcsec: float = 1.5):
        if threshold_arcsec <= 0:
            raise ValueError(f"threshold_arcsec must be positive, got {threshold_arcsec}")
        self._threshold_rad = threshold_arcsec * _ARCSEC_TO_RAD
        logger.debug(
            "ResidualThresholdOutlier: threshold = %.2f arcsec (%.2e rad)",
            threshold_arcsec,
            self._threshold_rad,
        )

    def apply(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
        view=None,
    ) -> tuple[obs.SingleObservationSet, dict]:
        # Build the "keep" filter: observations with residual <= threshold are kept.
        keep_filter = obs_proc.observation_filter(
            obs_proc.ObservationFilterType.residual_filtering,
            self._threshold_rad,
            filter_out=True,
            use_opposite_condition=False,
        )
        filtered_set = obs.create_filtered_observation_set(observation_set, keep_filter)

        # Build the opposite filter to identify rejected observations (for metadata).
        reject_filter = obs_proc.observation_filter(
            obs_proc.ObservationFilterType.residual_filtering,
            self._threshold_rad,
            filter_out=True,
            use_opposite_condition=True,
        )
        rejected_set = obs.create_filtered_observation_set(observation_set, reject_filter)

        # Collect metadata
        epochs_all = _get_epochs_float(observation_set)
        epochs_kept = _get_epochs_float(filtered_set)
        epochs_rejected = _get_epochs_float(rejected_set)

        if len(epochs_kept) + len(epochs_rejected) != len(epochs_all):
            raise ValueError(
                "ResidualThresholdOutlier: accepted + rejected epochs do not match total epochs"
            )

        metadata: dict[str, Any] = {
            "strategy": "residual_threshold",
            "threshold_arcsec": self._threshold_rad / _ARCSEC_TO_RAD,
            "n_accepted": len(epochs_kept),
            "n_rejected": len(epochs_rejected),
            "n_total": len(epochs_all),
            "rejected_epochs": epochs_rejected,
        }

        logger.debug(
            "ResidualThresholdOutlier: %d accepted, %d rejected out of %d",
            metadata["n_accepted"],
            metadata["n_rejected"],
            metadata["n_total"],
        )
        return filtered_set, metadata


# ---------------------------------------------------------------------------
# Epoch-based outlier rejection
# ---------------------------------------------------------------------------


@register_outlier_strategy("epoch_filter")
class EpochFilterOutlier(OutlierStrategy):
    """Remove observations whose epochs match a given list.

    Uses :meth:`~orbitdet.analysis.set_view.SetView.epochs` to resolve each
    configured epoch to the observation it names, then removes those
    observations.  Matching is by tolerance rather than by exact float equality:
    a configured epoch is a decimal literal, and the value Tudat holds can differ
    in the last digit or two (microseconds).  The actual error is logged, so a
    tolerance that is doing real work is visible rather than silent.

    Parameters
    ----------
    epochs_to_remove : list[float]
        Epochs (in seconds since J2000, TDB) to remove from the observation
        set.
    tolerance_seconds : float
        How close a configured epoch must be to an observation's epoch to match
        it.  Defaults to 1 ms, which is far below any real observation cadence
        but comfortably above the float noise in a decimal literal.
    """

    def __init__(
        self,
        epochs_to_remove: list[float] | None = None,
        tolerance_seconds: float = DEFAULT_EPOCH_TOLERANCE_SECONDS,
    ):
        if tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds must be non-negative, got {tolerance_seconds}"
            )
        self._epochs_to_remove = list(epochs_to_remove or [])
        self._tolerance_seconds = float(tolerance_seconds)
        logger.debug(
            "EpochFilterOutlier: configured to remove %d epoch(s) (tolerance %.3g s)",
            len(self._epochs_to_remove),
            self._tolerance_seconds,
        )

    def apply(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
        view=None,
    ) -> tuple[obs.SingleObservationSet, dict]:
        epochs_all = _get_epochs_float(observation_set)

        if not self._epochs_to_remove:
            # Nothing to remove — return the set unchanged.
            metadata: dict[str, Any] = {
                "strategy": "epoch_filter",
                "n_accepted": len(epochs_all),
                "n_rejected": 0,
                "n_total": len(epochs_all),
                "rejected_epochs": [],
            }
            return observation_set, metadata

        # Resolve each configured epoch to the observation it names.  Matching
        # here rather than in Tudat's exact-match filter is what makes a
        # decimal literal in a config actually remove the observation it means.
        matched, unmatched = self._resolve_epochs(epochs_all)

        if unmatched:
            logger.warning(
                "EpochFilterOutlier: %d configured epoch(s) matched no observation within "
                "%.3g s and will remove nothing: %s.  The nearest observation is %s s away.",
                len(unmatched),
                self._tolerance_seconds,
                [f"{epoch!r}" for epoch in unmatched],
                [f"{gap:.6g}" for gap in self._nearest_gaps(epochs_all, unmatched)],
            )

        if not matched:
            metadata = {
                "strategy": "epoch_filter",
                "n_accepted": len(epochs_all),
                "n_rejected": 0,
                "n_total": len(epochs_all),
                "rejected_epochs": [],
            }
            return observation_set, metadata

        epoch_filter = obs_proc.observation_filter(
            obs_proc.ObservationFilterType.epochs_filtering,
            matched,
            filter_out=True,
            use_opposite_condition=False,
        )
        filtered_set = obs.create_filtered_observation_set(observation_set, epoch_filter)

        epochs_kept = _get_epochs_float(filtered_set)
        epochs_rejected = [t for t in epochs_all if t not in epochs_kept]

        if len(epochs_kept) + len(epochs_rejected) != len(epochs_all):
            raise ValueError(
                "EpochFilterOutlier: accepted + rejected epochs do not match total epochs"
            )

        metadata = {
            "strategy": "epoch_filter",
            "n_accepted": len(epochs_kept),
            "n_rejected": len(epochs_rejected),
            "n_total": len(epochs_all),
            "rejected_epochs": epochs_rejected,
        }

        logger.debug(
            "EpochFilterOutlier: %d accepted, %d rejected out of %d",
            metadata["n_accepted"],
            metadata["n_rejected"],
            metadata["n_total"],
        )
        return filtered_set, metadata

    def _resolve_epochs(
        self, epochs_all: list[float]
    ) -> tuple[list[float], list[float]]:
        """Map configured epochs onto the set's actual epochs.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(matched, unmatched)``.  ``matched`` holds the *observation's* own
            epoch values, not the configured ones, so the downstream exact-match
            filter is guaranteed to hit.  ``unmatched`` holds the configured
            values that found nothing within tolerance.
        """
        if not epochs_all:
            return [], list(self._epochs_to_remove)

        available = np.asarray(epochs_all, dtype="float64")
        matched: list[float] = []
        unmatched: list[float] = []
        for configured in self._epochs_to_remove:
            gaps = np.abs(available - float(configured))
            nearest = int(np.argmin(gaps))
            if gaps[nearest] <= self._tolerance_seconds:
                actual = float(available[nearest])
                if gaps[nearest] > 0.0:
                    # The tolerance is doing real work; say so, because a
                    # configured epoch that is merely close is a config smell.
                    logger.warning(
                        "EpochFilterOutlier: configured epoch %.9f matched observation "
                        "%.9f, a difference of %.3g s (within the %.3g s tolerance).",
                        float(configured),
                        actual,
                        float(gaps[nearest]),
                        self._tolerance_seconds,
                    )
                matched.append(actual)
            else:
                unmatched.append(float(configured))
        return matched, unmatched

    @staticmethod
    def _nearest_gaps(epochs_all: list[float], configured: list[float]) -> list[float]:
        """Distance from each unmatched configured epoch to the nearest observation."""
        if not epochs_all:
            return [float("inf")] * len(configured)
        available = np.asarray(epochs_all, dtype="float64")
        return [float(np.min(np.abs(available - float(epoch)))) for epoch in configured]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_epochs_float(
    observation_set: obs.SingleObservationSet,
) -> list[float]:
    """Extract observation epochs as a list of floats (seconds since J2000)."""
    return [epoch.to_float() for epoch in observation_set.observation_times]
