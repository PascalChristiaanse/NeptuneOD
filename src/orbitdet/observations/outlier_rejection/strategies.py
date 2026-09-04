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

    Uses Tudat's ``epochs_filtering`` observation filter.  This is useful for
    manually removing specific bad observations identified by inspection.

    Parameters
    ----------
    epochs_to_remove : list[float]
        Epochs (in seconds since J2000, TDB) to remove from the observation
        set.  Only exact matches are removed (within floating-point tolerance).
    """

    def __init__(self, epochs_to_remove: list[float] | None = None):
        self._epochs_to_remove = list(epochs_to_remove or [])
        logger.debug(
            "EpochFilterOutlier: configured to remove %d epoch(s)", len(self._epochs_to_remove)
        )

    def apply(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.SingleObservationSet, dict]:
        if not self._epochs_to_remove:
            # Nothing to remove — return the set unchanged.
            epochs_all = _get_epochs_float(observation_set)
            metadata: dict[str, Any] = {
                "strategy": "epoch_filter",
                "n_accepted": len(epochs_all),
                "n_rejected": 0,
                "n_total": len(epochs_all),
                "rejected_epochs": [],
            }
            return observation_set, metadata

        epoch_filter = obs_proc.observation_filter(
            obs_proc.ObservationFilterType.epochs_filtering,
            self._epochs_to_remove,
            filter_out=True,
            use_opposite_condition=False,
        )
        filtered_set = obs.create_filtered_observation_set(observation_set, epoch_filter)

        epochs_all = _get_epochs_float(observation_set)
        epochs_kept = _get_epochs_float(filtered_set)
        epochs_rejected = [t for t in epochs_all if t not in epochs_kept]

        metadata: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_epochs_float(
    observation_set: obs.SingleObservationSet,
) -> list[float]:
    """Extract observation epochs as a list of floats (seconds since J2000)."""
    return [epoch.to_float() for epoch in observation_set.observation_times]