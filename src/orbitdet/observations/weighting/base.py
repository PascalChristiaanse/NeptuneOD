"""Abstract base class for weighting strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from .grouping import GroupList


class WeightStrategy(ABC):
    """Base class for all weighting strategies.

    Each strategy computes per-observation inverse-variance weights from
    residuals.  Strategies receive a
    :class:`~tudatpy.estimation.observations.SingleObservationSet` together
    with its pre-computed :class:`GroupList` and return both the interleaved
    weight array and a metadata DataFrame.

    Subclasses must implement :meth:`compute_weights` and should be registered
    via the :func:`register_weight_strategy` decorator.
    """

    @abstractmethod
    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        groups: GroupList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute per-observation weights from residuals.

        Parameters
        ----------
        observation_set : SingleObservationSet
            The observation set to weight.  Residuals must be populated.
        groups : GroupList
            Groups derived from the observation set (timeframes, sections, etc.).
        set_id : str
            Identifier for this observation set (e.g. observatory code).
        min_sigma_arcsec : float
            Floor on the RMSE in arcseconds — prevents a single near-perfect
            group from receiving an astronomically large weight.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame]
            - weights_array : shape ``(2 * n_obs,)`` with RA at even indices
              and DEC at odd indices (ready for ``set_tabulated_weights``).
            - weights_df : DataFrame with per-observation weight metadata
              (columns include ``group_id``, ``level``, ``ra_rmse``,
              ``dec_rmse``, ``weight_ra``, ``weight_dec``).
        """
        ...