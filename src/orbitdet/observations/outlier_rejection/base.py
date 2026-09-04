"""Abstract base class for outlier rejection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs


class OutlierStrategy(ABC):
    """Base class for all outlier rejection strategies.

    Each strategy implements a single rejection criterion and operates on one
    :class:`~tudatpy.estimation.observations.SingleObservationSet` at a time.
    Strategies are composed by the :class:`OutlierEngine` and applied in sequence
    to every observation set in a collection.

    Subclasses must implement :meth:`apply` and should be registered via the
    :func:`register_outlier_strategy` decorator.
    """

    @abstractmethod
    def apply(
        self,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.SingleObservationSet, dict]:
        """Apply the rejection strategy to a single observation set.

        Parameters
        ----------
        observation_set : SingleObservationSet
            The observation set to filter. Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (may be needed by some strategies for
            context, e.g. to compute residuals).

        Returns
        -------
        tuple[SingleObservationSet, dict]
            A tuple of:
            - The filtered observation set (a new object).
            - A metadata dictionary with at minimum:
                - ``'n_accepted'``: int — number of observations kept.
                - ``'n_rejected'``: int — number of observations removed.
                - ``'rejected_epochs'``: list[float] — epochs (s since J2000)
                  of rejected observations.
        """
        ...