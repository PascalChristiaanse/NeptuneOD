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
        view=None,
    ) -> tuple[obs.SingleObservationSet, dict]:
        """Apply the rejection strategy to a single observation set.

        Parameters
        ----------
        observation_set : SingleObservationSet
            The observation set to filter. Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (may be needed by some strategies for
            context, e.g. to compute residuals).
        view : SetView | None
            The set's window onto the access layer, carrying the standard-frame
            columns (``observatory_code``, ``data_source``, ``scan_angle``, ...).
            ``None`` when a strategy is called directly rather than through the
            engine; strategies that need frame columns must then fall back to a
            frame-free path.

            Note that the view describes the set **as it entered the engine**.
            Strategies filter observations, so after the first strategy the
            view's rows no longer line up with ``observation_set``; read it for
            per-observation *attributes*, not for index alignment.

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
