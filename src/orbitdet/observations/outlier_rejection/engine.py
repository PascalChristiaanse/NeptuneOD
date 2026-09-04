"""Engine that orchestrates outlier rejection over an ObservationCollection.

The :class:`OutlierEngine` applies a sequence of :class:`OutlierStrategy`
instances to every observation set in a collection, returning a filtered
collection plus per-set rejection metadata.
"""

from __future__ import annotations

import logging
from typing import Any

import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf

from .base import OutlierStrategy
from .registry import get_strategy_class, list_registered_strategies

logger = logging.getLogger(__name__)


class OutlierEngine:
    """Orchestrates outlier rejection over an ObservationCollection.

    Parameters
    ----------
    strategies : list[OutlierStrategy]
        Ordered list of strategies to apply.  Each strategy is applied in
        sequence to every observation set in the collection.
    """

    def __init__(self, strategies: list[OutlierStrategy]):
        if not strategies:
            raise ValueError("At least one outlier strategy is required.")
        self._strategies = list(strategies)

    @property
    def strategies(self) -> list[OutlierStrategy]:
        """The ordered list of strategies managed by this engine (read-only)."""
        return list(self._strategies)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.ObservationCollection, dict[str, Any]]:
        """Apply all strategies in sequence to every observation set.

        Parameters
        ----------
        collection : ObservationCollection
            The observation collection to filter.  Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (passed to each strategy).

        Returns
        -------
        tuple[ObservationCollection, dict[str, Any]]
            A tuple of:
            - A new filtered ``ObservationCollection``.
            - A metadata dictionary keyed by set identifier (``set_id``),
              with per-strategy metadata as values.
        """
        filtered_collection, rejected_collection, summary = self.apply_with_rejected(
            collection, bodies
        )
        return filtered_collection, summary

    def apply_with_rejected(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
    ) -> tuple[obs.ObservationCollection, obs.ObservationCollection, dict[str, Any]]:
        """Apply all strategies and also return the rejected observations.

        Like :meth:`apply` but additionally returns a second
        ``ObservationCollection`` containing only the observations that were
        rejected by the final strategy in the chain.

        Parameters
        ----------
        collection : ObservationCollection
            The observation collection to filter.  Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (passed to each strategy).

        Returns
        -------
        tuple[ObservationCollection, ObservationCollection, dict[str, Any]]
            A tuple of:
            - Filtered (accepted) ``ObservationCollection``.
            - Rejected ``ObservationCollection`` (observations removed by any
              strategy in the chain).
            - Summary metadata dictionary.
        """
        all_sets = collection.get_single_observation_sets()
        logger.info(
            "OutlierEngine: applying %d strategy/ies to %d observation set(s)",
            len(self._strategies),
            len(all_sets),
        )

        filtered_sets: list[obs.SingleObservationSet] = []
        rejected_sets: list[obs.SingleObservationSet] = []
        per_set_metadata: dict[str, Any] = {}

        for obs_set in all_sets:
            set_id = _get_set_id(obs_set)
            current_set = obs_set
            set_metadata: dict[str, Any] = {"set_id": set_id, "strategies": []}

            for strategy in self._strategies:
                strategy_name = strategy.__class__.__name__
                logger.debug(
                    "Applying strategy '%s' to set '%s'", strategy_name, set_id
                )
                current_set, strategy_meta = strategy.apply(current_set, bodies)
                set_metadata["strategies"].append(strategy_meta)

            # Aggregate totals for this set
            n_accepted = set_metadata["strategies"][-1]["n_accepted"]
            n_rejected = sum(
                s["n_rejected"] for s in set_metadata["strategies"]
            )
            n_total = set_metadata["strategies"][0]["n_total"]
            set_metadata["n_accepted"] = n_accepted
            set_metadata["n_rejected"] = n_rejected
            set_metadata["n_total"] = n_total

            filtered_sets.append(current_set)

            # Build rejected set: observations present in the original set
            # but absent from the accepted (filtered) set.  This is more
            # reliable than trying to invert the filter logic.
            rejected_set = _build_rejected_set(obs_set, current_set)
            rejected_sets.append(rejected_set)

            per_set_metadata[set_id] = set_metadata

            logger.debug(
                "Set '%s': %d accepted, %d rejected out of %d",
                set_id,
                n_accepted,
                n_rejected,
                n_total,
            )

        # Build collections
        filtered_collection = obs.ObservationCollection(filtered_sets)
        rejected_collection = obs.ObservationCollection(rejected_sets)

        # Aggregate summary
        total_accepted = sum(m["n_accepted"] for m in per_set_metadata.values())
        total_rejected = sum(m["n_rejected"] for m in per_set_metadata.values())
        total_obs = sum(m["n_total"] for m in per_set_metadata.values())
        summary = {
            "n_sets": len(filtered_sets),
            "n_total_observations": total_obs,
            "n_accepted": total_accepted,
            "n_rejected": total_rejected,
            "per_set": per_set_metadata,
        }

        logger.info(
            "OutlierEngine: %d accepted, %d rejected out of %d observations across %d set(s)",
            total_accepted,
            total_rejected,
            total_obs,
            len(filtered_sets),
        )
        return filtered_collection, rejected_collection, summary

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> OutlierEngine:
        """Build an engine from a Hydra configuration node.

        The config should have the following structure::

            outlier_rejection:
              enabled: true
              strategies:
                - type: residual_threshold
                  threshold_arcsec: 1.5
                - type: epoch_filter
                  epochs: [...]

        Parameters
        ----------
        cfg : DictConfig
            The configuration node (typically ``cfg.outlier_rejection``).

        Returns
        -------
        OutlierEngine
            An engine with the configured strategies.

        Raises
        ------
        ValueError
            If the config is invalid or a strategy type is not recognised.
        """
        strategies_cfg = OmegaConf.select(cfg, "strategies")
        if not strategies_cfg:
            raise ValueError(
                "Outlier rejection config must have a non-empty 'strategies' list."
            )

        strategies: list[OutlierStrategy] = []
        for entry in strategies_cfg:
            strategy_type = OmegaConf.select(entry, "type")
            if not strategy_type:
                raise ValueError(
                    "Each outlier strategy entry must have a 'type' field."
                )

            cls_strategy = get_strategy_class(strategy_type)

            # Build kwargs from the config entry (excluding 'type')
            kwargs = {}
            for key, value in entry.items():
                if key != "type":
                    kwargs[key] = value

            # Special handling: convert OmegaConf list to plain Python list
            # for epoch_filter's epochs_to_remove parameter.
            if strategy_type == "epoch_filter" and "epochs" in kwargs:
                kwargs["epochs_to_remove"] = list(kwargs.pop("epochs"))

            strategy_instance = cls_strategy(**kwargs)
            strategies.append(strategy_instance)

        return cls(strategies)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_set_id(observation_set: obs.SingleObservationSet) -> str:
    """Extract a human-readable identifier for an observation set.

    Tudat's ``SingleObservationSet`` does not expose a ``set_id`` attribute
    directly in the Python bindings.  We fall back to a string representation
    of the link definition.
    """
    try:
        return str(observation_set)
    except Exception:
        return f"set_{id(observation_set)}"


def _build_rejected_set(
    original_set: obs.SingleObservationSet,
    accepted_set: obs.SingleObservationSet,
) -> obs.SingleObservationSet:
    """Build a SingleObservationSet containing all rejected observations.

    Computes the set difference between the original and accepted (filtered)
    observation sets by comparing epochs.  The rejected epochs are then
    extracted from the original set using Tudat's epoch filter with inverted
    logic.

    Parameters
    ----------
    original_set : SingleObservationSet
        The original unfiltered observation set.
    accepted_set : SingleObservationSet
        The filtered (accepted) observation set.

    Returns
    -------
    SingleObservationSet
        A set containing only the rejected observations.
    """
    from tudatpy.estimation.observations import observations_processing as obs_proc

    # Collect epochs from both sets
    original_epochs = {epoch.to_float() for epoch in original_set.observation_times}
    accepted_epochs = {epoch.to_float() for epoch in accepted_set.observation_times}

    rejected_epochs = sorted(original_epochs - accepted_epochs)

    if not rejected_epochs:
        # Return an empty set: remove everything by using opposite condition
        # with a dummy epoch that won't match any observation.
        empty_filter = obs_proc.observation_filter(
            obs_proc.ObservationFilterType.epochs_filtering,
            [0.0],
            filter_out=True,
            use_opposite_condition=True,
        )
        return obs.create_filtered_observation_set(original_set, empty_filter)

    # Tudat requires filter_out=True, so we use use_opposite_condition=True
    # to remove everything EXCEPT the rejected epochs.
    keep_filter = obs_proc.observation_filter(
        obs_proc.ObservationFilterType.epochs_filtering,
        rejected_epochs,
        filter_out=True,
        use_opposite_condition=True,
    )
    return obs.create_filtered_observation_set(original_set, keep_filter)