"""Engine that orchestrates outlier rejection over an ObservationCollection.

The :class:`OutlierEngine` applies a sequence of :class:`OutlierStrategy`
instances to every observation set in a collection, returning a filtered
collection plus per-set rejection metadata.

Each strategy can optionally be restricted to specific observation sets
via a ``sets`` filter.  Strategies without a ``sets`` filter are applied
to **all** sets.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf

from ..grouping import build_group_list
from ..utils import get_set_identifier
from .base import OutlierStrategy
from .registry import get_strategy_class

logger = logging.getLogger(__name__)


class _ScopedStrategy:
    """A strategy paired with an optional set and/or group filter.

    Parameters
    ----------
    strategy : OutlierStrategy
        The strategy instance.
    set_filter : set[str] | None
        If not None, the strategy is only applied to observation sets whose
        :func:`get_set_identifier` is in this set.  If None, the strategy
        applies to all sets.
    group_filter : set[str] | None
        If not None, the strategy is only applied to observations that fall in
        one of the named groups (matched on the group's short ``name``, i.e.
        the section name or timeframe id).  Groups come from the outlier
        subsystem's **own** ``grouping`` config, which is independent of the
        weighting grouping.
    """

    def __init__(
        self,
        strategy: OutlierStrategy,
        set_filter: set[str] | None,
        group_filter: set[str] | None = None,
    ):
        self.strategy = strategy
        self.set_filter = set_filter
        self.group_filter = group_filter

    def applies_to(self, set_id: str) -> bool:
        """Check whether this strategy applies to the given set identifier."""
        return self.set_filter is None or set_id in self.set_filter

    def applies_to_group(self, group_name: str) -> bool:
        """Check whether this strategy applies to the given group name."""
        return self.group_filter is None or group_name in self.group_filter


class OutlierEngine:
    """Orchestrates outlier rejection over an ObservationCollection.

    Parameters
    ----------
    strategies : list[_ScopedStrategy]
        Ordered list of scoped strategies to apply.  Each strategy is applied
        in sequence to every observation set in the collection.
    grouping : DictConfig | None
        The outlier subsystem's **own** grouping configuration.  When present,
        strategies with a ``groups`` filter are applied per-group (matching the
        group's short name), using the same partitioning mechanisms the
        weighting subsystem uses but with a config that is independent of it.
        When None, only the whole-set ``sets`` scoping applies.
    """

    def __init__(
        self,
        strategies: list[_ScopedStrategy],
        grouping: DictConfig | None = None,
    ):
        if not strategies:
            raise ValueError("At least one outlier strategy is required.")
        self._strategies = list(strategies)
        self._grouping = grouping

    @classmethod
    def from_strategies(cls, strategies: list[OutlierStrategy]) -> OutlierEngine:
        """Build an engine from a list of plain strategies (no set filters).

        Parameters
        ----------
        strategies : list[OutlierStrategy]
            Ordered list of strategies.  Each applies to all observation sets.

        Returns
        -------
        OutlierEngine
        """
        return cls([_ScopedStrategy(s, None) for s in strategies])

    @property
    def strategies(self) -> list[OutlierStrategy]:
        """The ordered list of strategies managed by this engine (read-only)."""
        return [s.strategy for s in self._strategies]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
        dataset_metadata: dict[str, list[dict]] | None = None,
    ) -> tuple[obs.ObservationCollection, dict[str, Any]]:
        """Apply all strategies in sequence to every observation set.

        Parameters
        ----------
        collection : ObservationCollection
            The observation collection to filter.  Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (passed to each strategy).
        dataset_metadata : dict[str, list[dict]], optional
            Mapping from observatory-code set_id to a list of metadata dicts
            (one per dataset config that produced observations for that
            observatory).  Each dict has a ``dataset_id`` key (e.g. ``nm0083``).
            When provided, the per-set report uses ``{dataset_id}_{obs_code}``
            as the set identifier, giving unique and meaningful keys.

        Returns
        -------
        tuple[ObservationCollection, dict[str, Any]]
            A tuple of:
            - A new filtered ``ObservationCollection``.
            - A metadata dictionary keyed by set identifier (``set_id``),
              with per-strategy metadata as values.
        """
        filtered_collection, rejected_collection, summary = self.apply_with_rejected(
            collection, bodies, dataset_metadata=dataset_metadata
        )
        return filtered_collection, summary

    def apply_with_rejected(
        self,
        collection: obs.ObservationCollection,
        bodies: env.SystemOfBodies,
        dataset_metadata: dict[str, list[dict]] | None = None,
    ) -> tuple[obs.ObservationCollection, obs.ObservationCollection, dict[str, Any]]:
        """Apply all strategies and also return the rejected observations.

        Like :meth:`apply` but additionally returns a second
        ``ObservationCollection`` containing only the observations that were
        rejected by any strategy in the chain.

        Parameters
        ----------
        collection : ObservationCollection
            The observation collection to filter.  Not modified in place.
        bodies : SystemOfBodies
            The system of bodies (passed to each strategy).
        dataset_metadata : dict[str, list[dict]], optional
            Mapping from observatory-code set_id to a list of metadata dicts
            (one per dataset config that produced observations for that
            observatory).  Each dict has a ``dataset_id`` key (e.g. ``nm0083``).
            When provided, the per-set report uses ``{dataset_id}_{obs_code}``
            as the set identifier, giving unique and meaningful keys.

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

        # Build a FIFO queue per observatory-code set_id from dataset_metadata
        meta_queues: dict[str, list[dict]] = {}
        if dataset_metadata:
            meta_queues = {k: list(v) for k, v in dataset_metadata.items()}

        filtered_sets: list[obs.SingleObservationSet] = []
        rejected_sets: list[obs.SingleObservationSet] = []
        per_set_metadata: list[dict[str, Any]] = []

        for obs_set in all_sets:
            obs_code = get_set_identifier(obs_set)
            current_set = obs_set

            # Resolve a unique, meaningful set_id using dataset_metadata
            meta_queue = meta_queues.get(obs_code, [])
            if meta_queue:
                meta = meta_queue.pop(0)
                dataset_id = meta.get("dataset_id", obs_code)
                set_id = f"{dataset_id}_{obs_code}"
            else:
                set_id = obs_code

            set_metadata: dict[str, Any] = {"set_id": set_id, "strategies": []}

            # Build this subsystem's own grouping for this set (if configured).
            groups = None
            if self._grouping is not None:
                times = np.array([t.to_float() for t in obs_set.observation_times])
                groups = build_group_list(times, set_id, self._grouping)

            for scoped in self._strategies:
                # The sets filter in configs uses observatory codes (e.g. "689", "Voyager 2"),
                # so we check against the raw observatory code, not the composite key.
                if not scoped.applies_to(obs_code):
                    logger.debug(
                        "Skipping strategy '%s' for set '%s' (not in filter)",
                        scoped.strategy.__class__.__name__,
                        set_id,
                    )
                    continue

                strategy_name = scoped.strategy.__class__.__name__
                if groups is not None and scoped.group_filter is not None:
                    logger.debug(
                        "Applying strategy '%s' to set '%s' per-group %s",
                        strategy_name,
                        set_id,
                        scoped.group_filter,
                    )
                    current_set, strategy_meta = self._apply_group_scoped(
                        scoped, current_set, bodies, groups
                    )
                else:
                    logger.debug("Applying strategy '%s' to set '%s'", strategy_name, set_id)
                    current_set, strategy_meta = scoped.strategy.apply(current_set, bodies)

                set_metadata["strategies"].append(strategy_meta)

            # Aggregate totals for this set
            if set_metadata["strategies"]:
                n_accepted = set_metadata["strategies"][-1]["n_accepted"]
                n_rejected = sum(s["n_rejected"] for s in set_metadata["strategies"])
                n_total = set_metadata["strategies"][0]["n_total"]
            else:
                # No strategies applied to this set — keep all observations
                epochs_all = _get_epochs_float(obs_set)
                n_accepted = len(epochs_all)
                n_rejected = 0
                n_total = len(epochs_all)

            set_metadata["n_accepted"] = n_accepted
            set_metadata["n_rejected"] = n_rejected
            set_metadata["n_total"] = n_total

            filtered_sets.append(current_set)

            # Build rejected set: observations present in the original set
            # but absent from the accepted (filtered) set.
            rejected_set = _build_rejected_set(obs_set, current_set)
            rejected_sets.append(rejected_set)

            per_set_metadata.append(set_metadata)

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
        total_accepted = sum(m["n_accepted"] for m in per_set_metadata)
        total_rejected = sum(m["n_rejected"] for m in per_set_metadata)
        total_obs = sum(m["n_total"] for m in per_set_metadata)
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
    # Group-scoped application
    # ------------------------------------------------------------------

    def _apply_group_scoped(
        self,
        scoped: _ScopedStrategy,
        observation_set: obs.SingleObservationSet,
        bodies: env.SystemOfBodies,
        groups,
    ) -> tuple[obs.SingleObservationSet, dict[str, Any]]:
        """Apply a strategy to only the observations in matching groups.

        Scoping uses each group's time window (via Tudat's
        ``time_bounds_filtering``) rather than set indices, so it remains valid
        across repeated applications even as observations are removed.
        """
        # Resolve matching groups that have a usable time window.
        windows: list[tuple[float, float]] = []
        n_group = 0
        for g in groups:
            if not scoped.applies_to_group(g.name):
                continue
            t_start = g.metadata.get("t_start")
            t_end = g.metadata.get("t_end")
            if t_start is None or t_end is None:
                logger.warning(
                    "Group '%s' has no time bounds; cannot scope by group.", g.name
                )
                continue
            windows.append((float(t_start), float(t_end)))
            n_group += g.size

        n_total = len(_get_epochs_float(observation_set))
        if not windows:
            return observation_set, {
                "strategy": scoped.strategy.__class__.__name__,
                "n_accepted": n_total,
                "n_rejected": 0,
                "n_total": n_total,
                "rejected_epochs": [],
                "group_scoped": True,
            }

        current_set = observation_set
        rejected_epochs: list[float] = []
        for t_start, t_end in windows:
            epochs_before = _get_epochs_float(current_set)
            subset = obs.create_filtered_observation_set(
                current_set, _time_bounds_filter(t_start, t_end, filter_out=False)
            )
            kept_subset, _meta = scoped.strategy.apply(subset, bodies)

            kept_epochs = {float(t.to_float()) for t in kept_subset.observation_times}
            in_window = {float(e) for e in epochs_before if t_start <= e <= t_end}
            rejected = in_window - kept_epochs
            if not rejected:
                continue
            rejected_epochs.extend(sorted(rejected))
            current_set = obs.create_filtered_observation_set(
                current_set, _epochs_filter(sorted(rejected), filter_out=True)
            )

        n_now = len(_get_epochs_float(current_set))
        meta: dict[str, Any] = {
            "strategy": scoped.strategy.__class__.__name__,
            "group_scoped": True,
            "n_group": n_group,
            "n_accepted": n_now,
            "n_rejected": n_total - n_now,
            "n_total": n_total,
            "rejected_epochs": sorted(rejected_epochs),
        }
        return current_set, meta

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg: DictConfig, default_grouping: DictConfig | None = None
    ) -> OutlierEngine:
        """Build an engine from a Hydra configuration node.

        The config should have the following structure::

            outlier_rejection:
              enabled: true
              grouping:                 # optional, independent of weighting
                levels:
                  - type: section
                    sections: [...]
              strategies:
                - type: residual_threshold
                  threshold_arcsec: 1.5
                  sets: ["689"]         # optional set filter
                  groups: ["night_1"]   # optional group-name filter
                - type: epoch_filter
                  epochs: [...]

        Parameters
        ----------
        cfg : DictConfig
            The configuration node (typically ``cfg.outlier_rejection``).

        Returns
        -------
        OutlierEngine
            An engine with the configured strategies and its own grouping.

        Raises
        ------
        ValueError
            If the config is invalid or a strategy type is not recognised.
        """
        strategies_cfg = OmegaConf.select(cfg, "strategies")
        if not strategies_cfg:
            raise ValueError("Outlier rejection config must have a non-empty 'strategies' list.")

        grouping = OmegaConf.select(cfg, "grouping") or default_grouping

        scoped_strategies: list[_ScopedStrategy] = []
        for entry in strategies_cfg:
            strategy_type = OmegaConf.select(entry, "type")
            if not strategy_type:
                raise ValueError("Each outlier strategy entry must have a 'type' field.")

            cls_strategy = get_strategy_class(strategy_type)

            # Build kwargs from the config entry (excluding the filter keys)
            kwargs = {}
            set_filter: set[str] | None = None
            group_filter: set[str] | None = None
            for key, value in entry.items():
                if key == "type":
                    continue
                if key == "sets":
                    # Convert to a set of strings for fast lookup
                    set_filter = {str(s) for s in value}
                    continue
                if key == "groups":
                    group_filter = {str(s) for s in value}
                    continue
                kwargs[key] = value

            # Special handling: convert OmegaConf list to plain Python list
            # for epoch_filter's epochs_to_remove parameter.
            if strategy_type == "epoch_filter" and "epochs" in kwargs:
                kwargs["epochs_to_remove"] = list(kwargs.pop("epochs"))

            strategy_instance = cls_strategy(**kwargs)
            scoped_strategies.append(
                _ScopedStrategy(strategy_instance, set_filter, group_filter)
            )

        return cls(scoped_strategies, grouping=grouping)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_epochs_float(
    observation_set: obs.SingleObservationSet,
) -> list[float]:
    """Extract observation epochs as a list of floats (seconds since J2000)."""
    return [epoch.to_float() for epoch in observation_set.observation_times]


def _time_bounds_filter(t_start: float, t_end: float, filter_out: bool):
    """Build a Tudat time-bounds observation filter.

    ``filter_out=False`` keeps observations within ``[t_start, t_end]``;
    ``filter_out=True`` removes them.
    """
    from tudatpy.estimation.observations import observations_processing as obs_proc

    return obs_proc.observation_filter(
        obs_proc.ObservationFilterType.time_bounds_filtering,
        t_start,
        t_end,
        filter_out=filter_out,
        use_opposite_condition=False,
    )


def _epochs_filter(epochs: list[float], filter_out: bool):
    """Build a Tudat epochs observation filter (exact epoch matching)."""
    from tudatpy.estimation.observations import observations_processing as obs_proc

    return obs_proc.observation_filter(
        obs_proc.ObservationFilterType.epochs_filtering,
        list(epochs),
        filter_out=filter_out,
        use_opposite_condition=False,
    )


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
