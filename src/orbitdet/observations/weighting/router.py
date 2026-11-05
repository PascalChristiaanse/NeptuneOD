"""Per-set weighting router.

Dispatches each observation set to a sub-strategy chosen by matching rules,
so different datasets in the same collection (Voyager 2, Gaia, NSDB, ...) can
use different weighting schemes in a single configuration.

Each route can also override the clustering used for its set, which matters when
different datasets need different partitions (e.g. Voyager needs a
section+timeframe split, NSDB needs a set+timeframe split).

Example::

    weighting:
      strategy:
        type: router
        default:                     # applied to unmatched sets
          type: id_v2
        routes:
          - match: {set_id_contains: [Gaia]}
            source: {type: gaia_archive, covariance_mode: full_matrix}
          - match: {set_id_contains: ["Voyager 2"]}
            source:
              type: layered
              combine: geometric
              layers:
                - {level: section, mode: fixed,
                   sigmas: {pre_encounter: {ra: 1.0, dec: 1.0},
                            post_encounter: {ra: 4.0, dec: 4.0}}}
                - {level: timeframe, mode: rmse_descaled}
            clustering:                # overrides the engine clustering for this set
              levels:
                - type: section
                  sections: [...]
                - type: timeframe
                  gap_threshold_hours: 8.0
          - match: {set_id_contains: ["689", "675", "NSDB"]}
            source:
              type: layered
              combine: geometric
              layers:
                - {level: set, mode: rmse}
                - {level: timeframe, mode: rmse_descaled}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import tudatpy.estimation.observations as obs

from ..clustering import ClusterList
from .base import WeightStrategy
from .builder import build_strategy
from .registry import register_weight_strategy

logger = logging.getLogger(__name__)

#: Supported match keys.
_MATCH_KEYS = ("set_id", "set_id_contains", "set_id_matches")


@register_weight_strategy("router")
class RouterWeight(WeightStrategy):
    """Route each set to a sub-strategy by matching rules.

    Parameters
    ----------
    routes : list[dict]
        Ordered routing rules.  Each rule has:

        ``match``
            A mapping with one or more of ``set_id`` (exact),
            ``set_id_contains`` (substring(s)), ``set_id_matches`` (regex).
        ``source``
            The sub-strategy descriptor to use (may be a plain name or a
            mapping with a ``type``).
        ``clustering`` (optional)
            A clustering config overriding the engine clustering for matched sets.

        The first matching route wins.
    default : dict | str | None
        Strategy descriptor used for unmatched sets.  Defaults to ``id_v2``.
    """

    def __init__(
        self,
        routes: list[dict[str, Any]] | None = None,
        default: Any = None,
    ):
        # Routes are optional: a router with only a ``default`` is the
        # degenerate single-strategy case, kept so every weighting config has
        # exactly one shape (``type: router`` + ``default`` [+ ``routes``]).
        self._routes = [self._compile_route(r) for r in (routes or [])]
        self._default_spec = default if default is not None else "id_v2"
        self._default_strategy = None  # built lazily
        self._clustering = None
        # Cache built strategies keyed by route index (and "default").
        self._cache: dict[Any, WeightStrategy] = {}
        # Tracks full-matrix preservation per route, computed lazily.
        self._preserve: bool | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_route(route: Any) -> dict[str, Any]:
        from .builder import to_plain

        plain = to_plain(route)
        if "match" not in plain or "source" not in plain:
            raise ValueError("Each route must have 'match' and 'source' keys.")
        match = dict(plain["match"])
        unknown = set(match) - set(_MATCH_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown match key(s) {sorted(unknown)}; expected {list(_MATCH_KEYS)}."
            )
        return {
            "match": match,
            "source": plain["source"],
            "clustering": plain.get("clustering"),
        }

    def set_clustering(self, clustering) -> None:
        """Attach the engine clustering (used by routes without their own)."""
        self._clustering = clustering

    def _strategy_for(self, key: Any, spec: Any, clustering) -> WeightStrategy:
        if key not in self._cache:
            self._cache[key] = build_strategy(spec, clustering=clustering)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(match: dict[str, Any], set_id: str) -> bool:
        low = set_id.lower()
        if "set_id" in match:
            values = match["set_id"]
            values = [values] if isinstance(values, (str, int)) else list(values)
            if set_id not in {str(v) for v in values}:
                return False
        if "set_id_contains" in match:
            values = match["set_id_contains"]
            values = [values] if isinstance(values, str) else list(values)
            if not any(str(v).lower() in low for v in values):
                return False
        if "set_id_matches" in match:
            import re

            values = match["set_id_matches"]
            patterns = [values] if isinstance(values, str) else list(values)
            if not any(re.search(str(p), set_id) for p in patterns):
                return False
        return True

    def _route_for(self, set_id: str) -> tuple[Any, Any, Any]:
        """Return ``(key, source_spec, clustering)`` for *set_id*."""
        for i, route in enumerate(self._routes):
            if self._matches(route["match"], set_id):
                return i, route["source"], route["clustering"] or self._clustering
        return "default", self._default_spec, self._clustering

    # ------------------------------------------------------------------
    # WeightStrategy API
    # ------------------------------------------------------------------

    @property
    def preserve_full_weight_matrix(self) -> bool:
        """True if the route a set would take preserves its full matrix.

        The engine checks this once on the top-level strategy, so a router that
        can dispatch *any* set to a matrix-preserving sub-strategy reports
        True.  This mirrors the pre-existing behaviour for combined strategies
        and is refined per-set below via :meth:`compute_weights`.
        """
        if self._preserve is None:
            specs = [r["source"] for r in self._routes] + [self._default_spec]
            self._preserve = False
            for spec in specs:
                try:
                    strat = build_strategy(spec, clustering=self._clustering)
                except Exception:  # pragma: no cover - defensive
                    continue
                if getattr(strat, "preserve_full_weight_matrix", False):
                    self._preserve = True
                    break
        return self._preserve

    def compute_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        key, spec, route_clustering = self._route_for(set_id)

        # If the route overrides clustering, rebuild this set's clusters.
        if route_clustering is not None and route_clustering is not self._clustering:
            from ..clustering import build_cluster_list

            times = np.array([t.to_float() for t in observation_set.observation_times])
            clusters = build_cluster_list(times, set_id, route_clustering)

        strategy = self._strategy_for(key, spec, route_clustering)
        logger.debug(
            "RouterWeight [%s]: route '%s' -> %s",
            set_id,
            key,
            strategy.__class__.__name__,
        )
        return strategy.compute_weights(
            observation_set, clusters, set_id, min_sigma_arcsec, view=view
        )

    def route_source(self, set_id: str) -> WeightStrategy:
        """Return the sub-strategy that would handle *set_id* (for diagnostics)."""
        key, spec, route_clustering = self._route_for(set_id)
        return self._strategy_for(key, spec, route_clustering)

    def compute_block_weights(
        self,
        observation_set: obs.SingleObservationSet,
        clusters: ClusterList,
        set_id: str,
        min_sigma_arcsec: float = 0.01,
        view=None,
    ) -> list[np.ndarray] | None:
        """Delegate block-weight computation to the routed sub-strategy."""
        key, spec, route_clustering = self._route_for(set_id)

        if route_clustering is not None and route_clustering is not self._clustering:
            from ..clustering import build_cluster_list

            times = np.array([t.to_float() for t in observation_set.observation_times])
            clusters = build_cluster_list(times, set_id, route_clustering)

        strategy = self._strategy_for(key, spec, route_clustering)
        compute_blocks = getattr(strategy, "compute_block_weights", None)
        if not callable(compute_blocks):
            return None
        return compute_blocks(observation_set, clusters, set_id, min_sigma_arcsec, view=view)
