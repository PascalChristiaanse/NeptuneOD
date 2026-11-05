"""Default-plus-overrides plans for the weighting and outlier engines.

A *plan* is the config shape both engines share::

    weighting:
      enabled: true
      default:
        strategy:   {type: id_v2}
        clustering: {levels: [{type: timeframe, gap_threshold_hours: 8.0}]}
      overrides:
        - select: "data_source == 'gaia_archive'"
          strategy: {type: gaia_archive, covariance_mode: full_matrix}
        - select: "observatory_code == 'Voyager 2'"
          strategy: {type: layered, ...}
          clustering: {levels: [{type: section, ...}, {type: timeframe, ...}]}

The first override whose :class:`~orbitdet.observations.selection.Selection`
matches a set wins; ``default`` catches everything else.  Each entry carries its
own strategy *and* its own clustering, so a rule reads as one thought: "for
these observations, weight them this way, partitioned like that".

Why this shape
--------------
The previous shape split the two halves across separate config sections: a
``router`` matched on the set identifier and chose a strategy, while a
top-level ``clustering`` (or a per-route ``clustering``) chose the partition.
That meant a rule was assembled from two places, and the match could only look
at the set identifier.  Here the match is any frame column, and the strategy and
clustering sit together.

The router is still supported — it is a strategy, and a plan's ``strategy`` may
be ``{type: router, ...}`` — but a plan does not need it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .selection import Selection, compile_selection

logger = logging.getLogger(__name__)

__all__ = ["PlanEntry", "Plan", "compile_plan"]


@dataclass
class PlanEntry:
    """One rule in a plan: a selection, a strategy descriptor and a clustering.

    Attributes
    ----------
    selection : Selection
        Which observations this entry applies to.  A ``default`` entry uses the
        universal selection (every row).
    strategy : Any
        A strategy descriptor, as accepted by
        :func:`orbitdet.observations.weighting.builder.build_strategy` (a name
        or a mapping with a ``type`` key).
    clustering : Any
        A clustering config node, or ``None`` to fall back to the plan's
        default clustering.
    label : str
        Human-readable name for logging and error messages (``"default"`` or
        the override's query).
    """

    selection: Selection
    strategy: Any
    clustering: Any = None
    label: str = "default"

    def __repr__(self) -> str:
        return f"PlanEntry({self.label!r})"


@dataclass
class Plan:
    """An ordered default-plus-overrides plan.

    Attributes
    ----------
    default : PlanEntry
        The entry used when no override matches.
    overrides : list[PlanEntry]
        Ordered overrides; the first match wins.
    """

    default: PlanEntry
    overrides: list[PlanEntry] = field(default_factory=list)

    def entry_for(self, frame: pd.DataFrame | None) -> PlanEntry:
        """Return the entry that applies to the observations in *frame*.

        Parameters
        ----------
        frame : pd.DataFrame | None
            The rows of one observation set, in the standard (or derived) frame.
            ``None`` when no frame is available (no access layer was supplied),
            in which case the default entry is returned: an override cannot be
            evaluated without rows to match on.

        Returns
        -------
        PlanEntry
            The first override whose selection matches any of those rows, or
            :attr:`default` when none does.

        Notes
        -----
        A set is routed as a whole, so an override matches when *any* of the
        set's rows satisfies it.  Splitting a single set across two strategies
        would mean applying two weight matrices to one set, which Tudat does not
        support; a set that straddles two selections therefore takes the first
        matching one.  Use clustering to partition *within* a set instead.
        """
        if frame is None:
            return self.default
        for entry in self.overrides:
            if entry.selection.matches(frame):
                return entry
        return self.default

    def __len__(self) -> int:
        return 1 + len(self.overrides)

    def __repr__(self) -> str:
        return f"Plan(default={self.default.label!r}, overrides={len(self.overrides)})"


def compile_plan(
    cfg: Any,
    *,
    default_strategy: Any = None,
    default_clustering: Any = None,
) -> Plan:
    """Build a :class:`Plan` from a config node.

    Parameters
    ----------
    cfg : DictConfig | Mapping
        The engine's config node (``weighting`` or ``outlier_rejection``).  It
        may carry ``default`` and ``overrides`` keys.  For backwards
        compatibility a bare ``strategy`` key is accepted as the default entry,
        and a bare ``clustering`` key as the default clustering.
    default_strategy : Any, optional
        Strategy descriptor to use when the config names none.  Required if the
        config has neither ``default`` nor ``strategy``.
    default_clustering : Any, optional
        Clustering to use for entries that name none.

    Returns
    -------
    Plan
        The compiled plan.

    Raises
    ------
    ValueError
        If no default strategy can be determined, or an override is malformed.
    """
    plain = _to_plain(cfg)

    # A bare `clustering:` at the top level is the plan-wide default.
    plan_clustering = plain.get("clustering", default_clustering)

    default_node = plain.get("default")
    if default_node is None:
        # Legacy shape: `strategy:` at the top level is the default entry.
        if "strategy" in plain:
            default_node = {"strategy": plain["strategy"]}
        elif default_strategy is not None:
            default_node = {"strategy": default_strategy}
        else:
            raise ValueError(
                "The config has no 'default' entry and no top-level 'strategy'; "
                "one of them is required to say what happens to unmatched observations."
            )

    default_entry = _compile_entry(
        default_node,
        label="default",
        fallback_clustering=plan_clustering,
        require_selection=False,
    )

    overrides_node = plain.get("overrides") or []
    overrides: list[PlanEntry] = []
    for i, node in enumerate(overrides_node):
        overrides.append(
            _compile_entry(
                node,
                label=f"overrides[{i}]",
                fallback_clustering=plan_clustering,
                require_selection=True,
            )
        )

    return Plan(default=default_entry, overrides=overrides)


def _compile_entry(
    node: Any,
    *,
    label: str,
    fallback_clustering: Any,
    require_selection: bool,
) -> PlanEntry:
    """Compile one plan entry, validating its keys."""
    plain = _to_plain(node)
    if not isinstance(plain, Mapping):
        raise TypeError(
            f"Plan entry {label} must be a mapping with 'strategy' (and optionally "
            f"'select' and 'clustering'), got {type(plain).__name__}."
        )

    unknown = set(plain) - {"select", "strategy", "clustering"}
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in plan entry {label}; "
            f"expected 'select', 'strategy' and/or 'clustering'."
        )

    if "strategy" not in plain:
        raise ValueError(f"Plan entry {label} has no 'strategy'.")

    if require_selection and "select" not in plain:
        raise ValueError(
            f"Plan entry {label} has no 'select'.  An override must say which "
            f"observations it applies to; use 'default' for the catch-all."
        )

    selection = compile_selection(plain.get("select"))
    if require_selection and selection.is_universal:
        raise ValueError(
            f"Plan entry {label} has an empty 'select', which would match every "
            f"observation and shadow the entries after it."
        )

    clustering = plain.get("clustering", fallback_clustering)
    return PlanEntry(
        selection=selection,
        strategy=plain["strategy"],
        clustering=clustering,
        label=label if selection.is_universal else f"{label}: {selection.description}",
    )


def _to_plain(value: Any) -> Any:
    """Convert an OmegaConf node to plain Python containers."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value
