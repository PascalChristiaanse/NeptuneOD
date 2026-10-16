"""Data models for the shared observation-grouping system.

Grouping partitions the observations of a single observation set by time.
It is used by **both** the weighting subsystem and the outlier rejection
subsystem, but each subsystem derives its own grouping from its own config
(an outlier-rejection group is not necessarily a weighting group).

A grouping is built from one or more *partitioning levels* applied
coarse-to-fine:

- ``set``       — the whole set (always implicit).
- ``section``   — named absolute time windows (e.g. pre/post Voyager encounter).
- ``timeframe`` — clusters separated by time gaps (e.g. observing nights).

The order of levels is significant and is preserved: a strategy that needs
"split by section, then split by timeframe" walks the levels in that order.
The order is recorded in :attr:`Group.layer` (0-based, counting only
*partitioning* levels; the implicit set level is ``-1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Group:
    """A single group of observations within one set.

    Attributes
    ----------
    indices : np.ndarray
        Integer indices **into the observation set** (0-based) belonging to
        this group.
    group_id : str
        Globally unique, human-readable identifier (``"<set>_<name>"``).
    level : str
        The partitioning level that produced this group (``"set"``,
        ``"section"``, ``"timeframe"``).
    parent_level : str | None
        The coarser level containing this group, if any.
    parent_id : str | None
        The ``group_id`` of the parent at the coarser level, if any.
    layer : int
        The partitioning layer index (0-based, coarsest first).  The implicit
        set level uses ``-1``.
    layer_name : str
        Human-readable label for the layer (e.g. ``"section"``, ``"timeframe"``).
    metadata : dict
        Extra metadata (time bounds, section name, observation count, ...).
    """

    indices: np.ndarray
    group_id: str
    level: str = "timeframe"
    parent_level: str | None = None
    parent_id: str | None = None
    layer: int = 0
    layer_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of observations in this group."""
        return len(self.indices)

    @property
    def name(self) -> str:
        """Short name of this group (the section name if present, else ``group_id``)."""
        return str(self.metadata.get("section_name") or self.group_id)

    @property
    def time_bounds(self) -> tuple[float | None, float | None]:
        """``(t_start, t_end)`` metadata if known, else ``(None, None)``."""
        return self.metadata.get("t_start"), self.metadata.get("t_end")


@dataclass
class SectionInfo:
    """Lightweight descriptor of a section group."""

    name: str
    group_id: str
    t_start: float | None
    t_end: float | None
    n_obs: int
    layer: int = 0


@dataclass
class TimeframeInfo:
    """Lightweight descriptor of a timeframe group."""

    group_id: str
    name: str
    t_start: float | None
    t_end: float | None
    n_obs: int
    parent_level: str | None
    parent_id: str | None
    layer: int = 0


class GroupList:
    """Ordered collection of :class:`Group` objects.

    Groups are stored as a flat list and can be subset by level, layer or
    parent.  The preserving order of the levels is available via
    :attr:`levels` and :attr:`layer_count`.
    """

    def __init__(self, groups: list[Group] | None = None):
        self._groups: list[Group] = list(groups) if groups else []

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups)

    def __getitem__(self, index: int) -> Group:
        return self._groups[index]

    def add(self, group: Group) -> None:
        self._groups.append(group)

    def extend(self, groups: list[Group]) -> None:
        self._groups.extend(groups)

    @property
    def groups(self) -> list[Group]:
        """All groups (read-only view)."""
        return list(self._groups)

    def by_level(self, level: str) -> list[Group]:
        """Return groups at a specific level."""
        return [g for g in self._groups if g.level == level]

    def by_layer(self, layer: int) -> list[Group]:
        """Return groups at a specific partitioning layer."""
        return [g for g in self._groups if g.layer == layer]

    def by_parent(self, parent_id: str) -> list[Group]:
        """Return groups with a specific parent group id."""
        return [g for g in self._groups if g.parent_id == parent_id]

    @property
    def layers(self) -> list[int]:
        """Sorted list of distinct partitioning layers present (excluding the set layer)."""
        return sorted({g.layer for g in self._groups if g.layer >= 0})

    @property
    def layer_count(self) -> int:
        """Number of partitioning layers present."""
        return len(self.layers)

    @property
    def level_order(self) -> list[str]:
        """The distinct level names in the order of their layers (coarsest first)."""
        seen: dict[str, int] = {}
        for g in self._groups:
            if g.layer >= 0 and g.level not in seen:
                seen[g.level] = g.layer
        return [name for name, _ in sorted(seen.items(), key=lambda kv: kv[1])]

    def sections(self) -> list[Group]:
        """All section-level groups."""
        return self.by_level("section")

    def timeframes(self) -> list[Group]:
        """All timeframe-level groups."""
        return self.by_level("timeframe")

    def section_infos(self) -> list[SectionInfo]:
        """Descriptors for every section group."""
        return [
            SectionInfo(
                name=str(g.metadata.get("section_name", g.group_id)),
                group_id=g.group_id,
                t_start=g.metadata.get("t_start"),
                t_end=g.metadata.get("t_end"),
                n_obs=g.size,
                layer=g.layer,
            )
            for g in self.sections()
        ]

    def timeframe_infos(self) -> list[TimeframeInfo]:
        """Descriptors for every timeframe group."""
        return [
            TimeframeInfo(
                group_id=g.group_id,
                name=str(g.metadata.get("section_name") or g.group_id),
                t_start=g.metadata.get("t_start"),
                t_end=g.metadata.get("t_end"),
                n_obs=g.size,
                parent_level=g.parent_level,
                parent_id=g.parent_id,
                layer=g.layer,
            )
            for g in self.timeframes()
        ]

    def filter_to(self, indices: np.ndarray) -> GroupList:
        """Return a copy restricted to *indices*, dropping empties.

        ``indices`` are treated as **global positions** into the original set;
        the returned groups keep their global indices, filtered to the subset
        that survives.  This is useful when outlier rejection removes
        observations and a previously-built grouping must be re-applied.
        """
        keep = set(int(i) for i in indices)
        new_groups: list[Group] = []
        for g in self._groups:
            surviving = np.array([i for i in g.indices if int(i) in keep], dtype=int)
            if surviving.size == 0:
                continue
            new_groups.append(
                Group(
                    indices=surviving,
                    group_id=g.group_id,
                    level=g.level,
                    parent_level=g.parent_level,
                    parent_id=g.parent_id,
                    layer=g.layer,
                    layer_name=g.layer_name,
                    metadata=dict(g.metadata),
                )
            )
        return GroupList(new_groups)
