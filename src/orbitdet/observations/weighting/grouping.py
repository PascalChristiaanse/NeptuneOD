"""Grouping system for observation weighting.

Provides a flexible grouping abstraction that supports multiple partitioning
strategies:

- **Set-level**: all observations in a set form a single group.
- **Timeframes**: groups defined by time gaps (e.g. observing nights).
- **Sections**: groups defined by absolute time boundaries (e.g. pre/post
  Voyager encounter).

Groups can be **nested** — a section can contain multiple timeframes, and
each level can be used independently for weight computation.  The
:class:`GroupList` holds the flattened list of atomic groups at the finest
granularity, each annotated with metadata that identifies its parent groups
at coarser levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Group:
    """A single group of observations.

    Attributes
    ----------
    indices : np.ndarray
        Integer indices into the observation set belonging to this group.
    group_id : str
        Human-readable identifier for this group.
    level : str
        The partitioning level that produced this group (``"set"``,
        ``"section"``, ``"timeframe"``).
    parent_level : str | None
        The coarser level containing this group, if any.
    parent_id : str | None
        The group_id of the parent at the coarser level, if any.
    metadata : dict
        Additional metadata (e.g. time bounds for sections).
    """

    indices: np.ndarray
    group_id: str
    level: str = "timeframe"
    parent_level: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of observations in this group."""
        return len(self.indices)


class GroupList:
    """Ordered collection of :class:`Group` objects.

    Groups are stored as a flat list and can be subset by level or parent.
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

    def by_parent(self, parent_id: str) -> list[Group]:
        """Return groups with a specific parent group id."""
        return [g for g in self._groups if g.parent_id == parent_id]


# ---------------------------------------------------------------------------
# Partitioners
# ---------------------------------------------------------------------------


def partition_by_set(
    n_obs: int,
    set_id: str,
) -> list[Group]:
    """Create a single set-level group containing all observations.

    Parameters
    ----------
    n_obs : int
        Number of observations in the set.
    set_id : str
        Identifier for the set.

    Returns
    -------
    list[Group]
        A list with one set-level group.
    """
    return [
        Group(
            indices=np.arange(n_obs, dtype=int),
            group_id=f"{set_id}",
            level="set",
        )
    ]


def partition_by_timeframes(
    times: np.ndarray,
    gap_threshold_hours: float = 4.0,
    min_obs_per_frame: int = 1,
    set_id: str = "",
) -> list[Group]:
    """Split observations into timeframes (nights) based on time gaps.

    Parameters
    ----------
    times : np.ndarray
        Sorted observation times in seconds since J2000.
    gap_threshold_hours : float
        Maximum gap (hours) within a timeframe.  Default: 4.0.
    min_obs_per_frame : int
        Minimum observations before a gap triggers a new frame.  Default: 1.
    set_id : str
        Identifier for the set, used in group ids.

    Returns
    -------
    list[Group]
        Groups, one per timeframe.
    """
    gap_threshold_sec = gap_threshold_hours * 3600.0
    n = len(times)

    if n == 0:
        return []

    frame_indices = _split_by_gap(times, gap_threshold_sec, min_obs_per_frame)
    groups: list[Group] = []
    for i, idx in enumerate(frame_indices):
        groups.append(
            Group(
                indices=np.array(idx, dtype=int),
                group_id=f"{set_id}_tf_{i:04d}" if set_id else f"tf_{i:04d}",
                level="timeframe",
                metadata={
                    "t_start": float(times[idx[0]]),
                    "t_end": float(times[idx[-1]]),
                    "n_obs": len(idx),
                },
            )
        )
    return groups


def partition_by_sections(
    times: np.ndarray,
    section_bounds: list[tuple[str, float, float]],
    set_id: str = "",
) -> list[Group]:
    """Split observations into named time intervals.

    Parameters
    ----------
    times : np.ndarray
        Observation times in seconds since J2000.
    section_bounds : list[tuple[str, float, float]]
        List of ``(section_name, t_start, t_end)`` tuples defining each
        section's time window.  Sections are applied in order; an observation
        falls into the **first** matching section.
    set_id : str
        Identifier for the set, used in group ids.

    Returns
    -------
    list[Group]
        Groups, one per section with observations in it.
    """
    groups: list[Group] = []
    assigned = np.zeros(len(times), dtype=bool)

    for name, t_start, t_end in section_bounds:
        mask = (times >= t_start) & (times <= t_end) & ~assigned
        indices = np.where(mask)[0]
        if len(indices) > 0:
            assigned[indices] = True
            groups.append(
                Group(
                    indices=indices,
                    group_id=f"{set_id}_{name}" if set_id else name,
                    level="section",
                    metadata={
                        "t_start": t_start,
                        "t_end": t_end,
                        "n_obs": len(indices),
                        "section_name": name,
                    },
                )
            )

    return groups


# ---------------------------------------------------------------------------
# Grouping pipeline
# ---------------------------------------------------------------------------


def build_group_list(
    times: np.ndarray,
    set_id: str,
    config: DictConfig | None,
) -> GroupList:
    """Build a :class:`GroupList` from a grouping configuration.

    The config specifies a list of partitioning **levels** applied top-down
    (coarse-to-fine).  Each level refines the previous one (or starts from
    the set level).

    Configuration structure::

        grouping:
          levels:
            - type: section
              sections:
                - {name: "pre_encounter", start: ..., end: ...}
                - {name: "post_encounter", start: ..., end: ...}
            - type: timeframe
              gap_threshold_hours: 4.0

    Parameters
    ----------
    times : np.ndarray
        Sorted observation times.
    set_id : str
        Set identifier.
    config : DictConfig
        Grouping configuration from the weighting config.

    Returns
    -------
    GroupList
        Flattened list of atomic groups with parent references.
    """
    n_obs = len(times)

    if config is None:
        # No grouping config — single set-level group
        return GroupList(partition_by_set(n_obs, set_id))

    levels_cfg = OmegaConf.select(config, "levels")

    if not levels_cfg:
        # Default: single set-level group
        return GroupList(partition_by_set(n_obs, set_id))

    # Determine the effective set-level group
    gl = GroupList()
    set_group = partition_by_set(n_obs, set_id)

    # Process levels top-down (coarse-to-fine), tracking the current set of
    # children at each level.
    current_base_groups: list[Group] = set_group  # groups from previous level

    for level_entry in levels_cfg:
        level_type = OmegaConf.select(level_entry, "type")
        if level_type is None:
            raise ValueError("Each grouping level must have a 'type' field.")

        if level_type == "set":
            # Explicit set-level: replaces the default set group
            continue

        elif level_type == "section":
            sections_cfg = OmegaConf.select(level_entry, "sections")
            if not sections_cfg:
                raise ValueError("Section grouping requires a 'sections' list.")

            section_bounds: list[tuple[str, float, float]] = []
            for sc in sections_cfg:
                name = OmegaConf.select(sc, "name")
                t_start = OmegaConf.select(sc, "start")
                t_end = OmegaConf.select(sc, "end")
                if name is None or t_start is None or t_end is None:
                    raise ValueError("Each section must have 'name', 'start', and 'end' fields.")
                section_bounds.append((name, float(t_start), float(t_end)))

            section_groups = partition_by_sections(times, section_bounds, set_id)

            # Unassigned observations fall into a catch-all group
            all_assigned = set()
            for sg in section_groups:
                all_assigned.update(sg.indices.tolist())
            unassigned = [i for i in range(n_obs) if i not in all_assigned]
            if unassigned:
                section_groups.append(
                    Group(
                        indices=np.array(unassigned, dtype=int),
                        group_id=f"{set_id}_other" if set_id else "other",
                        level="section",
                        metadata={
                            "n_obs": len(unassigned),
                            "section_name": "other",
                        },
                    )
                )

            # Add parent references from the previous level
            for sg in section_groups:
                sg.level = "section"
            gl.extend(section_groups)

            # Set up parent references: each previous-level group is a child
            # of a section
            for bg in current_base_groups:
                if bg.level == "set":
                    # Assign each timeframe to its parent section
                    for sg in section_groups:
                        child_in_parent = np.isin(bg.indices, sg.indices)
                        if np.any(child_in_parent):
                            bg.parent_level = "section"
                            bg.parent_id = sg.group_id
                            bg.level = bg.level  # keep original level
                            break

            current_base_groups = section_groups

        elif level_type == "timeframe":
            gap_hours = float(OmegaConf.select(level_entry, "gap_threshold_hours", default=4.0))
            min_obs = int(OmegaConf.select(level_entry, "min_obs_per_frame", default=1))
            tf_groups = partition_by_timeframes(times, gap_hours, min_obs, set_id)

            # Attach parent references
            if current_base_groups and current_base_groups[0].level != "set":
                # There's a coarser level (sections); find parent for each timeframe
                for tg in tf_groups:
                    for parent_group in current_base_groups:
                        child_in_parent = np.isin(tg.indices, parent_group.indices)
                        if np.any(child_in_parent):
                            tg.parent_level = parent_group.level
                            tg.parent_id = parent_group.group_id
            elif current_base_groups and current_base_groups[0].level == "set":
                # Sibling set-level — timeframe's parent is the set
                for tg in tf_groups:
                    tg.parent_level = "set"
                    tg.parent_id = current_base_groups[0].group_id

            gl.extend(tf_groups)
            current_base_groups = tf_groups

        else:
            raise ValueError(f"Unknown grouping level type: '{level_type}'")

    return gl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_by_gap(
    times: np.ndarray,
    gap_threshold_sec: float,
    min_obs_per_frame: int = 1,
) -> list[list[int]]:
    """Split sorted time indices into frames separated by gaps."""
    if len(times) == 0:
        return []

    frames: list[list[int]] = [[0]]
    for i in range(1, len(times)):
        gap = times[i] - times[i - 1]
        if gap > gap_threshold_sec and len(frames[-1]) >= min_obs_per_frame:
            frames.append([i])
        else:
            frames[-1].append(i)
    return frames
