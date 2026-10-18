"""Grouping specification and the :func:`build_group_list` pipeline.

A *grouping* is an ordered list of partitioning **levels** applied
coarse-to-fine.  Each level independently partitions the whole set; the levels
are then linked into a hierarchy via parent references.  Any number of levels
is supported (not just two).

Example configuration::

    grouping:
      levels:
        - type: section
          sections:
            - {name: "pre_encounter",  start: -1e30,       end: -326688543.8172541}
            - {name: "post_encounter", start: -326688543.8172541, end: 1e30}
        - type: timeframe
          gap_threshold_hours: 8.0

Produces section groups at layer 0 and timeframe groups at layer 1, each
timeframe carrying a ``parent_id`` pointing at its enclosing section.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .models import Group, GroupList
from .partitioners import (
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
)

logger = logging.getLogger(__name__)


def _as_config(config: Any) -> DictConfig | None:
    """Coerce a mapping/config to a DictConfig (plain dicts are accepted)."""
    if config is None:
        return None
    if isinstance(config, DictConfig):
        return config
    if isinstance(config, Mapping):
        return OmegaConf.create(config)
    if isinstance(config, str):
        return OmegaConf.create({"levels": []})
    return config


def build_group_list(
    times: np.ndarray,
    set_id: str,
    config: DictConfig | Mapping | None,
) -> GroupList:
    """Build a :class:`GroupList` from a grouping configuration.

    Parameters
    ----------
    times : np.ndarray
        Sorted observation times (seconds since J2000).
    set_id : str
        Set identifier.
    config : DictConfig | Mapping | None
        Grouping configuration with a ``levels`` list.  Plain mappings are
        accepted and coerced to an OmegaConf node.

    Returns
    -------
    GroupList
        Flattened list of groups across every partitioning level, with parent
        references linking finer levels to their coarse parents.
    """
    n_obs = len(times)

    config = _as_config(config)
    if config is None:
        return GroupList(partition_by_set(n_obs, set_id))

    levels_cfg = OmegaConf.select(config, "levels")
    if not levels_cfg:
        return GroupList(partition_by_set(n_obs, set_id))

    set_group = partition_by_set(n_obs, set_id)
    set_group_id = set_group[0].group_id

    gl = GroupList()
    previous_groups: list[Group] = []  # groups from the immediately coarser level
    layer = 0

    for level_entry in levels_cfg:
        level_type = OmegaConf.select(level_entry, "type")
        if level_type is None:
            raise ValueError("Each grouping level must have a 'type' field.")

        if level_type == "set":
            # An explicit set level is a real, addressable partitioning layer.
            # This lets a strategy use "the whole set" as its primary layer
            # (e.g. per-file RMSE) and a finer level as its secondary layer.
            level_groups = _build_set_level(times, set_id, layer, n_obs)
        elif level_type == "section":
            level_groups = _build_section_level(level_entry, times, set_id, layer, n_obs)
        elif level_type == "timeframe":
            level_groups = _build_timeframe_level(level_entry, times, set_id, layer)
        else:
            raise ValueError(f"Unknown grouping level type: '{level_type}'")

        # Link each group to its parent in the coarser level.
        if not previous_groups:
            for g in level_groups:
                g.parent_level = "set"
                g.parent_id = set_group_id
        else:
            _link_parents(level_groups, previous_groups)

        gl.extend(level_groups)
        previous_groups = level_groups
        layer += 1

    return gl


# ---------------------------------------------------------------------------
# Per-level builders
# ---------------------------------------------------------------------------


def _build_section_level(
    level_entry: Any,
    times: np.ndarray,
    set_id: str,
    layer: int,
    n_obs: int,
) -> list[Group]:
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

    groups = partition_by_sections(times, section_bounds, set_id, layer=layer)

    # Observations outside every section fall into a catch-all group.
    assigned: set[int] = set()
    for g in groups:
        assigned.update(int(i) for i in g.indices)
    unassigned = [i for i in range(n_obs) if i not in assigned]
    if unassigned:
        groups.append(
            Group(
                indices=np.array(unassigned, dtype=int),
                group_id=f"{set_id}_other" if set_id else "other",
                level="section",
                layer=layer,
                layer_name="section",
                metadata={
                    "n_obs": len(unassigned),
                    "section_name": "other",
                },
            )
        )
    return groups


def _build_set_level(
    times: np.ndarray,
    set_id: str,
    layer: int,
    n_obs: int,
) -> list[Group]:
    """Build an explicit set-level partitioning group.

    The result is identical in content to the implicit set group but carries a
    real ``layer`` so it can be addressed as a primary partitioning layer.
    """
    return [
        Group(
            indices=np.arange(n_obs, dtype=int),
            group_id=set_id,
            level="set",
            layer=layer,
            layer_name="set",
            metadata={"n_obs": n_obs},
        )
    ]


def _build_timeframe_level(
    level_entry: Any,
    times: np.ndarray,
    set_id: str,
    layer: int,
) -> list[Group]:
    gap_hours = float(OmegaConf.select(level_entry, "gap_threshold_hours", default=4.0))
    min_obs = int(OmegaConf.select(level_entry, "min_obs_per_frame", default=1))
    return partition_by_timeframes(times, gap_hours, min_obs, set_id, layer=layer)


def _link_parents(children: list[Group], parents: list[Group]) -> None:
    """Attach each child to the first parent whose indices it overlaps."""
    for child in children:
        child_set = set(int(i) for i in child.indices)
        for parent in parents:
            if child_set & set(int(i) for i in parent.indices):
                child.parent_level = parent.level
                child.parent_id = parent.group_id
                break


__all__ = ["build_group_list"]
