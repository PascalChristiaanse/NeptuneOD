"""Grouping specification and the :func:`build_cluster_list` pipeline.

A *clustering* is an ordered list of partitioning **levels** applied
coarse-to-fine.  Each level independently partitions the whole set; the levels
are then linked into a hierarchy via parent references.  Any number of levels
is supported (not just two).

Example configuration::

    clustering:
      levels:
        - type: section
          sections:
            - {name: "pre_encounter",  start: -1e30,       end: -326688543.8172541}
            - {name: "post_encounter", start: -326688543.8172541, end: 1e30}
        - type: timeframe
          gap_threshold_hours: 8.0

Produces section clusters at layer 0 and timeframe clusters at layer 1, each
timeframe carrying a ``parent_id`` pointing at its enclosing section.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .models import Cluster, ClusterList
from .partitioners import (
    partition_by_column,
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


def build_cluster_list(
    times: np.ndarray,
    set_id: str,
    config: DictConfig | Mapping | None,
    view=None,
) -> ClusterList:
    """Build a :class:`ClusterList` from a clustering configuration.

    Parameters
    ----------
    times : np.ndarray
        Sorted observation times (seconds since J2000).
    set_id : str
        Set identifier.
    config : DictConfig | Mapping | None
        Clustering configuration with a ``levels`` list.  Plain mappings are
        accepted and coerced to an OmegaConf node.
    view : SetView | None
        The set's window onto the access layer.  Required by the ``column``
        level, which partitions on a standard-frame column; the time-based
        levels ignore it.

    Returns
    -------
    ClusterList
        Flattened list of clusters across every partitioning level, with parent
        references linking finer levels to their coarse parents.
    """
    n_obs = len(times)

    config = _as_config(config)
    if config is None:
        return ClusterList(partition_by_set(n_obs, set_id))

    levels_cfg = OmegaConf.select(config, "levels")
    if not levels_cfg:
        return ClusterList(partition_by_set(n_obs, set_id))

    set_cluster = partition_by_set(n_obs, set_id)
    set_cluster_id = set_cluster[0].cluster_id

    gl = ClusterList()
    previous_clusters: list[Cluster] = []  # clusters from the immediately coarser level
    layer = 0

    for level_entry in levels_cfg:
        level_type = OmegaConf.select(level_entry, "type")
        if level_type is None:
            raise ValueError("Each clustering level must have a 'type' field.")

        if level_type == "set":
            # An explicit set level is a real, addressable partitioning layer.
            # This lets a strategy use "the whole set" as its primary layer
            # (e.g. per-file RMSE) and a finer level as its secondary layer.
            level_clusters = _build_set_level(times, set_id, layer, n_obs)
        elif level_type == "section":
            level_clusters = _build_section_level(level_entry, times, set_id, layer, n_obs)
        elif level_type == "timeframe":
            level_clusters = _build_timeframe_level(level_entry, times, set_id, layer)
        elif level_type == "column":
            level_clusters = _build_column_level(level_entry, set_id, layer, view)
        else:
            raise ValueError(f"Unknown clustering level type: '{level_type}'")

        # Link each cluster to its parent in the coarser level.
        if not previous_clusters:
            for g in level_clusters:
                g.parent_level = "set"
                g.parent_id = set_cluster_id
        else:
            _link_parents(level_clusters, previous_clusters)

        gl.extend(level_clusters)
        previous_clusters = level_clusters
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
) -> list[Cluster]:
    sections_cfg = OmegaConf.select(level_entry, "sections")
    if not sections_cfg:
        raise ValueError("Section clustering requires a 'sections' list.")

    section_bounds: list[tuple[str, float, float]] = []
    for sc in sections_cfg:
        name = OmegaConf.select(sc, "name")
        t_start = OmegaConf.select(sc, "start")
        t_end = OmegaConf.select(sc, "end")
        if name is None or t_start is None or t_end is None:
            raise ValueError("Each section must have 'name', 'start', and 'end' fields.")
        section_bounds.append((name, float(t_start), float(t_end)))

    clusters = partition_by_sections(times, section_bounds, set_id, layer=layer)

    # Observations outside every section fall into a catch-all cluster.
    assigned: set[int] = set()
    for g in clusters:
        assigned.update(int(i) for i in g.indices)
    unassigned = [i for i in range(n_obs) if i not in assigned]
    if unassigned:
        clusters.append(
            Cluster(
                indices=np.array(unassigned, dtype=int),
                cluster_id=f"{set_id}_other" if set_id else "other",
                level="section",
                layer=layer,
                layer_name="section",
                metadata={
                    "n_obs": len(unassigned),
                    "section_name": "other",
                },
            )
        )
    return clusters


def _build_set_level(
    times: np.ndarray,
    set_id: str,
    layer: int,
    n_obs: int,
) -> list[Cluster]:
    """Build an explicit set-level partitioning cluster.

    The result is identical in content to the implicit set cluster but carries a
    real ``layer`` so it can be addressed as a primary partitioning layer.
    """
    return [
        Cluster(
            indices=np.arange(n_obs, dtype=int),
            cluster_id=set_id,
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
) -> list[Cluster]:
    gap_hours = float(OmegaConf.select(level_entry, "gap_threshold_hours", default=4.0))
    min_obs = int(OmegaConf.select(level_entry, "min_obs_per_frame", default=1))
    return partition_by_timeframes(times, gap_hours, min_obs, set_id, layer=layer)


def _build_column_level(
    level_entry: Any,
    set_id: str,
    layer: int,
    view,
) -> list[Cluster]:
    """Partition on a standard-frame column (``type: column``).

    This is the level that lets a clustering config name a column directly::

        - type: column
          column: data_source

    It needs the set's :class:`~orbitdet.analysis.set_view.SetView`, because the
    values live in the frame rather than in the observation times.
    """
    column = OmegaConf.select(level_entry, "column")
    if not column:
        raise ValueError("A 'column' clustering level requires a 'column' name.")
    if view is None:
        raise ValueError(
            f"The 'column' clustering level needs the set's frame to read '{column}', "
            f"but no view was supplied.  Pass access=... to the engine, or use a "
            f"time-based level instead."
        )
    try:
        values = view.column(str(column))
    except KeyError as exc:
        raise ValueError(
            f"The 'column' clustering level names '{column}', which is not a "
            f"standard-frame column."
        ) from exc
    return partition_by_column(values, set_id, layer=layer, column=str(column))


def _link_parents(children: list[Cluster], parents: list[Cluster]) -> None:
    """Attach each child to the first parent whose indices it overlaps."""
    for child in children:
        child_set = set(int(i) for i in child.indices)
        for parent in parents:
            if child_set & set(int(i) for i in parent.indices):
                child.parent_level = parent.level
                child.parent_id = parent.cluster_id
                break


__all__ = ["build_cluster_list"]
