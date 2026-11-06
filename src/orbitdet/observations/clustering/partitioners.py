"""Low-level partitioners for the shared observation-clustering system.

These functions split a set of observation times into atomic clusters.  They are
mechanism-only: they know nothing about weighting or outlier rejection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import Cluster


def partition_by_set(
    n_obs: int,
    set_id: str,
) -> list[Cluster]:
    """Create a single set-level cluster containing all observations.

    Parameters
    ----------
    n_obs : int
        Number of observations in the set.
    set_id : str
        Identifier for the set.

    Returns
    -------
    list[Cluster]
        A list with one set-level cluster.
    """
    return [
        Cluster(
            indices=np.arange(n_obs, dtype=int),
            cluster_id=f"{set_id}",
            level="set",
            layer=-1,
            layer_name="set",
        )
    ]


def partition_by_timeframes(
    times: np.ndarray,
    gap_threshold_hours: float = 4.0,
    min_obs_per_frame: int = 1,
    set_id: str = "",
    layer: int = 0,
) -> list[Cluster]:
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
        Identifier for the set, used in cluster ids.
    layer : int
        Partitioning layer index for the produced clusters.

    Returns
    -------
    list[Cluster]
        Groups, one per timeframe.
    """
    gap_threshold_sec = gap_threshold_hours * 3600.0
    n = len(times)

    if n == 0:
        return []

    frame_indices = _split_by_gap(times, gap_threshold_sec, min_obs_per_frame)
    clusters: list[Cluster] = []
    for i, idx in enumerate(frame_indices):
        clusters.append(
            Cluster(
                indices=np.array(idx, dtype=int),
                cluster_id=f"{set_id}_tf_{i:04d}" if set_id else f"tf_{i:04d}",
                level="timeframe",
                layer=layer,
                layer_name="timeframe",
                metadata={
                    "t_start": float(times[idx[0]]),
                    "t_end": float(times[idx[-1]]),
                    "n_obs": len(idx),
                },
            )
        )
    return clusters


def partition_by_sections(
    times: np.ndarray,
    section_bounds: list[tuple[str, float, float]],
    set_id: str = "",
    layer: int = 0,
) -> list[Cluster]:
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
        Identifier for the set, used in cluster ids.
    layer : int
        Partitioning layer index for the produced clusters.

    Returns
    -------
    list[Cluster]
        Groups, one per section with observations in it.
    """
    clusters: list[Cluster] = []
    assigned = np.zeros(len(times), dtype=bool)

    for name, t_start, t_end in section_bounds:
        mask = (times >= t_start) & (times <= t_end) & ~assigned
        indices = np.where(mask)[0]
        if len(indices) > 0:
            assigned[indices] = True
            clusters.append(
                Cluster(
                    indices=indices,
                    cluster_id=f"{set_id}_{name}" if set_id else name,
                    level="section",
                    layer=layer,
                    layer_name="section",
                    metadata={
                        "t_start": t_start,
                        "t_end": t_end,
                        "n_obs": len(indices),
                        "section_name": name,
                    },
                )
            )

    return clusters


def partition_by_column(
    values: np.ndarray,
    set_id: str = "",
    layer: int = 0,
    column: str = "",
) -> list[Cluster]:
    """Split observations into one cluster per distinct value of a frame column.

    This is the "cluster by whatever the frame already knows" level: observing
    night, instrument, data source, reduction type, or any other categorical
    column.  Unlike the time-based partitioners it needs no thresholds, so it is
    the natural way to group observations that a source already labelled.

    Parameters
    ----------
    values : np.ndarray
        One value per observation, in set order (e.g. ``view.column("data_source")``).
        Missing values (``None``/``NaN``) form their own cluster.
    set_id : str
        Identifier for the set, used in cluster ids.
    layer : int
        Partitioning layer index for the produced clusters.
    column : str
        Name of the column the values came from, recorded in the metadata and
        used in the cluster ids.

    Returns
    -------
    list[Cluster]
        One cluster per distinct value, ordered by first appearance so the
        result is stable and readable.
    """
    n = len(values)
    if n == 0:
        return []

    # Group positions by value, preserving first-appearance order.  ``repr`` is
    # used as the dict key so unhashable/NaN values still group correctly.
    order: list[str] = []
    buckets: dict[str, list[int]] = {}
    labels: dict[str, Any] = {}
    for i, value in enumerate(values):
        missing = value is None or (isinstance(value, float) and np.isnan(value))
        key = "<missing>" if missing else repr(value)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
            labels[key] = None if missing else value
        buckets[key].append(i)

    clusters: list[Cluster] = []
    for position, key in enumerate(order):
        indices = np.asarray(buckets[key], dtype=int)
        label = labels[key]
        slug = "missing" if label is None else _slugify(label)
        clusters.append(
            Cluster(
                indices=indices,
                cluster_id=f"{set_id}_{column}_{slug}" if set_id else f"{column}_{slug}",
                level="column",
                layer=layer,
                layer_name="column",
                metadata={
                    "n_obs": len(indices),
                    "column": column,
                    "value": label,
                    "column_value": label,
                    "position": position,
                },
            )
        )
    return clusters


def _slugify(value: Any) -> str:
    """Turn a column value into a filesystem- and log-friendly identifier."""
    text = str(value).strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in slug.split("_") if part) or "empty"


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
