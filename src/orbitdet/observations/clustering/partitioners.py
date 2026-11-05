"""Low-level partitioners for the shared observation-clustering system.

These functions split a set of observation times into atomic clusters.  They are
mechanism-only: they know nothing about weighting or outlier rejection.
"""

from __future__ import annotations

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
