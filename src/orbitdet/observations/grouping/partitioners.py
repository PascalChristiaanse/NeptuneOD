"""Low-level partitioners for the shared observation-grouping system.

These functions split a set of observation times into atomic groups.  They are
mechanism-only: they know nothing about weighting or outlier rejection.
"""

from __future__ import annotations

import numpy as np

from .models import Group


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
    layer : int
        Partitioning layer index for the produced groups.

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
                layer=layer,
                layer_name="timeframe",
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
    layer: int = 0,
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
    layer : int
        Partitioning layer index for the produced groups.

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

    return groups


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
