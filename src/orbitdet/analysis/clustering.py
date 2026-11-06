"""Cluster observations in time by epoch gaps.

Several parts of the pipeline need to decide which observations belong to the
same *night* or *transit* - the weighting subsystem (``clustering.partitioners``),
the Gaia transit-error figure, and any future analysis clustering.  They all mean
the same thing by it: a new cluster starts wherever consecutive epochs differ by
more than a threshold.  This module is that one rule, in one place, so the
weighting clustering and the analysis figures can never disagree about where a
night ends.

It is pure: it takes epochs (already sorted or not) and returns integer labels.
Nothing reads config or touches the file system.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cluster_by_gap", "split_by_gap"]

#: A day in seconds - the natural unit for a "night" gap.
SECONDS_PER_DAY: float = 86400.0


def split_by_gap(
    epochs,
    gap_seconds: float,
    *,
    min_size: int = 1,
) -> list[np.ndarray]:
    """Split sorted epochs into clusters separated by more than ``gap_seconds``.

    A new cluster starts wherever the gap between consecutive (sorted) epochs
    exceeds ``gap_seconds``, provided the current cluster already holds at least
    ``min_size`` members.  The ``min_size`` guard lets a burst of closely spaced
    observations absorb a short internal gap without splitting, which is what a
    transit needs when its sampling is uneven.

    Parameters
    ----------
    epochs : array-like
        Observation epochs (any monotonic unit - typically seconds since J2000).
    gap_seconds : float
        Epoch gap above which a new cluster starts, in the same unit as ``epochs``.
    min_size : int
        Minimum members before a gap is allowed to start a new cluster.  Defaults
        to ``1`` (any gap splits).

    Returns
    -------
    list[np.ndarray]
        One array of positional indices (into the **sorted** ``epochs``) per
        cluster, in time order.  Empty when ``epochs`` is empty.

    Raises
    ------
    ValueError
        If ``gap_seconds`` is negative or ``min_size`` is less than one.
    """
    if gap_seconds < 0.0:
        raise ValueError(f"gap_seconds must be non-negative, got {gap_seconds}.")
    if min_size < 1:
        raise ValueError(f"min_size must be at least one, got {min_size}.")

    epochs = np.asarray(epochs, dtype="float64").ravel()
    if epochs.size == 0:
        return []

    clusters: list[list[int]] = [[0]]
    for i in range(1, epochs.size):
        gap = epochs[i] - epochs[i - 1]
        if gap > gap_seconds and len(clusters[-1]) >= min_size:
            clusters.append([i])
        else:
            clusters[-1].append(i)
    return [np.asarray(cluster, dtype=int) for cluster in clusters]


def cluster_by_gap(
    epochs,
    gap_seconds: float,
    *,
    min_size: int = 1,
) -> np.ndarray:
    """Label each observation with the index of the gap-cluster it belongs to.

    The label is ``0, 1, 2, ...`` in time order.  ``epochs`` may be unsorted;
    the labels are assigned on the sorted order and then mapped back to the
    input positions, so ``labels[i]`` corresponds to ``epochs[i]`` as given.

    This is the labelling form of :func:`split_by_gap` (which returns the clusters
    themselves); use it when a clustering column is needed rather than the clusters.

    Parameters
    ----------
    epochs : array-like
        Observation epochs (any monotonic unit).
    gap_seconds : float
        Epoch gap above which a new cluster starts, in the same unit as ``epochs``.
    min_size : int
        Minimum members before a gap is allowed to start a new cluster.

    Returns
    -------
    np.ndarray
        ``(n,)`` integer array of cluster labels (``0`` for a single cluster,
        empty when ``epochs`` is empty).

    Raises
    ------
    ValueError
        If ``epochs`` contains a non-finite value.
    """
    epochs = np.asarray(epochs, dtype="float64").ravel()
    if epochs.size == 0:
        return np.zeros(0, dtype=int)
    if not np.all(np.isfinite(epochs)):
        raise ValueError("epochs contains a non-finite value; drop or fill it first.")

    order = np.argsort(epochs, kind="stable")
    clusters = split_by_gap(epochs[order], gap_seconds, min_size=min_size)

    labels_sorted = np.empty(epochs.size, dtype=int)
    for label, indices in enumerate(clusters):
        labels_sorted[indices] = label

    labels = np.empty(epochs.size, dtype=int)
    labels[order] = labels_sorted
    return labels
