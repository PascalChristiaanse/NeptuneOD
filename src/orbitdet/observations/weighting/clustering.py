"""Backward-compatibility shim for the shared clustering system.

Grouping was moved to :mod:`orbitdet.observations.clustering` because it is
shared between the weighting and outlier-rejection subsystems.  This module
keeps the old import path working during migration; new code should import
from :mod:`orbitdet.observations.clustering` instead.
"""

from __future__ import annotations

import warnings

from ..clustering import (  # noqa: F401
    Cluster,
    ClusterList,
    SectionInfo,
    TimeframeInfo,
    build_cluster_list,
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
)
from ..clustering.partitioners import _split_by_gap  # noqa: F401

warnings.warn(
    "orbitdet.observations.weighting.clustering has moved to "
    "orbitdet.observations.clustering; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Cluster",
    "ClusterList",
    "SectionInfo",
    "TimeframeInfo",
    "build_cluster_list",
    "partition_by_set",
    "partition_by_timeframes",
    "partition_by_sections",
]
