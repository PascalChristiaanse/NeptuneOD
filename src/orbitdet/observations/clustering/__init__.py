"""Shared observation-clustering system.

Clustering partitions the observations of a single set by time and is used by
**both** the weighting and the outlier rejection subsystems.  Each subsystem
builds its own clustering from its own config - an outlier-rejection cluster is
not necessarily the same as a weighting cluster - but they share this one module
rather than each re-implementing partitioning.

Public API
----------
- :class:`Cluster`, :class:`ClusterList` - the data models.
- :func:`build_cluster_list` - turn a clustering config into a :class:`ClusterList`.
- partitioners - :func:`partition_by_set`, :func:`partition_by_timeframes`,
  :func:`partition_by_sections`.
"""

from .models import Cluster, ClusterList, SectionInfo, TimeframeInfo
from .partitioners import (
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
)
from .spec import build_cluster_list

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
