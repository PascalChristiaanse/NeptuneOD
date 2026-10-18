"""Shared observation-grouping system.

Grouping partitions the observations of a single set by time and is used by
**both** the weighting and the outlier rejection subsystems.  Each subsystem
builds its own grouping from its own config — an outlier-rejection group is not
necessarily the same as a weighting group — but they share this one module
rather than each re-implementing partitioning.

Public API
----------
- :class:`Group`, :class:`GroupList` — the data models.
- :func:`build_group_list` — turn a grouping config into a :class:`GroupList`.
- partitioners — :func:`partition_by_set`, :func:`partition_by_timeframes`,
  :func:`partition_by_sections`.
"""

from .models import Group, GroupList, SectionInfo, TimeframeInfo
from .partitioners import (
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
)
from .spec import build_group_list

__all__ = [
    "Group",
    "GroupList",
    "SectionInfo",
    "TimeframeInfo",
    "build_group_list",
    "partition_by_set",
    "partition_by_timeframes",
    "partition_by_sections",
]
