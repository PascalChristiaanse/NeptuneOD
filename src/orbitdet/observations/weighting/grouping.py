"""Backward-compatibility shim for the shared grouping system.

Grouping was moved to :mod:`orbitdet.observations.grouping` because it is
shared between the weighting and outlier-rejection subsystems.  This module
keeps the old import path working during migration; new code should import
from :mod:`orbitdet.observations.grouping` instead.
"""

from __future__ import annotations

import warnings

from ..grouping import (  # noqa: F401
    Group,
    GroupList,
    SectionInfo,
    TimeframeInfo,
    build_group_list,
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
)
from ..grouping.partitioners import _split_by_gap  # noqa: F401

warnings.warn(
    "orbitdet.observations.weighting.grouping has moved to "
    "orbitdet.observations.grouping; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

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
