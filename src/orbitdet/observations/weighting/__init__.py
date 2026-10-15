"""Weighting subsystem for the NeptuneOD observation pipeline.

Provides pluggable strategies for computing inverse-variance weights from
observation residuals, with flexible grouping configurations (timeframes,
sections, etc.).
"""

from . import strategies  # noqa: F401
from .base import WeightStrategy
from .configs import WeightingConfig
from .engine import WeightEngine
from .gaia_cache import clear_gaia_weights, get_gaia_weights, store_gaia_weights
from ..grouping import Group, GroupList, build_group_list
from .registry import get_strategy_class, list_registered_strategies, register_weight_strategy

__all__ = [
    "WeightEngine",
    "WeightStrategy",
    "WeightingConfig",
    "Group",
    "GroupList",
    "build_group_list",
    "register_weight_strategy",
    "get_strategy_class",
    "list_registered_strategies",
    "store_gaia_weights",
    "get_gaia_weights",
    "clear_gaia_weights",
]
