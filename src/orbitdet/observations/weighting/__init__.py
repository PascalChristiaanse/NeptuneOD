"""Weighting subsystem for the NeptuneOD observation pipeline.

Provides pluggable strategies for computing inverse-variance weights from
observation residuals, with flexible clustering configurations (timeframes,
sections, etc.).
"""

from ..clustering import Cluster, ClusterList, build_cluster_list
from . import strategies  # noqa: F401
from .base import WeightStrategy
from .engine import WeightEngine
from .gaia_cache import (
    clear_gaia_weights,
    get_gaia_covariance_components,
    get_gaia_weights,
    store_gaia_covariance_components,
    store_gaia_weights,
)
from .registry import get_strategy_class, list_registered_strategies, register_weight_strategy

__all__ = [
    "WeightEngine",
    "WeightStrategy",
    "Cluster",
    "ClusterList",
    "build_cluster_list",
    "register_weight_strategy",
    "get_strategy_class",
    "list_registered_strategies",
    "store_gaia_weights",
    "get_gaia_weights",
    "store_gaia_covariance_components",
    "get_gaia_covariance_components",
    "clear_gaia_weights",
]
