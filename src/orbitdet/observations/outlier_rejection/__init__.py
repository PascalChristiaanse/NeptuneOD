"""Outlier rejection subsystem for the NeptuneOD observation pipeline.

Provides pluggable, composable strategies for removing bad observations
before estimation.  Follows the same registry-based pattern as the
observation dataset factories.
"""

from . import strategies  # noqa: F401
from .base import OutlierStrategy
from .configs import OutlierRejectionConfig, OutlierStrategyConfig
from .engine import OutlierEngine
from .registry import get_strategy_class, list_registered_strategies, register_outlier_strategy

__all__ = [
    "OutlierEngine",
    "OutlierRejectionConfig",
    "OutlierStrategy",
    "OutlierStrategyConfig",
    "get_strategy_class",
    "list_registered_strategies",
    "register_outlier_strategy",
]