"""Observation dataset factory system with registry-based dispatch."""

# Import all factory modules to trigger automatic registration

from . import (
    absolute_ccd_nsdb,
    relative_ccd_nsdb,
    relative_xy_nsdb,
    simulated,
    voyager,
)
from .collection import create_observation_collection
from .configs import (
    ObservationDatasetConfig,
    SimulatedObservationConfig,
)
from .factory import create_observation_dataset
from .helpers import get_observatory_info
from .outlier_rejection import (
    OutlierEngine,
    OutlierRejectionConfig,
    OutlierStrategy,
    OutlierStrategyConfig,
    get_strategy_class,
    list_registered_strategies,
    register_outlier_strategy,
)
from .registry import get_factory, list_registered_types, register_dataset_factory

__all__ = [
    # Config dataclasses
    "ObservationDatasetConfig",
    "SimulatedObservationConfig",
    # Registry functions
    "register_dataset_factory",
    "get_factory",
    "list_registered_types",
    # Factory functions
    "create_observation_dataset",
    "create_observation_collection",
    # Factory modules (implicitly imported for registration)
    "simulated",
    "absolute_ccd_nsdb",
    "relative_ccd_nsdb",
    "relative_xy_nsdb",
    "relative_xy_radec_nsdb",
    "voyager",
    # Helper functions
    "get_observatory_info",
    # Outlier rejection
    "OutlierStrategy",
    "OutlierEngine",
    "OutlierRejectionConfig",
    "OutlierStrategyConfig",
    "register_outlier_strategy",
    "get_strategy_class",
    "list_registered_strategies",
]

