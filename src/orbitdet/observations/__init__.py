"""Observation dataset factory system with registry-based dispatch."""

# Import all factory modules to trigger automatic registration.
#
# The NSDB modules that used to live here are superseded by ``standard_frames``:
# they preprocessed as well as built models, and are kept only while the call
# sites that still hand them a raw file are migrated.  Their factory names are
# already registered by ``standard_frames`` as aliases, so importing them here
# would raise a duplicate-registration error - which is why they are no longer
# listed.  Voyager is the same story: its factory is now the shared absolute
# RA/Dec one, aliased under ``voyager``.
from . import (
    gaia,
    simulated,
    standard_frames,
)
from .clustering import (
    Cluster,
    ClusterList,
    build_cluster_list,
    partition_by_column,
    partition_by_sections,
    partition_by_set,
    partition_by_timeframes,
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
    get_set_identifier,
    register_outlier_strategy,
)
from .outlier_rejection import (
    get_strategy_class as get_outlier_strategy_class,
)
from .outlier_rejection import (
    list_registered_strategies as list_registered_outlier_strategies,
)
from .plan import Plan, PlanEntry, compile_plan
from .registry import get_factory, list_registered_types, register_dataset_factory
from .selection import Selection, compile_selection
from .weighting import (
    WeightEngine,
    WeightStrategy,
    clear_gaia_weights,
    get_gaia_covariance_components,
    get_gaia_weights,
    register_weight_strategy,
    store_gaia_covariance_components,
    store_gaia_weights,
)
from .weighting import (
    get_strategy_class as get_weight_strategy_class,
)
from .weighting import (
    list_registered_strategies as list_registered_weight_strategies,
)

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
    "standard_frames",
    "gaia",
    # Helper functions
    "get_observatory_info",
    # Selection and plans
    "Selection",
    "compile_selection",
    "Plan",
    "PlanEntry",
    "compile_plan",
    # Clustering
    "Cluster",
    "ClusterList",
    "build_cluster_list",
    "partition_by_column",
    "partition_by_sections",
    "partition_by_set",
    "partition_by_timeframes",
    # Outlier rejection
    "OutlierStrategy",
    "OutlierEngine",
    "OutlierRejectionConfig",
    "OutlierStrategyConfig",
    "get_set_identifier",
    "register_outlier_strategy",
    "get_outlier_strategy_class",
    "list_registered_outlier_strategies",
    # Weighting
    "WeightStrategy",
    "WeightEngine",
    # Grouping (shared by weighting and outlier rejection)
    "Cluster",
    "ClusterList",
    "build_cluster_list",
    "partition_by_set",
    "partition_by_timeframes",
    "partition_by_sections",
    "register_weight_strategy",
    "get_weight_strategy_class",
    "list_registered_weight_strategies",
    "store_gaia_weights",
    "get_gaia_weights",
    "store_gaia_covariance_components",
    "get_gaia_covariance_components",
    "clear_gaia_weights",
]
