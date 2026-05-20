"""Observation dataset factory system with registry-based dispatch."""

# Import all factory modules to trigger automatic registration
from . import simulated
from . import absolute_ccd_nsdb
from .collection import create_observation_collection
from .configs import (
    ObservationDatasetConfig,
    SimulatedObservationConfig,
)
from .factory import create_observation_dataset
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
]
