"""Registry for observation dataset factories."""

import logging
from collections.abc import Callable
from typing import TypeVar

from .configs import ObservationDatasetConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic return type for factories

# Global factory registry: maps dataset type string to factory callable
_FACTORY_REGISTRY: dict[str, Callable[[ObservationDatasetConfig], T]] = {}


def register_dataset_factory(dataset_type: str) -> Callable:
    """Decorator to register a dataset factory function.

    Args:
        dataset_type: The dataset type string (e.g., 'ground_ccd', 'space_ccd').

    Returns:
        Decorator function that registers the wrapped factory.

    Raises:
        ValueError: If dataset_type is already registered.

    Example:
        @register_dataset_factory('ground_ccd')
        def create_ground_ccd_dataset(cfg: GroundCCDConfig):
            ...
    """

    def decorator(factory_func: Callable) -> Callable:
        if dataset_type in _FACTORY_REGISTRY:
            raise ValueError(
                f"Dataset factory for type '{dataset_type}' is already registered. "
                f"Duplicate registration of {factory_func.__module__}.{factory_func.__name__} "
                f"conflicts with existing registration."
            )
        _FACTORY_REGISTRY[dataset_type] = factory_func
        logger.debug(f"Registered observation factory for type '{dataset_type}'")
        return factory_func

    return decorator


def get_factory(dataset_type: str) -> Callable:
    """Retrieve a registered factory by dataset type.

    Args:
        dataset_type: The dataset type string to look up.

    Returns:
        The registered factory callable.

    Raises:
        ValueError: If dataset_type is not registered.
    """
    if dataset_type not in _FACTORY_REGISTRY:
        available = ", ".join(sorted(_FACTORY_REGISTRY.keys()))
        raise ValueError(
            f"No factory registered for dataset type '{dataset_type}'. Available types: {available}"
        )
    return _FACTORY_REGISTRY[dataset_type]


def list_registered_types() -> list[str]:
    """Return a sorted list of all registered dataset types.

    Returns:
        List of registered dataset type strings.
    """
    return sorted(_FACTORY_REGISTRY.keys())
