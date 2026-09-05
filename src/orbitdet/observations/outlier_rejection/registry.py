"""Registry for outlier rejection strategies.

Follows the same decorator-based pattern as
:mod:`orbitdet.observations.registry` for dataset factories.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .base import OutlierStrategy

logger = logging.getLogger(__name__)

# Global registry: maps strategy type string to factory callable
_STRATEGY_REGISTRY: dict[str, type[OutlierStrategy]] = {}


def register_outlier_strategy(strategy_type: str) -> Callable:
    """Decorator to register an outlier strategy class.

    Parameters
    ----------
    strategy_type : str
        The strategy type identifier (e.g. ``'residual_threshold'``,
        ``'epoch_filter'``).

    Returns
    -------
    Callable
        Decorator that registers the wrapped class.

    Raises
    ------
    ValueError
        If ``strategy_type`` is already registered.

    Example
    -------
    .. code-block:: python

        @register_outlier_strategy("residual_threshold")
        class ResidualThresholdOutlier(OutlierStrategy):
            ...
    """

    def decorator(cls: type[OutlierStrategy]) -> type[OutlierStrategy]:
        if strategy_type in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Outlier strategy '{strategy_type}' is already registered. "
                f"Duplicate registration of {cls.__module__}.{cls.__qualname__} "
                f"conflicts with existing registration."
            )
        if not issubclass(cls, OutlierStrategy):
            raise TypeError(
                f"Class {cls.__name__} must inherit from OutlierStrategy."
            )
        _STRATEGY_REGISTRY[strategy_type] = cls
        logger.debug("Registered outlier strategy '%s' -> %s", strategy_type, cls.__qualname__)
        return cls

    return decorator


def get_strategy_class(strategy_type: str) -> type[OutlierStrategy]:
    """Retrieve a registered strategy class by type identifier.

    Parameters
    ----------
    strategy_type : str
        The strategy type identifier.

    Returns
    -------
    type[OutlierStrategy]
        The registered strategy class.

    Raises
    ------
    ValueError
        If ``strategy_type`` is not registered.
    """
    if strategy_type not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"No outlier strategy registered for type '{strategy_type}'. "
            f"Available types: {available}"
        )
    return _STRATEGY_REGISTRY[strategy_type]


def list_registered_strategies() -> list[str]:
    """Return a sorted list of all registered strategy type identifiers.

    Returns
    -------
    list[str]
        Sorted list of registered strategy type strings.
    """
    return sorted(_STRATEGY_REGISTRY.keys())