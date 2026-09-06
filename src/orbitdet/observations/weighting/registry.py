"""Registry for weighting strategies.

Follows the same decorator-based pattern as the outlier rejection subsystem.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .base import WeightStrategy

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY: dict[str, type[WeightStrategy]] = {}


def register_weight_strategy(strategy_type: str) -> Callable:
    """Decorator to register a weight strategy class.

    Parameters
    ----------
    strategy_type : str
        The strategy type identifier (e.g. ``'id_v2'``, ``'hybrid_geometric'``).

    Returns
    -------
    Callable
        Decorator that registers the wrapped class.

    Raises
    ------
    ValueError
        If ``strategy_type`` is already registered.
    TypeError
        If the class does not inherit from :class:`WeightStrategy`.
    """

    def decorator(cls: type[WeightStrategy]) -> type[WeightStrategy]:
        if strategy_type in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Weight strategy '{strategy_type}' is already registered. "
                f"Duplicate registration of {cls.__module__}.{cls.__qualname__} "
                f"conflicts with existing registration."
            )
        if not issubclass(cls, WeightStrategy):
            raise TypeError(f"Class {cls.__name__} must inherit from WeightStrategy.")
        _STRATEGY_REGISTRY[strategy_type] = cls
        logger.debug("Registered weight strategy '%s' -> %s", strategy_type, cls.__qualname__)
        return cls

    return decorator


def get_strategy_class(strategy_type: str) -> type[WeightStrategy]:
    """Retrieve a registered strategy class by type identifier.

    Parameters
    ----------
    strategy_type : str
        The strategy type identifier.

    Returns
    -------
    type[WeightStrategy]
        The registered strategy class.

    Raises
    ------
    ValueError
        If ``strategy_type`` is not registered.
    """
    if strategy_type not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"No weight strategy registered for type '{strategy_type}'. "
            f"Available types: {available}"
        )
    return _STRATEGY_REGISTRY[strategy_type]


def list_registered_strategies() -> list[str]:
    """Return a sorted list of all registered strategy type identifiers."""
    return sorted(_STRATEGY_REGISTRY.keys())