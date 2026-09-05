"""Configuration dataclasses for the outlier rejection subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutlierStrategyConfig:
    """Configuration for a single outlier rejection strategy.

    Attributes
    ----------
    type : str
        The strategy type identifier (e.g. ``'residual_threshold'``,
        ``'epoch_filter'``).
    kwargs : dict
        Additional keyword arguments passed to the strategy constructor.
    """

    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutlierRejectionConfig:
    """Top-level configuration for the outlier rejection subsystem.

    Attributes
    ----------
    enabled : bool
        Whether outlier rejection is enabled.  When ``False``, the engine
        is not invoked and the collection passes through unchanged.
    strategies : tuple[OutlierStrategyConfig, ...]
        Ordered sequence of strategies to apply.  Ignored when ``enabled``
        is ``False``.
    """

    enabled: bool = False
    strategies: tuple[OutlierStrategyConfig, ...] = ()