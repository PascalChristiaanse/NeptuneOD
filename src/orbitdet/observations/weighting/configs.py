"""Configuration dataclasses for the weighting subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WeightingConfig:
    """Top-level configuration for the weighting subsystem.

    Attributes
    ----------
    enabled : bool
        Whether weighting is enabled.
    strategy : str
        Weight strategy type (e.g. ``'id_v2'``, ``'hybrid_geometric'``).
    min_sigma_arcsec : float
        Floor on the RMSE in arcseconds.
    grouping : dict | None
        Grouping configuration with ``levels`` list.
    """

    enabled: bool = False
    strategy: str = "id_v2"
    min_sigma_arcsec: float = 0.01
    grouping: dict[str, Any] | None = None