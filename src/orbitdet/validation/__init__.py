"""Validation helpers for propagator/integrator combinations.

See :mod:`orbitdet.validation.combination` for the comparison machinery used by
``scripts/validate_combination.py``.
"""

from .combination import (
    ComparisonResult,
    PropagationOutcome,
    build_combination_settings,
    compare_outcomes,
    format_metrics,
    run_propagation,
)

__all__ = [
    "ComparisonResult",
    "PropagationOutcome",
    "build_combination_settings",
    "compare_outcomes",
    "format_metrics",
    "run_propagation",
]
