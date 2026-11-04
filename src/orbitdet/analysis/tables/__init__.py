"""Table producers built on the analysis access layer.

The :class:`Table` base mirrors :class:`orbitdet.visualization.base.Plot`: a
subclass builds a :class:`pandas.DataFrame` and the base publishes it as CSV and
Markdown, records it with Aim, and logs simple metrics.

See :mod:`orbitdet.analysis.tables.residual_summary` for the concrete tables.
"""

from .base import Table
from .residual_summary import (
    ObservatoryResidualSummary,
    ResidualSummary,
    SourceResidualSummary,
)

__all__ = [
    "Table",
    "ResidualSummary",
    "SourceResidualSummary",
    "ObservatoryResidualSummary",
]
