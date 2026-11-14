"""Table producers built on the analysis access layer.

The :class:`Table` base mirrors :class:`orbitdet.visualization.base.Plot`: a
subclass builds a :class:`pandas.DataFrame` and the base publishes it as CSV and
Markdown, plus LaTeX when the table carries a
:class:`~orbitdet.analysis.tables.latex.LatexTableWriter`.

See :mod:`orbitdet.analysis.tables.residual_summary` for the observation tables
and :mod:`orbitdet.analysis.tables.parameter_table` for the estimation table.
"""

from .base import Table
from .config import (
    DEFAULT_STYLE,
    WRITER_STYLES,
    attach_table_config,
    build_writer,
    build_writers,
    resolve_table_builder,
)
from .goodness_of_fit import GoodnessOfFitTable, build_goodness_of_fit_frame
from .latex import AATableWriter, LatexTableWriter, escape_latex
from .parameter_table import (
    DEFAULT_PARAMETER_NAMES,
    EstimationParameterTable,
    build_parameter_frame,
)
from .residual_summary import (
    ObservatoryResidualSummary,
    ResidualSummary,
    SourceResidualSummary,
)
from .sweep_comparison import (
    COMPARISON_COLUMNS,
    SweepComparisonTable,
    build_sweep_frame,
    read_convergence,
    specification_cells,
    specification_columns,
    summarise_fit,
)
from .view import TableView
from .weight_summary import WeightSummary, coordinate_axis_labels

__all__ = [
    "Table",
    "LatexTableWriter",
    "AATableWriter",
    "escape_latex",
    "WRITER_STYLES",
    "DEFAULT_STYLE",
    "attach_table_config",
    "build_writer",
    "build_writers",
    "resolve_table_builder",
    "TableView",
    "EstimationParameterTable",
    "GoodnessOfFitTable",
    "build_goodness_of_fit_frame",
    "build_parameter_frame",
    "DEFAULT_PARAMETER_NAMES",
    "ResidualSummary",
    "SourceResidualSummary",
    "ObservatoryResidualSummary",
    "WeightSummary",
    "coordinate_axis_labels",
    "SweepComparisonTable",
    "COMPARISON_COLUMNS",
    "build_sweep_frame",
    "read_convergence",
    "specification_cells",
    "specification_columns",
    "summarise_fit",
]
