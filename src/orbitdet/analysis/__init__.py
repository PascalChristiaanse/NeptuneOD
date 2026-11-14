"""Analysis layer for the NeptuneOD observation pipeline.

The pipeline already produces two per-observation dataframes: the standard
observation frame (:mod:`orbitdet.preprocessing.schema`), which is the canonical
*ingest* contract, and the per-observation weights frame
(:mod:`orbitdet.observations.weighting.dataframe`), which carries residuals,
weights and clustering.  This package adds the missing third piece: an **access
layer** that joins the standard frame to the live
:class:`~tudatpy.estimation.observations.SingleObservationSet` objects through
the ``observation_set_index`` foreign key, a set of pure **view** functions that
convert the canonical frame to other representations on demand, **statistics**
that reduce it to tidy tables, and a **table** layer that publishes those tables
alongside the run's figures.

Design rules
------------
* The derived frame keeps the **canonical** representation (radians, ``first`` /
  ``second`` components).  Conversions are views - pure functions returning
  copies - and are never written back.
* The standard frame is never mutated; only the access layer's derived frame
  gains columns.
* Weights are read, never recomputed.  :meth:`ObservationAccess.refresh`
  re-reads residuals and weights from the collection after estimation.

Example
-------
::

    from orbitdet.analysis import ObservationAccess, aggregate

    access = ObservationAccess.build(collection, standard_frame)
    # ... run the estimation, which mutates the collection ...
    access.refresh()

    summary = aggregate(access.frame, by="observatory_code")
"""

from .access import (
    JOINED_COLUMNS,
    OBSERVABLE_TYPE_COLUMN,
    REJECTED_COLUMN,
    SET_ID_COLUMN,
    SET_INDEX_COLUMN,
    ObservationAccess,
    SetKey,
)
from .basis import BasisTransform, block_diagonal_basis, identity_basis, scan_basis
from .clustering import cluster_by_gap, split_by_gap
from .covariance import block_sigmas, condition_index, marginal_sigmas, principal_axes
from .distribution import (
    density_histogram,
    find_psd_peaks,
    lombscargle_psd,
    normality_test,
    qq_points,
    standardize,
    welch_psd,
)
from .frames import analysis_frame
from .set_view import SetView
from .statistics import (
    aggregate,
    chi2,
    chi2_from_matrix,
    mahalanobis,
    maximum,
    mean,
    median,
    minimum,
    normalized_residuals,
    pooled_within_group_rms,
    quadrature,
    reduced_chi2,
    rms,
    sample_std,
    weighted_rms,
)
from .tables import (
    DEFAULT_PARAMETER_NAMES,
    WRITER_STYLES,
    AATableWriter,
    EstimationParameterTable,
    GoodnessOfFitTable,
    LatexTableWriter,
    ObservatoryResidualSummary,
    ResidualSummary,
    SourceResidualSummary,
    Table,
    TableView,
    WeightSummary,
    attach_table_config,
    build_goodness_of_fit_frame,
    build_parameter_frame,
    build_writer,
    build_writers,
    coordinate_axis_labels,
    escape_latex,
    resolve_table_builder,
)
from .views import (
    ARCSEC_PER_RADIAN,
    rotate,
    to_arcsec,
    to_mas,
    with_scan_frame,
)
from .whitening import Whitening

__all__ = [
    # Access layer
    "ObservationAccess",
    "JOINED_COLUMNS",
    "SET_INDEX_COLUMN",
    "SET_ID_COLUMN",
    "OBSERVABLE_TYPE_COLUMN",
    "REJECTED_COLUMN",
    "SetKey",
    # Basis
    "BasisTransform",
    "scan_basis",
    "identity_basis",
    "block_diagonal_basis",
    # Frames
    "analysis_frame",
    # Per-set view
    "SetView",
    # Clustering
    "cluster_by_gap",
    "split_by_gap",
    # Covariance
    "marginal_sigmas",
    "block_sigmas",
    "principal_axes",
    "condition_index",
    # Distribution
    "standardize",
    "qq_points",
    "density_histogram",
    "normality_test",
    "welch_psd",
    "lombscargle_psd",
    "find_psd_peaks",
    # Views
    "to_arcsec",
    "to_mas",
    "rotate",
    "with_scan_frame",
    "ARCSEC_PER_RADIAN",
    # Statistics
    "rms",
    "weighted_rms",
    "mean",
    "sample_std",
    "median",
    "minimum",
    "maximum",
    "quadrature",
    "pooled_within_group_rms",
    "normalized_residuals",
    "chi2",
    "chi2_from_matrix",
    "mahalanobis",
    "reduced_chi2",
    "aggregate",
    # Whitening
    "Whitening",
    # Tables
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
    "build_parameter_frame",
    "build_goodness_of_fit_frame",
    "DEFAULT_PARAMETER_NAMES",
    "ResidualSummary",
    "SourceResidualSummary",
    "ObservatoryResidualSummary",
    "WeightSummary",
    "coordinate_axis_labels",
]
