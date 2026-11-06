"""Residual summary tables built from a derived observation frame.

These are the table-layer counterparts of the descriptive reductions the
analysis work needs: residual behaviour grouped by observatory, source, catalog
or coordinate kind, expressed as a table that can be read at a glance or diffed
between runs.
"""

from __future__ import annotations

import pandas as pd

from orbitdet.analysis.access import ObservationAccess
from orbitdet.analysis.statistics import aggregate
from orbitdet.analysis.tables.base import Table

__all__ = ["ResidualSummary", "SourceResidualSummary", "ObservatoryResidualSummary"]

#: Groupings that are meaningful for Triton astrometry, tried in order.
_SUMMARY_GROUPINGS: tuple[tuple[str, list[str]], ...] = (
    ("by_source", ["source_set_id"]),
    ("by_observatory", ["observatory_code"]),
    ("by_catalog", ["original_reduction"]),
    ("by_coordinate_kind", ["coordinate_kind"]),
)


class ResidualSummary(Table):
    """One residual summary table per meaningful clustering of the frame.

    Parameters
    ----------
    access : ObservationAccess
        The access layer whose derived frame is summarised.
    groupings : tuple[tuple[str, list[str]], ...] | None
        Named groupings to summarise.  Defaults to
        :data:`_SUMMARY_GROUPINGS` (source, observatory, catalog, coordinate
        kind), skipping any clustering whose columns are absent from the frame.
    to_unit : str
        Reporting unit for the residual columns (``"arcsec"`` by default).
    """

    def __init__(
        self,
        access: ObservationAccess,
        groupings: tuple[tuple[str, list[str]], ...] | None = None,
        to_unit: str = "arcsec",
    ):
        super().__init__()
        self.access = access
        self.groupings = groupings or _SUMMARY_GROUPINGS
        self.to_unit = to_unit

    def _build(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        frame = self.access.frame
        tables: dict[str, pd.DataFrame] = {}
        for table_name, columns in self.groupings:
            if not all(column in frame.columns for column in columns):
                continue
            summary = aggregate(frame, by=columns, to_unit=self.to_unit)
            if not summary.empty:
                tables[f"{self.name}_{table_name}"] = summary
        if not tables:
            tables[self.name] = aggregate(frame, by=None, to_unit=self.to_unit)
        return tables


class SourceResidualSummary(Table):
    """One residual summary table grouped by academic source.

    Parameters
    ----------
    access : ObservationAccess
        The access layer whose derived frame is summarised.
    to_unit : str
        Reporting unit for the residual columns (``"arcsec"`` by default).
    """

    def __init__(self, access: ObservationAccess, to_unit: str = "arcsec"):
        super().__init__()
        self.access = access
        self.to_unit = to_unit

    def _build(self) -> pd.DataFrame:
        return aggregate(self.access.frame, by=["source_set_id"], to_unit=self.to_unit)


class ObservatoryResidualSummary(Table):
    """One residual summary table grouped by observatory.

    Parameters
    ----------
    access : ObservationAccess
        The access layer whose derived frame is summarised.
    to_unit : str
        Reporting unit for the residual columns (``"arcsec"`` by default).
    """

    def __init__(self, access: ObservationAccess, to_unit: str = "arcsec"):
        super().__init__()
        self.access = access
        self.to_unit = to_unit

    def _build(self) -> pd.DataFrame:
        return aggregate(self.access.frame, by=["observatory_code"], to_unit=self.to_unit)
