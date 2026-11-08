"""Per-group weight and residual summary tables.

This is the table-layer replacement for the old
:class:`orbitdet.visualization.WeightSummaryTable`, which read a weights CSV
that carried its own copy of the dataset metadata.  The access layer now owns
that provenance, so the table reads the derived frame instead and only borrows
the weights report for the one thing the frame does not carry: the cluster
assignment.

The default output is the same layout as before — one row per source, with the
observation count and the first/second coordinate RMS side by side::

    +----------------------------+------+------+-------------+-----------+------+
    | Source                     | Obs  | N    | RA type     | RA RMSE   | N    |
    |                            |      |      | DEC type    | DEC RMSE  |      |
    +----------------------------+------+------+-------------+-----------+------+
    | Veiga et al. (1996)        | 874  | 51   | $\\alpha$   | 0.0800    | 51   |
    |                            |      |      | $\\delta$   | 0.0460    |      |

What is configurable
--------------------
``by``
    The grouping columns, as any column names the derived frame carries.  The
    default is ``("source_set_id",)``; ``("source_set_id", "cluster_id")``
    reproduces the old per-cluster table, and ``("observatory_code",)`` or
    ``("original_reduction", "coordinate_kind")`` are equally valid.
``columns``
    Extra frame columns to carry through as additional table columns.  They are
    read per group, so they should be constant within a group (a varying column
    is reported with its distinct values joined by ``", "`` and a warning).
``to_unit``
    Reporting unit for the RMS columns (``"arcsec"`` by default).

Example
-------
::

    from orbitdet.analysis import ObservationAccess
    from orbitdet.analysis.tables import WeightSummary

    access = ObservationAccess.build(collection, standard_frame)
    # ... outlier rejection and weighting ...
    access.refresh()

    table = WeightSummary(
        access,
        weights=weights_df,
        by=("source_set_id", "cluster_id"),
        columns=("original_reduction", "frame_center"),
    )
    frame = table.build()          # publishes CSV + Markdown
    print(table.to_string(frame))
    print(table.to_latex(frame))
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from orbitdet.analysis.access import SET_ID_COLUMN, ObservationAccess
from orbitdet.analysis.tables.base import Table

logger = logging.getLogger(__name__)

__all__ = ["WeightSummary", "coordinate_axis_labels"]

#: Columns the weights report contributes that the derived frame does not carry.
#: They are joined onto the frame by ``(set_id, epoch)`` when a weights report is
#: supplied, so a table can group by cluster.
_WEIGHTS_JOIN_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "parent_id",
    "parent_level",
    "cluster_level",
    "strategy",
)

#: Reporting units and the factor that converts radians into them.
_UNIT_FACTOR: dict[str, float] = {
    "rad": 1.0,
    "arcsec": np.rad2deg(1.0) * 3600.0,
    "mas": np.rad2deg(1.0) * 3600.0 * 1e3,
}

#: Default grouping: one row per academic source.
_DEFAULT_BY: tuple[str, ...] = ("source_set_id",)


def coordinate_axis_labels(coordinate_kind: str) -> tuple[str, str]:
    """Map a ``coordinate_kind`` value to human-readable axis labels.

    The frame records *what* the two components mean in ``coordinate_kind``, so
    the table can label a position-angle dataset ``(rho, theta)`` rather than
    calling everything RA/Dec.  The labels are LaTeX so they render in the
    LaTeX output and read acceptably in plain text.

    Parameters
    ----------
    coordinate_kind : str
        The standard frame's ``coordinate_kind`` value (``"ra_dec"``,
        ``"position_angle_separation"``, ``"cartesian_xy"``, ...).

    Returns
    -------
    tuple[str, str]
        ``(first_label, second_label)``, empty strings when the kind is unknown.
    """
    kind = str(coordinate_kind).lower()
    if "position_angle" in kind:
        return (r"$\rho$", r"$\theta$")
    if "cartesian" in kind:
        return ("$x$", "$y$")
    if "ra_dec" in kind:
        return (r"$\alpha$", r"$\delta$")
    return ("", "")


class WeightSummary(Table):
    """Summarise observations per group with their residual RMS.

    Parameters
    ----------
    access : ObservationAccess
        The access layer whose derived frame is summarised.  Only accepted rows
        are used, so the table describes the fit's actual input set.
    weights : pd.DataFrame, optional
        The per-observation weights report returned by
        :meth:`orbitdet.observations.weighting.WeightEngine.apply`.  When
        supplied, its cluster columns are joined onto the frame by
        ``(set_id, epoch)``, which is what makes ``cluster_id`` available as a
        grouping or an extra column.  When omitted, only frame columns can be
        used.
    by : Sequence[str]
        Grouping columns.  Defaults to ``("source_set_id",)``.
    columns : Sequence[str], optional
        Extra columns to carry through, read per group.
    to_unit : str
        Reporting unit for the RMS columns: ``"arcsec"`` (default), ``"mas"``
        or ``"rad"``.
    name : str, optional
        Base name for the published files.  Defaults to ``weight_summary``.

    Raises
    ------
    ValueError
        If ``to_unit`` is unknown, or if a grouping column is absent from the
        frame (and the weights report).
    """

    def __init__(
        self,
        access: ObservationAccess | None,
        weights: pd.DataFrame | None = None,
        by: Sequence[str] = _DEFAULT_BY,
        columns: Sequence[str] | None = None,
        to_unit: str = "arcsec",
        name: str | None = None,
    ):
        super().__init__(name or "weight_summary")
        if to_unit not in _UNIT_FACTOR:
            raise ValueError(f"Unknown unit '{to_unit}'; expected one of {sorted(_UNIT_FACTOR)}.")
        self.access = access
        self.weights = weights
        self.by = tuple(by)
        self.columns = tuple(columns or ())
        self.to_unit = to_unit

        #: Column kinds of the most recently built table, used by the renderers
        #: to decide between text and fixed-point formatting.
        self.column_kinds: dict[str, str] = {}

    @classmethod
    def from_weights(
        cls,
        weights: pd.DataFrame,
        by: Sequence[str] = ("source_name",),
        columns: Sequence[str] | None = None,
        to_unit: str = "arcsec",
        name: str | None = None,
    ) -> WeightSummary:
        """Build a summary from a weights report alone, without an access layer.

        The engine enriches its report with the set's provenance
        (``source_name``, ``dataset_id``, ``observatory_code``, ``ra_type``,
        ``dec_type``), so a report read back from ``observation_weights.csv``
        can be summarised on its own.  This is what the standalone
        ``plot_weight_groups`` script uses; inside a run, prefer the access-layer
        constructor, which reads the live frame.

        Parameters
        ----------
        weights : pd.DataFrame
            A weights report, as returned by
            :meth:`orbitdet.observations.weighting.WeightEngine.apply` or read
            from ``observation_weights.csv``.
        by : Sequence[str]
            Grouping columns.  Defaults to ``("source_name",)``.
        columns : Sequence[str], optional
            Extra report columns to carry through.
        to_unit : str
            Reporting unit for the RMS columns.
        name : str, optional
            Base name for the published files.
        """
        summary = cls(None, weights=weights, by=by, columns=columns, to_unit=to_unit, name=name)
        summary._from_report = True
        return summary

    # -- Building -------------------------------------------------------

    def _build(self) -> pd.DataFrame:
        """Build the summary table, one row per group."""
        frame = self._joined_frame()
        if frame.empty:
            self.column_kinds = {}
            return pd.DataFrame()

        missing = [column for column in self.by if column not in frame.columns]
        if missing:
            raise ValueError(
                f"WeightSummary: grouping column(s) {missing} are not in the frame. "
                f"Available: {sorted(frame.columns)}."
            )

        factor = _UNIT_FACTOR[self.to_unit]
        keys = list(self.by)
        records: list[dict] = []

        for key, group in frame.groupby(keys, dropna=False, sort=True):
            key_values = key if isinstance(key, tuple) else (key,)
            record: dict = dict(zip(keys, key_values, strict=True))

            record["n_obs"] = int(len(group))
            record["ra_type"], record["dec_type"] = self._axis_labels(group)
            record["ra_rms_arcsec"] = _rms(group["residual_first_rad"].to_numpy("float64")) * factor
            record["dec_rms_arcsec"] = (
                _rms(group["residual_second_rad"].to_numpy("float64")) * factor
            )

            for column in self.columns:
                if column not in group.columns:
                    logger.warning(
                        "WeightSummary: extra column '%s' is not in the frame; skipping.",
                        column,
                    )
                    continue
                record[column] = _constant_value(group[column], column)

            records.append(record)

        result = pd.DataFrame(records)
        self.column_kinds = self._classify_columns(result)
        return result

    def _joined_frame(self) -> pd.DataFrame:
        """Return the frame to summarise, with the weights report's columns joined on.

        In access-layer mode the frame is the accepted rows that carry a
        residual, and the report is joined on ``(set_id, epoch)``.  Rows without
        a residual are dropped: a row that never joined a set (or that fed two
        sets, so only the first names it) is not *rejected*, it simply carries no
        measurement, and including it would add an empty group with a NaN RMS.

        In report-only mode (:meth:`from_weights`) the report *is* the frame, so
        it is returned as-is with its columns renamed to the frame's names.
        """
        if getattr(self, "_from_report", False):
            return self._report_as_frame()

        frame = self.access.accepted()
        frame = frame.loc[frame["residual_first_rad"].notna()].copy()
        if frame.empty or self.weights is None or self.weights.empty:
            return frame

        available = [c for c in _WEIGHTS_JOIN_COLUMNS if c in self.weights.columns]
        if not available:
            return frame

        report = self.weights[["set_id", "time", *available]].copy()
        report = report.rename(columns={"set_id": SET_ID_COLUMN, "time": "epoch_tdb"})

        before = len(frame)
        merged = frame.merge(
            report,
            on=[SET_ID_COLUMN, "epoch_tdb"],
            how="left",
            suffixes=("", "_weights"),
        )
        if len(merged) != before:
            logger.warning(
                "WeightSummary: joining the weights report changed the row count "
                "(%d -> %d); the report has duplicate (set_id, epoch) pairs.",
                before,
                len(merged),
            )
        return merged

    def _report_as_frame(self) -> pd.DataFrame:
        """Return the weights report with its columns renamed to the frame's names.

        The report stores residuals in arcseconds and calls the source
        ``source_name``; the frame stores radians and calls it ``source_set_id``.
        Renaming here keeps one code path for the grouping and statistics.
        """
        report = self.weights
        if report is None or report.empty:
            return pd.DataFrame()

        frame = report.copy()
        if "ra_residual_arcsec" in frame.columns:
            frame["residual_first_rad"] = np.deg2rad(frame["ra_residual_arcsec"] / 3600.0)
        if "dec_residual_arcsec" in frame.columns:
            frame["residual_second_rad"] = np.deg2rad(frame["dec_residual_arcsec"] / 3600.0)
        if "source_name" in frame.columns and "source_set_id" not in frame.columns:
            frame["source_set_id"] = frame["source_name"]

        if "residual_first_rad" not in frame.columns:
            logger.warning(
                "WeightSummary: the weights report has no residual columns; the "
                "RMS columns will be empty."
            )
            frame["residual_first_rad"] = np.nan
            frame["residual_second_rad"] = np.nan
        return frame

    def _axis_labels(self, group: pd.DataFrame) -> tuple[str, str]:
        """Return the coordinate axis labels for a group.

        The frame records the semantic kind in ``coordinate_kind``; a weights
        report read back from CSV carries the labels the engine already derived
        (``ra_type`` / ``dec_type``).  Prefer the explicit labels when present so
        a report-only summary keeps them.
        """
        if "ra_type" in group.columns and "dec_type" in group.columns:
            ra_type = _constant_value(group["ra_type"], "ra_type")
            dec_type = _constant_value(group["dec_type"], "dec_type")
            if ra_type or dec_type:
                return (str(ra_type), str(dec_type))

        if "coordinate_kind" not in group.columns:
            return ("", "")
        kinds = group["coordinate_kind"].dropna().unique()
        if len(kinds) == 0:
            return ("", "")
        if len(kinds) > 1:
            logger.warning(
                "WeightSummary: group spans several coordinate kinds %s; labelling with the first.",
                sorted(str(k) for k in kinds),
            )
        return coordinate_axis_labels(str(kinds[0]))

    def _classify_columns(self, table: pd.DataFrame) -> dict[str, str]:
        """Tag each column as ``text``, ``count`` or ``angle`` for rendering."""
        kinds: dict[str, str] = {}
        for column in table.columns:
            if column == "n_obs":
                kinds[column] = "count"
            elif column in ("ra_rms_arcsec", "dec_rms_arcsec"):
                kinds[column] = "angle"
            elif pd.api.types.is_numeric_dtype(table[column]):
                kinds[column] = "angle"
            else:
                kinds[column] = "text"
        return kinds

    # -- Rendering ------------------------------------------------------

    def to_string(self, table: pd.DataFrame | None = None) -> str:
        """Render the table as an aligned plain-text block.

        Parameters
        ----------
        table : pd.DataFrame, optional
            A table from :meth:`build`.  Rebuilt when omitted.
        """
        if table is None:
            table = self._build()
        if table.empty:
            return "No weight data."

        headers = [self._header(column) for column in table.columns]
        rows = [
            [self._cell(row[column], column) for column in table.columns]
            for _, row in table.iterrows()
        ]

        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
            for i in range(len(headers))
        ]

        rule = "=" * (sum(widths) + 2 * (len(widths) - 1))
        lines = [rule, "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)), rule]
        for row in rows:
            lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
        lines.append(rule)

        if "n_obs" in table.columns:
            total = int(table["n_obs"].sum())
            total_width = widths[table.columns.get_loc("n_obs")]
            lines.append(f"{'Total':<{widths[0]}}  {total:>{total_width}}")
        return "\n".join(lines)

    def to_latex(self, table: pd.DataFrame | None = None) -> str:
        """Render the table as a LaTeX ``booktabs`` table.

        Parameters
        ----------
        table : pd.DataFrame, optional
            A table from :meth:`build`.  Rebuilt when omitted.
        """
        if table is None:
            table = self._build()
        if table.empty:
            return "% No weight data."

        alignment = " ".join(
            "r" if self.column_kinds.get(column) in ("count", "angle") else "l"
            for column in table.columns
        )
        headers = " & ".join(self._header(column) for column in table.columns)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Observation summary with per-group residual RMS.}",
            r"\label{tab:weight_summary}",
            f"\\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            f"{headers} \\\\",
            r"\midrule",
        ]

        for _, row in table.iterrows():
            cells = [self._latex_cell(row[column], column) for column in table.columns]
            lines.append(" & ".join(cells) + r" \\")

        if "n_obs" in table.columns:
            total = int(table["n_obs"].sum())
            total_cells = [
                "Total" if i == 0 else (str(total) if column == "n_obs" else "")
                for i, column in enumerate(table.columns)
            ]
            lines.extend([r"\midrule", " & ".join(total_cells) + r" \\"])

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _header(self, column: str) -> str:
        """Return the display header for a column."""
        return {
            "n_obs": "N",
            "ra_type": "RA type",
            "dec_type": "DEC type",
            "ra_rms_arcsec": f"RA RMS [{self.to_unit}]",
            "dec_rms_arcsec": f"DEC RMS [{self.to_unit}]",
        }.get(column, column)

    def _cell(self, value, column: str) -> str:
        """Format one cell for the plain-text renderer."""
        kind = self.column_kinds.get(column, "text")
        if kind == "count":
            return str(int(value))
        if kind == "angle":
            return f"{float(value):.4f}" if np.isfinite(value) else "-"
        return str(value)

    def _latex_cell(self, value, column: str) -> str:
        """Format one cell for the LaTeX renderer."""
        kind = self.column_kinds.get(column, "text")
        if kind == "count":
            return str(int(value))
        if kind == "angle":
            return f"{float(value):.4f}" if np.isfinite(value) else "--"
        return str(value).replace("&", r"\&").replace("_", r"\_")


def _rms(values: np.ndarray) -> float:
    """Root-mean-square of the finite values, or NaN when there are none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _constant_value(series: pd.Series, column: str):
    """Return a group's value for *column*, collapsing any variation.

    A column that is constant within the group (the usual case for provenance
    columns) is returned as-is.  One that varies is reported as its distinct
    values joined by ``", "``, with a warning, so the table never silently picks
    one of several.
    """
    distinct = series.dropna().unique()
    if len(distinct) == 0:
        return ""
    if len(distinct) == 1:
        return distinct[0]
    logger.warning(
        "WeightSummary: column '%s' varies within a group (%d distinct values); joining them.",
        column,
        len(distinct),
    )
    return ", ".join(str(value) for value in distinct)
