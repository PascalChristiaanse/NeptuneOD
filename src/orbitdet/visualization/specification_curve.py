"""Specification-curve view of a parameter sweep.

A sweep over binary toggles ("is this parameter estimated?") produces one fit per
combination.  The result is high-dimensional in the *inputs* but each input has
only two values, so the usual scatter or parallel-coordinates plot says nothing:
there is no continuous axis to spread the specifications along.

The specification curve (Simonsohn, Simmons & Nelson 2020) solves that by
splitting the figure into two vertically aligned panels that share the
specification axis:

* **top** — the outcome of each fit, one point per specification, sorted so the
  curve is monotone.  A dashed line marks the median.
* **bottom** — the truth table, drawn as a grid: one *row per parameter*, one
  column per specification, a filled square where that parameter was estimated
  and a pale square where it was held fixed.

The alignment is the whole argument: reading down from "what came out" to "which
parameters were estimated" makes the effect of each toggle visible as a step in
the sorted outcome panel.

This is the layout popularised by ``specification_curve`` (aeturrell): the
outcome on top, the choice matrix underneath, sharing one x axis.  Seaborn has
no table primitive, and neither does that library — the matrix is drawn with
:func:`matplotlib.axes.Axes.scatter` on an integer grid, which is both simpler
and more controllable than a table artist.

Non-converged fits are kept rather than dropped.  With a sweep where most
combinations fail, *which* combinations fail is itself a result, so they are
drawn with a distinct marker and excluded from the median.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbitdet.visualization.base import Plot, _cfg_get
from orbitdet.visualization.palettes import palette

__all__ = ["SpecificationCurve"]

#: Colour of a truth-table cell where the parameter was *not* estimated.  Pale
#: enough to read as "off" without competing with the filled cells.
_CELL_OFF = "#E6E6E6"

#: Height in inches of one truth-table row.
_ROW_HEIGHT = 0.34


def _output_name(cfg) -> str | None:
    """Derive the output stem from the config's ``output_file``, if present."""
    output_file = _cfg_get(cfg, "output_file", default=None)
    if not output_file:
        return None
    return Path(str(output_file)).stem


def _is_toggle(column: pd.Series) -> bool:
    """Return whether *column* holds only booleans, i.e. is a toggle."""
    return pd.api.types.is_bool_dtype(column)


def _state_label(value) -> str:
    """Return the cell text for a state column: an integer when it is one."""
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


class SpecificationCurve(Plot):
    """Draw a specification curve from a sweep's results table.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        Supported keys:

        - ``specification_columns``: the parameter columns, in display order.
        - ``outcome_column``: the column holding the metric to compare.
        - ``converged_column``: optional boolean column marking fits that
          converged.  When absent, a non-null outcome is taken as converged.
        - ``labels``: mapping of column name to display label.
        - ``sort``: ``"ascending"`` (default), ``"descending"`` or ``"none"``.
        - ``marker_size``: marker area in points squared.
        - ``show_median``: whether to draw the median line (default ``True``).
        - ``figure.paper_size`` / ``figure.width`` / ``figure.height``.
        - ``styling.palette``: a key of
          :data:`~orbitdet.visualization.palettes.PALETTES`.
        - ``titles.suptitle``, ``titles.outcome``, ``titles.matrix``.
        - ``axes.outcome_label``, ``axes.matrix_label``.
        - ``axes.outcome_scale``: ``"linear"`` (default), ``"log"`` or
          ``"symlog"``.
        - ``axes.outcome_clip``: cut the axis at this value and mark the
          outcomes above it with a caret, so a few bad fits cannot flatten the
          comparison.  ``axes.outcome_annotate_max`` labels the largest cut-off
          value (default ``False``).
    frame : pandas.DataFrame
        One row per specification.  Must contain every column named by
        ``specification_columns`` and ``outcome_column``.
    specification_columns : Sequence[str], optional
        The parameter columns.  Overrides the config.
    outcome_column : str, optional
        The metric column.  Overrides the config.
    converged_column : str, optional
        The convergence flag column.  Overrides the config.
    name : str, optional
        Output stem.  Defaults to the config's ``output_file``.

    Raises
    ------
    ValueError
        If no specification or outcome column is configured, or if a named
        column is missing from *frame*.
    """

    def __init__(
        self,
        cfg,
        frame: pd.DataFrame,
        specification_columns: Sequence[str] | None = None,
        outcome_column: str | None = None,
        converged_column: str | None = None,
        name: str | None = None,
    ):
        super().__init__(cfg, name=name or _output_name(cfg))
        self.frame = frame

        self.specification_columns = list(
            specification_columns or _cfg_get(cfg, "specification_columns", default=None) or []
        )
        self.outcome_column = outcome_column or _cfg_get(cfg, "outcome_column", default=None)
        self.converged_column = converged_column or _cfg_get(cfg, "converged_column", default=None)

        if not self.specification_columns:
            raise ValueError(
                "SpecificationCurve needs at least one specification column; set "
                "`specification_columns` in the figure config or pass it directly."
            )
        if not self.outcome_column:
            raise ValueError(
                "SpecificationCurve needs an outcome column; set `outcome_column` in "
                "the figure config or pass it directly."
            )

        required = [*self.specification_columns, self.outcome_column]
        if self.converged_column:
            required.append(self.converged_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} are not in the frame. Available: {list(frame.columns)}."
            )

    # -- Frame preparation --------------------------------------------------

    def _sorted_frame(self) -> pd.DataFrame:
        """Return the frame in display order, with a ``_col`` index column.

        Sorting by outcome is what makes the top panel a monotone curve.  Rows
        with a missing outcome sort last regardless of direction, so the
        non-converged fits collect at the right instead of interleaving.
        """
        frame = self.frame.copy()
        order = str(_cfg_get(self.cfg, "sort", default="ascending"))
        if order in ("ascending", "descending"):
            frame = frame.sort_values(
                self.outcome_column,
                ascending=order == "ascending",
                na_position="last",
            )
        elif order != "none":
            raise ValueError(f"Unknown sort '{order}'. Use 'ascending', 'descending' or 'none'.")

        frame = frame.reset_index(drop=True)
        frame["_col"] = np.arange(len(frame))
        return frame

    def _converged_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Return which rows converged, falling back to "has an outcome"."""
        if self.converged_column:
            return frame[self.converged_column].astype(bool)
        return frame[self.outcome_column].notna()

    # -- Figure -------------------------------------------------------------

    def _make_figure(self):
        """Build the two-panel figure and return ``(fig, (outcome, matrix))``."""
        frame = self._sorted_frame()
        columns = self.specification_columns
        outcome_column = self.outcome_column

        labels = _cfg_get(self.cfg, "labels", default=None) or {}
        display_labels = [str(labels.get(column, column)) for column in columns]

        colors = palette(_cfg_get(self.cfg, "styling", "palette", default=None))
        point_color = colors[0]
        failed_color = colors[1] if len(colors) > 1 else point_color
        marker_size = float(_cfg_get(self.cfg, "marker_size", default=28))

        width, height = self._figure_size(panels=1)
        # The truth table needs one row per parameter, so the figure grows with
        # the number of parameters.  A configured height is treated as a floor
        # rather than an override: a height that cannot fit the rows clips the
        # bottom parameter's label.
        height = max(height, 2.6 + _ROW_HEIGHT * len(columns))

        # The outcome panel gets a fixed share and the matrix grows with its
        # rows, so adding a parameter lengthens the table instead of squashing
        # the curve.
        matrix_height = max(_ROW_HEIGHT * len(columns), 0.8)
        fig, (ax_outcome, ax_matrix) = plt.subplots(
            2,
            1,
            figsize=(width, height),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3.0, matrix_height],
                # Room for the matrix panel's title, which otherwise sits on the
                # outcome panel's bottom spine.
                "hspace": 0.22,
            },
        )

        self._draw_outcome(
            ax_outcome,
            frame,
            outcome_column,
            self._converged_mask(frame),
            point_color,
            failed_color,
            marker_size,
        )
        self._draw_matrix(ax_matrix, frame, columns, display_labels, point_color, marker_size)

        ax_outcome.set_title(str(_cfg_get(self.cfg, "titles", "outcome", default="Outcome")))
        ax_outcome.set_ylabel(
            str(_cfg_get(self.cfg, "axes", "outcome_label", default=outcome_column))
        )
        outcome_scale = str(_cfg_get(self.cfg, "axes", "outcome_scale", default="linear"))
        if outcome_scale not in ("linear", "log", "symlog"):
            raise ValueError(
                f"axes.outcome_scale must be 'linear', 'log' or 'symlog', not {outcome_scale!r}."
            )
        ax_outcome.set_yscale(outcome_scale)

        clip = _cfg_get(self.cfg, "axes", "outcome_clip", default=None)
        if clip is not None:
            self._clip_outcome(
                ax_outcome,
                frame,
                outcome_column,
                float(clip),
                point_color,
                marker_size,
            )
        ax_matrix.set_title(
            str(_cfg_get(self.cfg, "titles", "matrix", default="Estimated parameters"))
        )

        suptitle = _cfg_get(self.cfg, "titles", "suptitle", default=None)
        if suptitle:
            fig.suptitle(str(suptitle))

        # The specification axis is shared, so only the bottom panel carries
        # ticks; the top panel's are hidden to keep the two panels flush.  The
        # tick labels are hidden too: a specification is identified by its
        # column in the matrix, not by an index the reader has to count.
        ax_outcome.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_matrix.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Constrained layout sizes the panels but leaves almost no margin below
        # the matrix, which clips the bottom parameter's label.  Reserve a
        # margin proportional to the table height.
        fig.set_layout_engine("constrained")
        bottom_margin = min(0.03 + 0.015 * len(columns), 0.12)
        fig.get_layout_engine().set(rect=(0.0, bottom_margin, 1.0, 1.0 - bottom_margin))
        return fig, (ax_outcome, ax_matrix)

    # -- Panels -------------------------------------------------------------

    def _clip_outcome(
        self,
        ax,
        frame: pd.DataFrame,
        outcome_column: str,
        clip: float,
        point_color: str,
        marker_size: float,
    ) -> None:
        """Cut the axis at *clip* and mark the outcomes above it.

        A sweep often has a few fits far worse than the rest, and letting them
        set the axis flattens everything worth comparing.  Those points are
        drawn as carets pinned just inside the top edge, so the reader sees that
        a result exists there without it dictating the scale.
        """
        known = frame[outcome_column].dropna()
        if known.empty:
            return

        low = float(known.min())
        # Leave a little headroom above the clip so the carets are not on the
        # frame itself, and pad below the smallest outcome.
        top = clip * 1.04
        ax.set_ylim(max(low * 0.85, 0.0), top)

        above = frame[outcome_column] > clip
        if not above.any():
            return

        caret_y = clip * 0.97
        ax.scatter(
            frame.loc[above, "_col"],
            np.full(int(above.sum()), caret_y),
            marker="^",
            s=marker_size * 1.6,
            color=point_color,
            clip_on=False,
            zorder=4,
            label=f"> {clip:g}",
        )
        # The carets sit along the top edge and the outcomes along the bottom,
        # so the legend goes in the empty middle of the panel.
        ax.legend(loc="center right", frameon=False, fontsize=8)

        if bool(_cfg_get(self.cfg, "axes", "outcome_annotate_max", default=False)):
            # One label for the largest cut-off value, so its magnitude is not
            # lost entirely.  Labelling every caret would just repeat itself.
            worst = frame[outcome_column].idxmax()
            ax.annotate(
                f"{float(frame.loc[worst, outcome_column]):,.0f}",
                xy=(float(frame.loc[worst, "_col"]), caret_y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=point_color,
            )

    def _draw_outcome(
        self,
        ax,
        frame: pd.DataFrame,
        outcome_column: str,
        converged: pd.Series,
        point_color: str,
        failed_color: str,
        marker_size: float,
    ) -> None:
        """Draw the outcome per specification, marking non-converged fits."""
        good = frame[converged]
        bad = frame[~converged]

        if not good.empty:
            ax.scatter(
                good["_col"],
                good[outcome_column],
                marker="o",
                s=marker_size,
                color=point_color,
                zorder=3,
            )

        if not bad.empty:
            # A failed fit is marked, not dropped: which specifications fail is
            # itself a result.  A failed fit that still produced an outcome is
            # drawn at that outcome, so the axis keeps its real scale; only a
            # fit with no outcome at all has no y position, and those are placed
            # at the bottom of the axis, where they read as "no value".
            has_outcome = bad[outcome_column].notna()
            placed = bad[has_outcome]
            unplaced = bad[~has_outcome]

            if not placed.empty:
                ax.scatter(
                    placed["_col"],
                    placed[outcome_column],
                    marker="x",
                    s=marker_size,
                    color=failed_color,
                    zorder=3,
                    label="did not converge",
                )

            if not unplaced.empty:
                # Anchor the "no value" row to the real data range when there is
                # one, so the markers sit just below the plotted outcomes instead
                # of defining the axis themselves.  A log axis has no room below
                # the minimum, so there the row goes a fraction of the way down
                # towards it instead.
                known = frame[outcome_column].dropna()
                # A log axis has no room below the minimum; symlog does, since
                # it is linear there.
                log_scale = ax.get_yscale() == "log"
                if known.empty:
                    bottom, span = 0.0, 1.0
                else:
                    bottom = float(known.min())
                    span = float(known.max()) - bottom
                if log_scale and bottom > 0:
                    failed_y = bottom * 0.85
                else:
                    failed_y = bottom - 0.06 * (span if span > 0 else 1.0)
                ax.scatter(
                    unplaced["_col"],
                    np.full(len(unplaced), failed_y),
                    marker="x",
                    s=marker_size,
                    color=failed_color,
                    zorder=3,
                    # Only the first group carries the legend label, so a sweep
                    # with both kinds of failure does not list it twice.
                    label=None if not placed.empty else "did not converge",
                )

            ax.legend(loc="lower right", frameon=False, fontsize=8)

        if bool(_cfg_get(self.cfg, "show_median", default=True)):
            known = frame[outcome_column].dropna()
            if not known.empty:
                median = float(known.median())
                ax.axhline(median, color="0.35", linewidth=0.7, linestyle=(0, (8, 4)), zorder=1)
                ax.annotate(
                    "median",
                    xy=(0.0, median),
                    xycoords=("axes fraction", "data"),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="0.35",
                    va="bottom",
                    ha="left",
                )

        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)

    def _draw_matrix(
        self,
        ax,
        frame: pd.DataFrame,
        specification_columns: Sequence[str],
        display_labels: Sequence[str],
        point_color: str,
        marker_size: float,
    ) -> None:
        """Draw the truth table: one row per parameter, one column per fit.

        A toggle row is a filled square where the parameter was estimated and a
        pale square where it was not.  A *state* row (the libration order is 0,
        1 or 2) is written as its value instead: a square would say "on" for
        order 1 and order 2 alike, which is exactly the distinction the column
        exists to make.
        """
        columns = list(specification_columns)
        toggle_columns = [column for column in columns if _is_toggle(frame[column])]
        state_columns = [column for column in columns if column not in toggle_columns]

        # Row 0 is the first parameter, so the table reads top-down in the same
        # order as the config lists the columns.
        row_of = {column: index for index, column in enumerate(columns)}
        n_rows = len(columns)
        n_cols = len(frame)

        # The pale "off" squares are drawn for every toggle cell first, so the
        # filled ones sit on top and the grid reads as a table.
        if toggle_columns:
            off_rows, off_cols = [], []
            on_rows, on_cols = [], []
            for column in toggle_columns:
                row = row_of[column]
                values = frame[column].astype(bool).to_numpy()
                off_cols.extend(frame["_col"][~values])
                off_rows.extend([row] * int((~values).sum()))
                on_cols.extend(frame["_col"][values])
                on_rows.extend([row] * int(values.sum()))

            if off_cols:
                ax.scatter(
                    off_cols,
                    off_rows,
                    marker="s",
                    s=marker_size,
                    color=_CELL_OFF,
                    linewidths=0,
                    zorder=2,
                )
            if on_cols:
                ax.scatter(
                    on_cols,
                    on_rows,
                    marker="s",
                    s=marker_size,
                    color=point_color,
                    linewidths=0,
                    zorder=3,
                )

        for column in state_columns:
            row = row_of[column]
            for col, value in zip(frame["_col"], frame[column]):
                if pd.isna(value):
                    continue
                ax.text(
                    col,
                    row,
                    _state_label(value),
                    ha="center",
                    va="center",
                    fontsize=marker_size**0.5 * 0.85,
                    color=point_color,
                    zorder=3,
                )

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(display_labels)
        # The axis is padded by half a row so the first and last parameter rows
        # are not clipped by the panel edge.
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xlim(-0.6, n_cols - 0.4)
        ax.set_xlabel("")
        matrix_label = str(_cfg_get(self.cfg, "axes", "matrix_label", default=""))
        if matrix_label:
            ax.set_ylabel(matrix_label, rotation=-90, labelpad=14)
            ax.yaxis.set_label_position("right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for side in ("right", "top", "bottom"):
            ax.spines[side].set_visible(False)
