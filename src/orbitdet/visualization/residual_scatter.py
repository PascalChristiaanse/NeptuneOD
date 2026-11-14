"""RA-vs-Dec residual scatter with observation time encoded as colour.

Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the standard
observation frame joined to the live observation sets - so the residuals, the
clustering column and the epochs all come from one tidy frame instead of a
per-set walk over the collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get

logger = logging.getLogger(__name__)

__all__ = ["ResidualScatter"]


def _format_colorbar_tick(value, _position: int | None = None) -> str:
    """Format colorbar ticks as ``YYYY-MM`` dates.

    Module-level (not a nested closure) so the figure remains picklable.
    """
    try:
        dt = mdates.num2date(mdates.epoch2num(float(value)))
        return dt.strftime("%Y-%m")
    except Exception:
        return f"{value:.0f}"


class ResidualScatter(ResidualPlot):
    """Plot RA vs Dec residuals as a scatter, with observation time as colour.

    Each cluster (by default ``observatory_code``) gets its own marker shape, and
    every point is coloured on a gradient of its observation epoch, so temporal
    trends are visible even though time is not an axis.  A colourbar maps the
    colour back to a date.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by``, ``axes.unit`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        unit = self.unit

        # One tidy frame: standard columns + residual/weight/sigma + unit columns.
        frame = self._frame(unit=unit, scan=False, normalized=False)

        first_col = f"residual_first_rad_{unit}"
        second_col = f"residual_second_rad_{unit}"
        if first_col not in frame.columns or second_col not in frame.columns:
            raise ValueError(
                f"Analysis frame has no '{first_col}'/'{second_col}' column; "
                f"cannot plot residuals in unit '{unit}'."
            )

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        # Global epoch range, so every cluster shares one colour scale.
        epochs_all = frame["epoch_tdb"].to_numpy("float64")
        finite_epochs = epochs_all[np.isfinite(epochs_all)]
        if finite_epochs.size:
            global_t_min = float(finite_epochs.min())
            global_t_max = float(finite_epochs.max())
        else:
            global_t_min, global_t_max = 0.0, 1.0
        time_range = global_t_max - global_t_min if global_t_max > global_t_min else 1.0

        cmap_name = _cfg_get(plot_cfg, "styling", "cmap", default="rainbow")
        cmap = plt.get_cmap(cmap_name)
        _, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            epochs = cluster["epoch_tdb"].to_numpy("float64")
            first = cluster[first_col].to_numpy("float64")
            second = cluster[second_col].to_numpy("float64")
            keep = np.isfinite(epochs) & np.isfinite(first) & np.isfinite(second)
            if not keep.any():
                continue

            time_normalized = (epochs[keep] - global_t_min) / time_range
            colors = cmap(time_normalized)

            ax.scatter(
                first[keep],
                second[keep],
                marker=marker,
                s=marker_size,
                c=colors,
                label=label,
                alpha=0.5,
                edgecolors="none",
            )

        title = self._suptitle(frame, "title", "RA vs DEC Residuals for {target_name}")

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default=f"RA Residual [{unit}]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"DEC Residual [{unit}]")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Colourbar mapping colour back to an observation date.
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=global_t_min, vmax=global_t_max),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(_cfg_get(plot_cfg, "axes", "colorbar_label", default="Epoch"))
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_colorbar_tick))

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=f"RA Residual [{unit}]")
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default=f"DEC Residual [{unit}]"
        )
        ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)

        self._legend(ax)

        # Keep the constrained layout engine the journal style installs.  The
        # colorbar is created through that engine's gridspec, so switching to
        # tight layout here would raise (matplotlib refuses to swap engines once
        # a colorbar exists).
        fig.set_layout_engine("constrained")

        return fig, ax
