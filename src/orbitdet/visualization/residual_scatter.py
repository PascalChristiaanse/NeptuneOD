"""RA-vs-Dec residual scatter with observation time encoded as colour.

Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the standard
observation frame joined to the live observation sets - so the residuals, the
grouping column and the epochs all come from one tidy frame instead of a
per-set walk over the collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.visualization.base import Plot, _make_hover_formatter, _snake_case
from orbitdet.visualization.residuals import _cfg_get

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


class ResidualScatter(Plot):
    """Plot RA vs Dec residuals as a scatter, with observation time as colour.

    Each group (by default ``observatory_code``) gets its own marker shape, and
    every point is coloured on a gradient of its observation epoch, so temporal
    trends are visible even though time is not an axis.  A colourbar maps the
    colour back to a date.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    use_figure_name : str | None
        Name of the figure configuration block under ``cfg.figures``.  Defaults
        to ``"residual_scatter"``.
    group_by : str | list[str] | None
        Standard-frame column(s) to group the marker shapes by.  ``None``
        (default) falls back to the figure config's ``group_by`` and then to
        ``"observatory_code"``.
    unit : str | None
        Display unit for the residual axes: ``"arcsec"`` or ``"mas"``.  ``None``
        (default) uses the figure config's ``axes.unit`` (itself defaulting to
        ``"arcsec"``).
    fig : plt.Figure | None
        An optional pre-existing figure.  Must be provided together with ``ax``.
    ax : plt.Axes | None
        An optional pre-existing axes.  Must be provided together with ``fig``.
    include_rejected : bool
        Whether to plot rows that outlier rejection removed.  Defaults to
        ``False``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        access: ObservationAccess,
        use_figure_name: str | None = "residual_scatter",
        group_by: str | list[str] | None = None,
        unit: str | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        include_rejected: bool = False,
    ):
        super().__init__(cfg)
        self.use_figure_name = use_figure_name or "residual_scatter"
        self.access = access
        self.group_by = group_by
        self.unit = unit
        self.include_rejected = include_rejected
        if (fig is not None and ax is None) or (fig is None and ax is not None):
            raise ValueError("Both fig and ax must be provided together.")
        self.fig = fig
        self.ax = ax
        self.name = self.use_figure_name or _snake_case(self.__class__.__name__)

    def _grouping(self, plot_cfg) -> list[str]:
        """Resolve the grouping columns: argument, then config, then default."""
        group_by = self.group_by
        if group_by is None:
            group_by = _cfg_get(plot_cfg, "group_by", default=None)
        if group_by is None:
            group_by = "observatory_code"
        return [group_by] if isinstance(group_by, str) else list(group_by)

    @staticmethod
    def _group_label(key, columns: list[str]) -> str:
        """Render a groupby key as a legend label."""
        values = key if isinstance(key, tuple) else (key,)
        if len(columns) == 1:
            return str(values[0])
        return ", ".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))

    def _make_figure(self):
        plot_cfg = _cfg_get(self.cfg.figures, self.use_figure_name, default=None)
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27)

        if self.fig is None and self.ax is None:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        else:
            fig, ax = self.fig, self.ax

        unit = self.unit or _cfg_get(plot_cfg, "axes", "unit", default="arcsec")

        # One tidy frame: standard columns + residual/weight/sigma + unit columns.
        frame = analysis_frame(self.access, unit=unit, scan=False, normalized=False)
        if not self.include_rejected and "rejected" in frame.columns:
            frame = frame.loc[~frame["rejected"].astype(bool)]

        first_col = f"residual_first_rad_{unit}"
        second_col = f"residual_second_rad_{unit}"
        if first_col not in frame.columns or second_col not in frame.columns:
            raise ValueError(
                f"Analysis frame has no '{first_col}'/'{second_col}' column; "
                f"cannot plot residuals in unit '{unit}'."
            )

        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        # Global epoch range, so every group shares one colour scale.
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
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)
        marker_types = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

        for index, (key, group) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            epochs = group["epoch_tdb"].to_numpy("float64")
            first = group[first_col].to_numpy("float64")
            second = group[second_col].to_numpy("float64")
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

        target_name = "target"
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            target_name = str(frame["observed_object"].dropna().iloc[0])

        title = _cfg_get(
            plot_cfg, "titles", "title", default="RA vs DEC Residuals for {target_name}"
        )
        try:
            if isinstance(title, str):
                title = title.format(target_name=target_name)
        except Exception:
            pass

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

        hover_x_label = _cfg_get(
            plot_cfg, "axes", "hover_x_label", default=f"RA Residual [{unit}]"
        )
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default=f"DEC Residual [{unit}]"
        )
        ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)

        try:
            legend_ncols = int(_cfg_get(plot_cfg, "legend", "ncols", default=2))
        except Exception:
            legend_ncols = 2

        bbox = _cfg_get(plot_cfg, "legend", "bbox_to_anchor", default={"x": 0.5, "y": -0.15})
        try:
            if isinstance(bbox, dict):
                bbox_tuple = (float(bbox.get("x", 0.5)), float(bbox.get("y", -0.15)))
            else:
                bbox_tuple = tuple(float(x) for x in bbox)
        except Exception:
            bbox_tuple = (0.5, -0.15)

        ax.legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        fig.set_tight_layout(True)

        return fig, ax
