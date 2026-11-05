"""Normal Q-Q plots of the RA and Dec residuals.

Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the standard
observation frame joined to the live observation sets - so the residuals and the
grouping column come from one tidy frame instead of a per-set walk over the
collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.analysis.distribution import qq_points
from orbitdet.visualization.base import Plot, _make_hover_formatter, _snake_case
from orbitdet.visualization.residuals import _cfg_get

logger = logging.getLogger(__name__)

__all__ = ["ResidualQQ"]


class ResidualQQ(Plot):
    """Q-Q plots of the RA and Dec residuals against a normal distribution.

    Each group (by default ``observatory_code``) contributes one series per
    component.  The sample is standardized before plotting, so the axes are in
    units of standard deviations and the residual unit does not matter - a
    normally distributed sample lies on the ``y = x`` reference line.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    use_figure_name : str | None
        Name of the figure configuration block under ``cfg.figures``.  Defaults
        to ``"residual_qq"``.
    group_by : str | list[str] | None
        Standard-frame column(s) to group the series by.  ``None`` (default)
        falls back to the figure config's ``group_by`` and then to
        ``"observatory_code"``.
    fig, ax : optional
        A pre-existing figure and a length-2 axes sequence (RA then Dec).
    include_rejected : bool
        Whether to include rows that outlier rejection removed.
    """

    def __init__(
        self,
        cfg: DictConfig,
        access: ObservationAccess,
        use_figure_name: str | None = "residual_qq",
        group_by: str | list[str] | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        include_rejected: bool = False,
    ):
        super().__init__(cfg)
        self.use_figure_name = use_figure_name or "residual_qq"
        self.access = access
        self.group_by = group_by
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
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        if self.fig is None and self.ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
        else:
            if len(self.ax) != 2:
                raise ValueError("Expected two subplots for RA and Dec residuals.")
            fig, axs = self.fig, self.ax

        # The Q-Q sample is standardized, so no unit conversion is needed.
        frame = analysis_frame(self.access, unit=None, scan=False, normalized=False)
        if not self.include_rejected and "rejected" in frame.columns:
            frame = frame.loc[~frame["rejected"].astype(bool)]

        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)
        marker_types = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

        for index, (key, group) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = "residual_first_rad" if component == 0 else "residual_second_rad"
                theoretical, sample = qq_points(group[column].to_numpy("float64"))
                if theoretical.size == 0:
                    continue
                ax.scatter(
                    theoretical,
                    sample,
                    marker=marker,
                    s=marker_size,
                    label=label,
                    color=color,
                    alpha=0.5,
                )

        title_ra = _cfg_get(plot_cfg, "titles", "ra", default="Right Ascension Q-Q")
        title_dec = _cfg_get(plot_cfg, "titles", "dec", default="Declination Q-Q")
        axs[0].set_title(title_ra)
        axs[1].set_title(title_dec)

        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Sample Quantiles [std]")
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Theoretical Quantiles [std]")
        for ax in axs:
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)

        # Reference line y = x.
        ref_line_style = _cfg_get(
            plot_cfg, "styling", "ref_line", default={"ls": "--", "color": "k", "lw": 1.0}
        )
        ref_ls = ref_line_style.get("ls", "--") if isinstance(ref_line_style, dict) else "--"
        ref_color = ref_line_style.get("color", "k") if isinstance(ref_line_style, dict) else "k"
        ref_lw = float(ref_line_style.get("lw", 1.0)) if isinstance(ref_line_style, dict) else 1.0

        for ax in axs:
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, ls=ref_ls, color=ref_color, lw=ref_lw, label="Normal reference")

        hover_x_label = _cfg_get(
            plot_cfg, "axes", "hover_x_label", default="Theoretical Quantiles [std]"
        )
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default="Sample Quantiles [std]"
        )
        fmt = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

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

        axs[0].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        axs[1].legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)

        target_name = "target"
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            target_name = str(frame["observed_object"].dropna().iloc[0])
        suptitle = _cfg_get(
            plot_cfg, "titles", "suptitle", default="Residual Q-Q Plot for {target_name}"
        )
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=target_name)
        except Exception:
            pass

        fig.suptitle(suptitle)
        fig.set_tight_layout(True)
        return fig, axs
