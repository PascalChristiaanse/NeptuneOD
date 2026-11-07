"""Normal Q-Q plots of the RA and Dec residuals.

Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the standard
observation frame joined to the live observation sets - so the residuals and the
clustering column come from one tidy frame instead of a per-set walk over the
collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from orbitdet.analysis.distribution import qq_points
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get

logger = logging.getLogger(__name__)

__all__ = ["ResidualQQ"]


class ResidualQQ(ResidualPlot):
    """Q-Q plots of the RA and Dec residuals against a normal distribution.

    Each cluster (by default ``observatory_code``) contributes one series per
    component.  The sample is standardized before plotting, so the axes are in
    units of standard deviations and the residual unit does not matter - a
    normally distributed sample lies on the ``y = x`` reference line.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size(8.27 * 2, 8.27 * 2 / 2)
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        # The Q-Q sample is standardized, so no unit conversion is needed.
        frame = self._frame(unit=None, scan=False, normalized=False)

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = "residual_first_rad" if component == 0 else "residual_second_rad"
                theoretical, sample = qq_points(cluster[column].to_numpy("float64"))
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

        self._legend(axs[0])
        self._legend(axs[1])

        fig.suptitle(
            self._suptitle(frame, "suptitle", "Residual Q-Q Plot for {target_name}")
        )
        fig.set_tight_layout(True)
        return fig, axs
