"""Residual histograms with fitted normal curves and normality tests.

Both plots consume an :class:`~orbitdet.analysis.access.ObservationAccess` - the
standard observation frame joined to the live observation sets - so the
residuals, the clustering column and the scan-frame rotation all come from one
tidy frame instead of a per-set walk over the collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from orbitdet.analysis.distribution import density_histogram, normality_test
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get

logger = logging.getLogger(__name__)

__all__ = ["ResidualHistogram", "ResidualScanHistogram"]


def _shapiro_label(statistic: float, p_value: float) -> str:
    """Render a Shapiro-Wilk result as an annotation label."""
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return "Shapiro-Wilk: n/a"
    return f"Shapiro-Wilk W={statistic:.4f}, p={p_value:.3e}"


class _HistogramBase(ResidualPlot):
    """Shared plumbing for the residual histograms."""

    def _styling(self):
        """Return ``(colors, n_bins)`` from the figure config."""
        cmap = _cfg_get(self.cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        n_bins = int(_cfg_get(self.cfg, "styling", "n_bins", default=30))
        return colors, n_bins


class ResidualHistogram(_HistogramBase):
    """Histograms of the RA and Dec residuals with fitted normal curves.

    Each cluster (by default ``observatory_code``) contributes one histogram per
    component, overlaid with a normal PDF fitted to that cluster's sample and
    annotated with the Shapiro-Wilk normality test.  The bin heights are
    probability densities, so clusters of different sizes are comparable.

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
        fig_w, fig_h = self._figure_size(8.27 * 2, 8.27 * 2 / 2)
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=False)

        unit = self.unit
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

        colors, n_bins = self._styling()
        bell_curve_color = _cfg_get(plot_cfg, "styling", "bell_curve_color", default="red")
        bell_curve_lw = float(_cfg_get(plot_cfg, "styling", "bell_curve_lw", default=1.5))

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = first_col if component == 0 else second_col
                values = cluster[column].to_numpy("float64")
                values = values[np.isfinite(values)]
                if values.size < 2:
                    continue

                density, edges, _error, _centers = density_histogram(values, bins=n_bins)
                if density.size == 0:
                    continue
                ax.hist(
                    values,
                    bins=edges,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=label,
                )

                # Normal curve fitted to this cluster's sample.
                mu = float(np.mean(values))
                sigma = float(np.std(values, ddof=1))
                if sigma > 0.0:
                    x = np.linspace(edges[0], edges[-1], 300)
                    ax.plot(
                        x,
                        stats.norm.pdf(x, mu, sigma),
                        color=bell_curve_color,
                        linewidth=bell_curve_lw,
                    )

                # Shapiro-Wilk normality test, annotated in the axes corner.
                statistic, p_value = normality_test(values)
                ax.annotate(
                    _shapiro_label(statistic, p_value),
                    xy=(0.98, 0.95 - index * 0.08),
                    xycoords="axes fraction",
                    fontsize=7,
                    ha="right",
                    va="top",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor=color),
                )

        title_ra = _cfg_get(plot_cfg, "titles", "ra", default="Right Ascension Residuals")
        title_dec = _cfg_get(plot_cfg, "titles", "dec", default="Declination Residuals")
        axs[0].set_title(title_ra)
        axs[1].set_title(title_dec)

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default=f"Residual [{unit}]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Probability Density")
        for ax in axs:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=f"Residual [{unit}]")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="Probability Density")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(
            self._suptitle(frame, "suptitle", "Residual Histograms for {target_name}")
        )
        fig.set_tight_layout(True)
        return fig, axs


class ResidualScanHistogram(_HistogramBase):
    """Sideways histograms of the along-scan and across-scan residuals.

    The scan-frame rotation is read straight from the analysis frame
    (``residual_along_scan_rad`` / ``residual_across_scan_rad``), so no per-set
    link-end walk or scan-angle lookup is needed.  Each component is shown as a
    horizontal histogram with Poisson error bars on the bin densities, a fitted
    normal curve, and the sample mean marked.

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
        fig_w, fig_h = self._figure_size(8.27 * 2, 8.27 * 2 / 2)
        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=False)

        unit = self.unit
        frame = self._frame(unit=unit, scan=True, normalized=False)

        al_col = f"residual_along_scan_rad_{unit}"
        ac_col = f"residual_across_scan_rad_{unit}"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                "ResidualScanHistogram requires scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )
        frame = frame.loc[frame["scan_angle"].notna()]

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, n_bins = self._styling()
        fit_color = _cfg_get(plot_cfg, "styling", "fit_color", default="red")
        fit_lw = float(_cfg_get(plot_cfg, "styling", "fit_lw", default=1.5))
        mean_color = _cfg_get(plot_cfg, "styling", "mean_color", default="black")
        mean_ls = _cfg_get(plot_cfg, "styling", "mean_ls", default="--")
        error_color = _cfg_get(plot_cfg, "styling", "error_color", default="black")
        error_lw = float(_cfg_get(plot_cfg, "styling", "error_lw", default=1.0))

        for index, (key, cluster) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = al_col if component == 0 else ac_col
                values = cluster[column].to_numpy("float64")
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue

                density, edges, error, centers = density_histogram(values, bins=n_bins)
                if density.size == 0:
                    continue
                ax.hist(
                    values,
                    bins=edges,
                    orientation="horizontal",
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=label,
                )
                ax.errorbar(
                    density,
                    centers,
                    xerr=error,
                    fmt="none",
                    ecolor=error_color,
                    elinewidth=error_lw,
                    capsize=2,
                )

                mu = float(np.mean(values))
                sigma = float(np.std(values, ddof=1))
                if sigma > 0.0:
                    y = np.linspace(edges[0], edges[-1], 300)
                    ax.plot(
                        stats.norm.pdf(y, mu, sigma),
                        y,
                        color=fit_color,
                        linewidth=fit_lw,
                        label=f"Fit: mu={mu:.3e}, sigma={sigma:.3e}",
                    )
                ax.axhline(
                    mu,
                    color=mean_color,
                    linestyle=mean_ls,
                    linewidth=fit_lw,
                    label=f"Mean: {mu:.3e} {unit}",
                )

        title_al = _cfg_get(plot_cfg, "titles", "al", default="Along-scan")
        title_ac = _cfg_get(plot_cfg, "titles", "ac", default="Across-scan")
        axs[0].set_title(title_al)
        axs[1].set_title(title_ac)

        y_label = _cfg_get(plot_cfg, "axes", "y_label", default=f"Residual [{unit}]")
        x_label = _cfg_get(plot_cfg, "axes", "hist_x_label", default="Probability Density")
        for ax in axs:
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Probability Density")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(
            self._suptitle(
                frame,
                "suptitle_scan_hist",
                "Scan Residual Histograms for {target_name}",
            )
        )
        fig.set_tight_layout(True)
        return fig, axs
