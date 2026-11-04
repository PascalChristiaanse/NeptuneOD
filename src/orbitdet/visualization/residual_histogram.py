"""Residual histograms with fitted normal curves and normality tests.

Both plots consume an :class:`~orbitdet.analysis.access.ObservationAccess` - the
standard observation frame joined to the live observation sets - so the
residuals, the grouping column and the scan-frame rotation all come from one
tidy frame instead of a per-set walk over the collection's link ends.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy import stats

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.analysis.distribution import density_histogram, normality_test
from orbitdet.visualization.base import Plot, _make_hover_formatter, _snake_case
from orbitdet.visualization.residuals import _cfg_get

logger = logging.getLogger(__name__)

__all__ = ["ResidualHistogram", "ResidualScanHistogram"]


def _shapiro_label(statistic: float, p_value: float) -> str:
    """Render a Shapiro-Wilk result as an annotation label."""
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return "Shapiro-Wilk: n/a"
    return f"Shapiro-Wilk W={statistic:.4f}, p={p_value:.3e}"


class _HistogramBase(Plot):
    """Shared plumbing for the residual histograms."""

    def __init__(
        self,
        cfg: DictConfig,
        access: ObservationAccess,
        use_figure_name: str | None = None,
        group_by: str | list[str] | None = None,
        unit: str | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        include_rejected: bool = False,
    ):
        super().__init__(cfg)
        self.access = access
        self.use_figure_name = use_figure_name
        self.group_by = group_by
        self.unit = unit
        self.include_rejected = include_rejected
        if (fig is not None and ax is None) or (fig is None and ax is not None):
            raise ValueError("Both fig and ax must be provided together.")
        self.fig = fig
        self.ax = ax
        self.name = use_figure_name or _snake_case(self.__class__.__name__)

    def _plot_cfg(self):
        """Return the figure config block for this plot."""
        return _cfg_get(self.cfg.figures, self.use_figure_name, default=None)

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

    def _frame(self, unit: str, scan: bool):
        """Return the analysis frame, filtered to the rows this plot should show."""
        frame = analysis_frame(self.access, unit=unit, scan=scan, normalized=False)
        if not self.include_rejected and "rejected" in frame.columns:
            frame = frame.loc[~frame["rejected"].astype(bool)]
        return frame

    def _target_name(self, frame) -> str:
        """Return the observed object's name for the suptitle, or ``"target"``."""
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            return str(frame["observed_object"].dropna().iloc[0])
        return "target"

    def _suptitle(self, plot_cfg, frame, key: str, default: str) -> str:
        """Resolve and format the suptitle."""
        suptitle = _cfg_get(plot_cfg, "titles", key, default=default)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=self._target_name(frame))
        except Exception:
            pass
        return suptitle

    def _legend(self, ax: plt.Axes, plot_cfg) -> None:
        """Apply the configured legend layout to an axes."""
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

    def _styling(self, plot_cfg):
        """Return ``(colors, n_bins)`` from the figure config."""
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        n_bins = int(_cfg_get(plot_cfg, "styling", "n_bins", default=30))
        return colors, n_bins


class ResidualHistogram(_HistogramBase):
    """Histograms of the RA and Dec residuals with fitted normal curves.

    Each group (by default ``observatory_code``) contributes one histogram per
    component, overlaid with a normal PDF fitted to that group's sample and
    annotated with the Shapiro-Wilk normality test.  The bin heights are
    probability densities, so groups of different sizes are comparable.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    use_figure_name : str | None
        Name of the figure configuration block under ``cfg.figures``.  Defaults
        to ``"residual_histogram"``.
    group_by : str | list[str] | None
        Standard-frame column(s) to group the histograms by.  ``None`` (default)
        falls back to the figure config's ``group_by`` and then to
        ``"observatory_code"``.
    unit : str | None
        Display unit for the residual axis: ``"arcsec"`` or ``"mas"``.  ``None``
        (default) uses the figure config's ``axes.unit`` (itself defaulting to
        ``"arcsec"``).
    fig, ax : optional
        A pre-existing figure and a length-2 axes sequence (RA then Dec).
    include_rejected : bool
        Whether to include rows that outlier rejection removed.
    """

    def __init__(self, *args, use_figure_name: str | None = "residual_histogram", **kwargs):
        super().__init__(*args, use_figure_name=use_figure_name, **kwargs)

    def _make_figure(self):
        plot_cfg = self._plot_cfg()
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        if self.fig is None and self.ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=False)
        else:
            if len(self.ax) != 2:
                raise ValueError("Expected two subplots for RA and Dec residuals.")
            fig, axs = self.fig, self.ax

        unit = self.unit or _cfg_get(plot_cfg, "axes", "unit", default="arcsec")
        frame = self._frame(unit, scan=False)

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

        colors, n_bins = self._styling(plot_cfg)
        bell_curve_color = _cfg_get(plot_cfg, "styling", "bell_curve_color", default="red")
        bell_curve_lw = float(_cfg_get(plot_cfg, "styling", "bell_curve_lw", default=1.5))

        for index, (key, group) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = first_col if component == 0 else second_col
                values = group[column].to_numpy("float64")
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

                # Normal curve fitted to this group's sample.
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

        self._legend(axs[0], plot_cfg)
        self._legend(axs[1], plot_cfg)
        fig.suptitle(
            self._suptitle(
                plot_cfg, frame, "suptitle", "Residual Histograms for {target_name}"
            )
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
        The Hydra experiment configuration.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    use_figure_name : str | None
        Name of the figure configuration block under ``cfg.figures``.  Defaults
        to ``"residual_histogram"``.
    group_by : str | list[str] | None
        Standard-frame column(s) to group the histograms by.
    unit : str | None
        Display unit for the residual axis: ``"arcsec"`` or ``"mas"``.
    fig, ax : optional
        A pre-existing figure and a length-2 axes sequence (AL then AC).
    include_rejected : bool
        Whether to include rows that outlier rejection removed.
    """

    def __init__(self, *args, use_figure_name: str | None = "residual_histogram", **kwargs):
        super().__init__(*args, use_figure_name=use_figure_name, **kwargs)

    def _make_figure(self):
        plot_cfg = self._plot_cfg()
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        if self.fig is None and self.ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=False)
        else:
            if len(self.ax) != 2:
                raise ValueError("Expected two subplots for AL and AC residuals.")
            fig, axs = self.fig, self.ax

        unit = self.unit or _cfg_get(plot_cfg, "axes", "unit", default="arcsec")
        frame = self._frame(unit, scan=True)

        al_col = f"residual_along_scan_rad_{unit}"
        ac_col = f"residual_across_scan_rad_{unit}"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                "ResidualScanHistogram requires scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )
        frame = frame.loc[frame["scan_angle"].notna()]

        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        colors, n_bins = self._styling(plot_cfg)
        fit_color = _cfg_get(plot_cfg, "styling", "fit_color", default="red")
        fit_lw = float(_cfg_get(plot_cfg, "styling", "fit_lw", default=1.5))
        mean_color = _cfg_get(plot_cfg, "styling", "mean_color", default="black")
        mean_ls = _cfg_get(plot_cfg, "styling", "mean_ls", default="--")
        error_color = _cfg_get(plot_cfg, "styling", "error_color", default="black")
        error_lw = float(_cfg_get(plot_cfg, "styling", "error_lw", default=1.0))

        for index, (key, group) in enumerate(frame.groupby(group_cols, dropna=False, sort=True)):
            color = colors(index % colors.N)
            label = self._group_label(key, group_cols)

            for component, ax in enumerate(axs):
                column = al_col if component == 0 else ac_col
                values = group[column].to_numpy("float64")
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

        self._legend(axs[0], plot_cfg)
        self._legend(axs[1], plot_cfg)
        fig.suptitle(
            self._suptitle(
                plot_cfg,
                frame,
                "suptitle_scan_hist",
                "Scan Residual Histograms for {target_name}",
            )
        )
        fig.set_tight_layout(True)
        return fig, axs
