import functools
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object

from orbitdet.analysis.distribution import find_psd_peaks, lombscargle_psd, welch_psd
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get

display = os.environ.get("DISPLAY")
is_headless_display = display == ":99" or display == "localhost:99" or display == "127.0.0.1:99"

matplotlib.rcParams["webagg.port"] = 8988
matplotlib.rcParams["webagg.open_in_browser"] = False

if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or is_headless_display:
    matplotlib.use("WebAgg", force=True)
elif display or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("QtAgg", force=True)
    except Exception:
        matplotlib.use("TkAgg", force=True)
else:
    matplotlib.use("WebAgg", force=True)


logger = logging.getLogger(__name__)

DEFAULT_WINDOW_LENGTH_DAYS = 30.0
DEFAULT_WINDOW_TYPE = "hamming"
DEFAULT_FIGURE_WIDTH = 16.54
DEFAULT_FIGURE_HEIGHT = 8.27
DEFAULT_PERIOD_TICKS_DAYS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# Triton's orbital period around Neptune (days). Used to highlight the expected
# signal in the residual spectrum.
TRITON_PERIOD_DAYS = 5.877

# Number of frequency samples per decade for the Lomb-Scargle grid.
DEFAULT_LOMB_FREQS_PER_DECADE = 200


def _parse_time_range(plot_cfg):
    time_range_cfg = plot_cfg.get("time_range", {})
    start = time_range_cfg.get("start")
    end = time_range_cfg.get("end")
    start_epoch = iso_string_to_epoch_time_object(start) if start else None
    end_epoch = iso_string_to_epoch_time_object(end) if end else None
    start_seconds = float(start_epoch.to_float()) if start_epoch is not None else None
    end_seconds = float(end_epoch.to_float()) if end_epoch is not None else None
    return start_seconds, end_seconds


def _normalize_window_type(window_type: str, method: str) -> str:
    normalized = window_type.lower().strip()
    if method == "welch" and normalized == "welch":
        logger.warning(
            "window.type='welch' is not a valid SciPy window name; using '%s' instead.",
            DEFAULT_WINDOW_TYPE,
        )
        return DEFAULT_WINDOW_TYPE
    return normalized


def _rad_to_arcsec(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad) * 3600.0


def _principal_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return np.remainder(angle_rad + np.pi, 2.0 * np.pi) - np.pi


def _format_period_days(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return ""

    seconds = value * 86400.0
    if seconds >= 86400.0:
        return f"{seconds / 86400.0:.1f} d"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.1f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.0f} s"


def _format_frequency_cycles_per_day(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return ""
    return f"{value:g}"


def _format_hover_coord(x_value: float, y_value: float, axes_cfg) -> str:
    hover_x_label = axes_cfg.get("hover_x_label", "Period [days/cycle]")
    hover_y_label = axes_cfg.get("hover_y_label", "Power density")

    if not np.isfinite(x_value) or x_value <= 0.0:
        return f"{hover_x_label}=undefined, {hover_y_label}={y_value:.6g}"

    period_days = 1.0 / x_value
    return f"{hover_x_label}={period_days:.6g}, {hover_y_label}={y_value:.6g}"


def _frequency_to_period(frequency: np.ndarray) -> np.ndarray:
    """Convert frequency [cycles/day] to period [days]; module-level for picklability."""
    return np.divide(
        1.0,
        frequency,
        where=frequency != 0,
        out=np.full_like(frequency, np.inf, dtype=float),
    )


def _period_to_frequency(period: np.ndarray) -> np.ndarray:
    """Convert period [days] to frequency [cycles/day]; module-level for picklability."""
    return np.divide(
        1.0,
        period,
        where=period != 0,
        out=np.full_like(period, np.inf, dtype=float),
    )


def _plot_psd_series(
    ax: plt.Axes,
    frequencies: np.ndarray,
    power_density: np.ndarray,
    label: str,
    color,
    line_width: float,
) -> None:
    positive = frequencies > 0.0
    frequencies = frequencies[positive]
    power_density = power_density[positive]

    if frequencies.size == 0:
        raise ValueError("PSD produced no positive frequencies to plot.")

    ax.plot(frequencies, power_density, label=label, color=color, linewidth=line_width)


def _configure_psd_axis(
    ax: plt.Axes,
    min_frequency: float,
    max_frequency: float,
    x_lim_cfg,
    axes_cfg,
) -> None:
    x_scale = axes_cfg.get("x_scale", "log")
    ax.set_xscale(x_scale)
    ax.set_xlabel(axes_cfg.get("x_label", "Frequency [cycles/day]"))
    ax.set_ylabel(axes_cfg.get("y_label", "Power density [arcsec²/day]"))

    x_min = float(x_lim_cfg.get("min", min_frequency))
    x_max = float(x_lim_cfg.get("max", max_frequency))
    if x_scale == "log" and (x_min <= 0.0 or x_max <= 0.0):
        raise ValueError("x_lim must be positive when using a log frequency axis.")
    if x_min >= x_max:
        raise ValueError("x_lim min must be smaller than x_lim max.")
    ax.set_xlim(x_min, x_max)

    if x_scale == "log":
        major_subs = tuple(
            float(value) for value in axes_cfg.get("frequency_tick_subs", (1.0, 2.0, 5.0))
        )
        minor_subs = tuple(
            float(value)
            for value in axes_cfg.get("frequency_minor_subs", (3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
        )
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=major_subs, numticks=20))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs, numticks=100))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_frequency_cycles_per_day))

    period_axis = ax.secondary_xaxis(
        "top",
        functions=(_frequency_to_period, _period_to_frequency),
    )
    period_axis.set_xlabel(axes_cfg.get("period_label", "Period [days/cycle]"))
    period_ticks = axes_cfg.get("period_ticks_days", DEFAULT_PERIOD_TICKS_DAYS)
    period_ticks = [float(value) for value in period_ticks if float(value) > 0.0]
    period_min = 1.0 / x_max
    period_max = 1.0 / x_min
    period_ticks = [value for value in period_ticks if period_min <= value <= period_max]
    if period_ticks:
        period_axis.set_xticks(period_ticks)
    period_axis.xaxis.set_major_formatter(FuncFormatter(_format_period_days))

    ax.format_coord = functools.partial(_format_hover_coord, axes_cfg=axes_cfg)


class ResidualsPSD(ResidualPlot):
    """Plot Welch or Lomb-Scargle PSD spectra of the residual signal.

    Consumes an :class:`~orbitdet.analysis.access.ObservationAccess` - the
    standard observation frame joined to the live observation sets - so the
    residuals, epochs and clustering column come from one tidy frame instead of a
    per-set walk over the collection's link ends.  The spectral estimate itself
    is delegated to :mod:`orbitdet.analysis.distribution`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by``, ``include_rejected``, ``method`` and ``window_length_days``
        are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    @property
    def window_length_days(self) -> float:
        """Welch segment length in days, from the config."""
        return float(_cfg_get(self.cfg, "window_length_days", default=DEFAULT_WINDOW_LENGTH_DAYS))

    def _make_figure(self):
        window_length_days = self.window_length_days
        plot_cfg = self.cfg

        styling_cfg = plot_cfg.get("styling", {})
        x_lim_cfg = plot_cfg.get("x_lim", plot_cfg.get("axes", {}).get("x_lim", {}))
        axes_cfg = plot_cfg.get("axes", {})
        titles_cfg = plot_cfg.get("titles", {})
        legend_cfg = plot_cfg.get("legend", {})
        window_cfg = plot_cfg.get("window", {})
        window_type = str(plot_cfg.get("window_type", window_cfg.get("type", DEFAULT_WINDOW_TYPE)))
        method = str(plot_cfg.get("method", "welch")).lower()
        window_type = _normalize_window_type(window_type, method)
        time_start_seconds, time_end_seconds = _parse_time_range(plot_cfg)
        peak_cfg = plot_cfg.get("peaks", {})
        peak_prominence = float(peak_cfg.get("prominence", 0.1))
        annotate_peaks = bool(peak_cfg.get("annotate", True))
        show_triton_period = bool(peak_cfg.get("show_triton_period", True))
        show_average_error = bool(peak_cfg.get("show_average_error", True))

        fig_width, fig_height = self._figure_size(cfg=plot_cfg)
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
        colors = plt.get_cmap(styling_cfg.get("cmap", "tab20"))
        line_width = float(styling_cfg.get("line_width", 1.2))
        legend_ncols = int(legend_cfg.get("ncols", 2))
        legend_bbox = legend_cfg.get("bbox_to_anchor", {})
        legend_bbox_x = float(legend_bbox.get("x", 0.5))
        legend_bbox_y = float(legend_bbox.get("y", -0.15))
        min_frequency = np.inf
        max_frequency = 0.0

        # One tidy frame: standard columns + residual/weight/sigma columns.
        frame = self._frame(unit=None, scan=False, normalized=False)

        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        target_name = "target"
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            target_name = str(frame["observed_object"].dropna().iloc[0])

        for set_index, (key, cluster) in enumerate(
            frame.groupby(group_cols, dropna=False, sort=True)
        ):
            label = self._group_label(key, group_cols)
            color = colors(set_index % colors.N)

            obs_times_sec_j2000 = cluster["epoch_tdb"].to_numpy("float64")
            residuals = np.column_stack(
                [
                    cluster["residual_first_rad"].to_numpy("float64"),
                    cluster["residual_second_rad"].to_numpy("float64"),
                ]
            )

            time_mask = np.isfinite(obs_times_sec_j2000)
            if time_start_seconds is not None:
                time_mask &= obs_times_sec_j2000 >= time_start_seconds
            if time_end_seconds is not None:
                time_mask &= obs_times_sec_j2000 <= time_end_seconds

            if not np.any(time_mask):
                logger.warning(
                    "Skipping %s because no observations fall inside the configured time range.",
                    label,
                )
                continue

            obs_times_sec_j2000 = obs_times_sec_j2000[time_mask]
            residuals = residuals[time_mask]

            reference_seconds_since_j2000 = float(obs_times_sec_j2000.min())
            obs_times_days = (obs_times_sec_j2000 - reference_seconds_since_j2000) / 86400.0

            ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
            dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

            try:
                if method == "welch":
                    ra_freq, ra_psd = welch_psd(
                        obs_times_days,
                        ra_residuals_arcsec,
                        window_length=window_length_days,
                        window=window_type,
                    )
                    dec_freq, dec_psd = welch_psd(
                        obs_times_days,
                        dec_residuals_arcsec,
                        window_length=window_length_days,
                        window=window_type,
                    )
                else:
                    ra_freq, ra_psd = lombscargle_psd(obs_times_days, ra_residuals_arcsec)
                    dec_freq, dec_psd = lombscargle_psd(obs_times_days, dec_residuals_arcsec)
            except ValueError as exc:
                logger.warning("Skipping %s because PSD computation failed: %s", label, str(exc))
                continue
            positive_ra = ra_freq > 0.0
            positive_dec = dec_freq > 0.0
            if np.any(positive_ra):
                min_frequency = min(min_frequency, float(np.min(ra_freq[positive_ra])))
                max_frequency = max(max_frequency, float(np.max(ra_freq[positive_ra])))
            if np.any(positive_dec):
                min_frequency = min(min_frequency, float(np.min(dec_freq[positive_dec])))
                max_frequency = max(max_frequency, float(np.max(dec_freq[positive_dec])))

            _plot_psd_series(
                axs[0],
                ra_freq,
                ra_psd,
                label=label,
                color=color,
                line_width=line_width,
            )
            _plot_psd_series(
                axs[1],
                dec_freq,
                dec_psd,
                label=label,
                color=color,
                line_width=line_width,
            )

            if annotate_peaks:
                for axis, freq, psd in ((axs[0], ra_freq, ra_psd), (axs[1], dec_freq, dec_psd)):
                    for peak_freq, peak_power in find_psd_peaks(
                        freq, psd, prominence=peak_prominence
                    )[:3]:
                        axis.axvline(
                            peak_freq,
                            color=color,
                            linestyle=":",
                            linewidth=0.8,
                            alpha=0.6,
                        )
                        axis.annotate(
                            f"{_format_period_days(1.0 / peak_freq)} ({peak_freq:.4g} c/d)",
                            xy=(peak_freq, peak_power),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                            color=color,
                        )

            if show_triton_period:
                triton_frequency = 1.0 / TRITON_PERIOD_DAYS
                for axis in axs:
                    axis.axvline(
                        triton_frequency,
                        color="black",
                        linewidth=1.2,
                        linestyle="--",
                        label=f"Triton period ({TRITON_PERIOD_DAYS:.3f} d)",
                    )

        if not np.isfinite(min_frequency) or max_frequency <= 0.0:
            raise ValueError("Could not determine a valid PSD frequency range to plot.")

        _configure_psd_axis(axs[0], min_frequency, max_frequency, x_lim_cfg, axes_cfg)
        _configure_psd_axis(axs[1], min_frequency, max_frequency, x_lim_cfg, axes_cfg)

        if show_average_error:
            ra_rms = float(np.sqrt(np.mean(np.square(ra_residuals_arcsec))))
            dec_rms = float(np.sqrt(np.mean(np.square(dec_residuals_arcsec))))
            axs[0].text(
                0.02,
                0.98,
                f"RA RMS = {ra_rms:.3f} arcsec",
                transform=axs[0].transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
            axs[1].text(
                0.02,
                0.98,
                f"Dec RMS = {dec_rms:.3f} arcsec",
                transform=axs[1].transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        axs[0].set_title(titles_cfg.get("ra", "Right Ascension PSD"))
        axs[1].set_title(titles_cfg.get("dec", "Declination PSD"))
        axs[0].legend(
            ncols=legend_ncols, loc="upper center", bbox_to_anchor=(legend_bbox_x, legend_bbox_y)
        )
        axs[1].legend(
            ncols=legend_ncols, loc="upper center", bbox_to_anchor=(legend_bbox_x, legend_bbox_y)
        )
        fig.suptitle(
            titles_cfg.get("suptitle", f"Residual PSD for {target_name}").format(
                target_name=target_name
            )
        )
        fig.set_tight_layout(True)

        return fig, axs
