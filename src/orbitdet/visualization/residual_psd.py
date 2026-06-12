import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
from omegaconf import DictConfig
from scipy.signal import welch
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.observations import get_observatory_info

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


def _plot_config(cfg: DictConfig):
    return cfg.get("residual_psd", {})


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


def _resample_uniform(
    times_days: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    finite_mask = np.isfinite(times_days) & np.isfinite(values)
    times_days = times_days[finite_mask]
    values = values[finite_mask]

    if times_days.size < 2:
        raise ValueError("Need at least two finite residual samples to compute a PSD.")

    order = np.argsort(times_days)
    times_days = times_days[order]
    values = values[order]

    unique_times, unique_indices = np.unique(times_days, return_index=True)
    values = values[unique_indices]

    if unique_times.size < 2:
        raise ValueError("Need at least two unique observation times to compute a PSD.")

    sample_spacing_days = float(np.median(np.diff(unique_times)))
    if not np.isfinite(sample_spacing_days) or sample_spacing_days <= 0.0:
        raise ValueError("Could not infer a valid sampling interval for the residual PSD.")

    uniform_times = np.arange(
        unique_times[0], unique_times[-1] + sample_spacing_days / 2.0, sample_spacing_days
    )
    uniform_values = np.interp(uniform_times, unique_times, values)
    return uniform_times, uniform_values, sample_spacing_days


def _compute_psd(
    values: np.ndarray,
    sample_spacing_days: float,
    window_length_days: float,
    window_type: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method != "welch":
        raise ValueError(f"Unsupported PSD method '{method}'. Supported methods: welch")

    centered = values - np.mean(values)
    sample_count = centered.size
    window_length_samples = int(round(window_length_days / sample_spacing_days))
    window_length_samples = max(8, window_length_samples)
    window_length_samples = min(window_length_samples, sample_count)

    noverlap = window_length_samples // 2
    try:
        frequencies, power_density = welch(
            centered,
            fs=1.0 / sample_spacing_days,
            window=window_type,
            nperseg=window_length_samples,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            return_onesided=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsupported window type '{window_type}' for Welch PSD. "
            "Use a SciPy-compatible window name (e.g., hamming, hann, boxcar, blackman)."
        ) from exc
    return frequencies, power_density


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
        functions=(lambda frequency: 1.0 / frequency, lambda period: 1.0 / period),
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

    ax.format_coord = lambda x_value, y_value: _format_hover_coord(x_value, y_value, axes_cfg)


def plot_residuals_psd(
    cfg: DictConfig,
    observation_collection: obs.ObservationCollection,
    window_length_days: float,
    plot_cfg,
    observation_parsers: list[obs_proc.ObservationParserType] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot Welch PSD spectra of the residual signal for the orbit determination."""
    if observation_parsers is None:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets()
        )
    else:
        observation_sets: list[obs.SingleObservationSet] = (
            observation_collection.get_single_observation_sets(observation_parsers)
        )

    figure_cfg = plot_cfg.get("figure", {})
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

    fig_width = float(figure_cfg.get("width", DEFAULT_FIGURE_WIDTH))
    fig_height = float(figure_cfg.get("height", DEFAULT_FIGURE_HEIGHT))
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    colors = plt.get_cmap(styling_cfg.get("cmap", "tab20"))
    line_width = float(styling_cfg.get("line_width", 1.2))
    legend_ncols = int(legend_cfg.get("ncols", 2))
    legend_bbox = legend_cfg.get("bbox_to_anchor", {})
    legend_bbox_x = float(legend_bbox.get("x", 0.5))
    legend_bbox_y = float(legend_bbox.get("y", -0.15))
    min_frequency = np.inf
    max_frequency = 0.0

    for set_index, obs_set in enumerate(observation_sets):
        observatory_code = obs_set.link_definition.link_ends[links.receiver].reference_point
        target_name = obs_set.link_definition.link_ends[links.transmitter].body_name
        info = get_observatory_info(cfg, observatory_code)
        color = colors(set_index % colors.N)

        obs_times_sec_j2000 = np.array([epoch.to_float() for epoch in obs_set.observation_times])
        residuals = np.array(obs_set.residuals)

        time_mask = np.ones(obs_times_sec_j2000.shape, dtype=bool)
        if time_start_seconds is not None:
            time_mask &= obs_times_sec_j2000 >= time_start_seconds
        if time_end_seconds is not None:
            time_mask &= obs_times_sec_j2000 <= time_end_seconds

        if not np.any(time_mask):
            logger.warning(
                "Skipping %s because no observations fall inside the configured time range.",
                info["name"],
            )
            continue

        obs_times_sec_j2000 = obs_times_sec_j2000[time_mask]
        residuals = residuals[time_mask]

        reference_seconds_since_j2000 = float(obs_times_sec_j2000.min())
        obs_times_days = (obs_times_sec_j2000 - reference_seconds_since_j2000) / 86400.0

        ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
        dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

        _, ra_uniform, ra_dt_days = _resample_uniform(obs_times_days, ra_residuals_arcsec)
        _, dec_uniform, dec_dt_days = _resample_uniform(obs_times_days, dec_residuals_arcsec)

        ra_freq, ra_psd = _compute_psd(
            ra_uniform, ra_dt_days, window_length_days, window_type, method
        )
        dec_freq, dec_psd = _compute_psd(
            dec_uniform, dec_dt_days, window_length_days, window_type, method
        )

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
            label=f"{info['name']} - {info['region']}",
            color=color,
            line_width=line_width,
        )
        _plot_psd_series(
            axs[1],
            dec_freq,
            dec_psd,
            label=f"{info['name']} - {info['region']}",
            color=color,
            line_width=line_width,
        )

    if not np.isfinite(min_frequency) or max_frequency <= 0.0:
        raise ValueError("Could not determine a valid PSD frequency range to plot.")

    _configure_psd_axis(axs[0], min_frequency, max_frequency, x_lim_cfg, axes_cfg)
    _configure_psd_axis(axs[1], min_frequency, max_frequency, x_lim_cfg, axes_cfg)

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
