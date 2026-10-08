"""Plotter classes for integrator and interpolator convergence validation.

These classes produce two figures per validation run:

1. **Difference over time** — For each timestep, the position difference norm
   (Richardson error estimate) is plotted over time on a log-y axis. One line
   per timestep, n-1 lines total.

2. **RMS per timestep** — The RMS of the position difference norm over all
   epochs is plotted against the timestep on a log-log axis. A single line
   connecting the n points.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from orbitdet.visualization.base import Plot


def _cfg_get(cfg: DictConfig | dict | None, *keys, default=None):
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        try:
            cur = cur.get(k)
        except Exception:
            try:
                cur = cur[k]
            except Exception:
                return default
    return default if cur is None else cur


def _seconds_since_j2000_to_datetimes(seconds_since_j2000):
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _configure_datetime_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    def _format(x, y):
        try:
            dt = mdates.num2date(x)
            xs = dt.isoformat(sep=" ")
        except Exception:
            xs = f"{x:.6g}"
        return f"{hover_x_label}: {xs}, {hover_y_label}: {y:.3e}"

    return _format


def _integrator_order(integrator_type: str) -> int:
    """Return the nominal order of an integrator type.

    Used to determine how many initial samples to skip when computing
    error statistics, avoiding Runge's phenomenon at the start of the arc.
    """
    match integrator_type:
        case "Euler":
            return 1
        case "RK3" | "ralston_3":
            return 3
        case "RK4" | "Ralston4" | "ralston_4" | "three_eight_rule_rk_4":
            return 4
        case "RKF78":
            return 8
        case "RKF89":
            return 9
        case "RKV89":
            return 9
        case "RKF108":
            return 10
        case "RKF1210":
            return 12
        case "RKF1412":
            return 14
        case "BulirschStoer6":
            return 6
        case "BulirschStoer8":
            return 8
        case "BulirschStoer10":
            return 10
        case _:
            # Default: try to extract a number from the type string
            import re
            nums = re.findall(r"\d+", integrator_type)
            return int(nums[-1]) if nums else 4


class IntegratorConvergence(Plot):
    """Plot integrator convergence results from a sweep.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    results : dict[float, dict]
        Dictionary mapping timestep (seconds) to a dict with keys:
        ``epochs`` (np.ndarray), ``pos_diff_norm`` (np.ndarray),
        ``integrator_type`` (str).
    integrator_type : str
        The integrator type being plotted (e.g. ``"RKF78"``).
    strategy : str, optional
        The validation strategy (``"richardson"`` or ``"forward_backward"``).
        Used in plot titles.
    """

    def __init__(
        self,
        cfg: DictConfig,
        results: dict[float, dict],
        integrator_type: str,
        strategy: str = "richardson",
    ):
        super().__init__(cfg)
        self.results = results
        self.integrator_type = integrator_type
        self.strategy = strategy

    def _make_figure(self):
        cfg = self.cfg
        results = self.results
        integrator_type = self.integrator_type
        strategy = self.strategy

        plot_cfg = _cfg_get(cfg, "integrator_convergence", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=14)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=10)

        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=False)

        # Sort timesteps
        sorted_timesteps = sorted(results.keys())

        # Strategy-specific labels
        if strategy == "forward_backward":
            diff_label = "Round-trip Position Error Over Time"
            rms_label = "RMS Round-trip Position Error vs Timestep"
            suptitle_prefix = "Forward-Backward"
            y_label = "||Δr_roundtrip|| [m]"
        else:
            diff_label = "Position Error Over Time"
            rms_label = "RMS Position Error vs Timestep"
            suptitle_prefix = "Richardson"
            y_label = "||Δr|| [m]"

        # ---- Panel 1: Difference over time (log y) ----
        ax1 = axes[0]
        for dt in sorted_timesteps:
            r = results[dt]
            times = _seconds_since_j2000_to_datetimes(r["epochs"])
            ax1.plot(times, r["pos_diff_norm"], label=f"Δt = {dt:.0f} s", alpha=0.8)

        ax1.set_yscale("log")
        ax1.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "difference_over_time",
                default=f"{diff_label} — {integrator_type}",
            )
        )
        ax1.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label_diff", default=y_label)
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, which="both")
        ax1.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch"),
            _cfg_get(plot_cfg, "axes", "hover_y_label_diff", default=y_label),
        )
        _configure_datetime_axis(ax1)

        # ---- Panel 2: RMS and Max per timestep (log-log) ----
        ax2 = axes[1]
        timesteps_arr = np.array(sorted_timesteps)
        rms_arr = np.array([np.sqrt(np.mean(results[dt]["pos_diff_norm"] ** 2))
                            for dt in sorted_timesteps])
        max_arr = np.array([np.max(results[dt]["pos_diff_norm"])
                            for dt in sorted_timesteps])

        ax2.loglog(timesteps_arr, rms_arr, "o-", markersize=6, linewidth=2,
                   label="RMS")
        ax2.loglog(timesteps_arr, max_arr, "s--", markersize=5, linewidth=1.5,
                   label="Max", alpha=0.7)

        # Annotate each RMS point with the timestep value
        for dt, rms in zip(timesteps_arr, rms_arr):
            ax2.annotate(
                f"{dt:.0f} s",
                (dt, rms),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                ha="center",
            )

        ax2.legend(fontsize=8)
        ax2.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "rms_per_timestep",
                default=f"{rms_label} — {integrator_type}",
            )
        )
        ax2.set_xlabel(
            _cfg_get(plot_cfg, "axes", "x_label_rms", default="Timestep Δt [s]")
        )
        ax2.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label_rms", default="||Δr|| [m]")
        )
        ax2.grid(True, alpha=0.3, which="both")
        ax2.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label_rms", default="Timestep [s]"),
            _cfg_get(plot_cfg, "axes", "hover_y_label_rms", default="||Δr|| [m]"),
        )

        fig.suptitle(
            _cfg_get(
                plot_cfg,
                "titles",
                "suptitle",
                default=f"{suptitle_prefix} Convergence — {integrator_type}",
            ),
            fontsize=14,
        )
        fig.set_tight_layout(True)

        return fig, axes


class InterpolatorConvergence(Plot):
    """Plot interpolator convergence results from a sweep.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    results : dict[float, dict]
        Dictionary mapping timestep (seconds) to a dict with keys:
        ``epochs`` (np.ndarray), ``pos_diff_norm`` (np.ndarray),
        ``interpolator_type`` (str).
    interpolator_type : str
        The interpolator type being plotted (e.g. ``"linear"``).
    """

    def __init__(
        self,
        cfg: DictConfig,
        results: dict[float, dict],
        interpolator_type: str,
    ):
        super().__init__(cfg)
        self.results = results
        self.interpolator_type = interpolator_type

    def _make_figure(self):
        cfg = self.cfg
        results = self.results
        interpolator_type = self.interpolator_type

        plot_cfg = _cfg_get(cfg, "interpolator_convergence", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=14)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=10)

        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=False)

        # Sort timesteps
        sorted_timesteps = sorted(results.keys())

        # ---- Panel 1: Difference over time (log y) ----
        ax1 = axes[0]
        for dt in sorted_timesteps:
            r = results[dt]
            times = _seconds_since_j2000_to_datetimes(r["epochs"])
            ax1.plot(times, r["pos_diff_norm"], label=f"Δt = {dt:.0f} s", alpha=0.8)

        ax1.set_yscale("log")
        ax1.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "difference_over_time",
                default=f"Interpolation Error Over Time — {interpolator_type}",
            )
        )
        ax1.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label_diff", default="||Δr|| [m]")
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, which="both")
        ax1.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch"),
            _cfg_get(plot_cfg, "axes", "hover_y_label_diff", default="||Δr|| [m]"),
        )
        _configure_datetime_axis(ax1)

        # ---- Panel 2: RMS and Max per timestep (log-log) ----
        ax2 = axes[1]
        timesteps_arr = np.array(sorted_timesteps)
        rms_arr = np.array([np.sqrt(np.mean(results[dt]["pos_diff_norm"] ** 2))
                            for dt in sorted_timesteps])
        max_arr = np.array([np.max(results[dt]["pos_diff_norm"])
                            for dt in sorted_timesteps])

        ax2.loglog(timesteps_arr, rms_arr, "o-", markersize=6, linewidth=2,
                   label="RMS")
        ax2.loglog(timesteps_arr, max_arr, "s--", markersize=5, linewidth=1.5,
                   label="Max", alpha=0.7)

        for dt, rms in zip(timesteps_arr, rms_arr):
            ax2.annotate(
                f"{dt:.0f} s",
                (dt, rms),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                ha="center",
            )

        ax2.legend(fontsize=8)
        ax2.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "rms_per_timestep",
                default=f"RMS Interpolation Error vs Timestep — {interpolator_type}",
            )
        )
        ax2.set_xlabel(
            _cfg_get(plot_cfg, "axes", "x_label_rms", default="Timestep Δt [s]")
        )
        ax2.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label_rms", default="||Δr|| [m]")
        )
        ax2.grid(True, alpha=0.3, which="both")
        ax2.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label_rms", default="Timestep [s]"),
            _cfg_get(plot_cfg, "axes", "hover_y_label_rms", default="||Δr|| [m]"),
        )

        fig.suptitle(
            _cfg_get(
                plot_cfg,
                "titles",
                "suptitle",
                default=f"Interpolator Convergence — {interpolator_type}",
            ),
            fontsize=14,
        )
        fig.set_tight_layout(True)

        return fig, axes


class IntegratorFamilyComparison(Plot):
    """Plot all integrator families on a single comparison figure.

    Each integrator type gets a distinct color and marker shape. RMS is shown
    as a solid line, Max as a dashed line, both on log-log axes.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    results_by_type : dict[str, dict[float, dict]]
        Maps integrator type name to its ``{timestep: result_data}`` dict.
    strategy : str, optional
        The validation strategy (``"richardson"`` or ``"forward_backward"``).
    """

    def __init__(
        self,
        cfg: DictConfig,
        results_by_type: dict[str, dict[float, dict]],
        strategy: str = "richardson",
    ):
        super().__init__(cfg)
        self.results_by_type = results_by_type
        self.strategy = strategy

    def _make_figure(self):
        cfg = self.cfg
        results_by_type = self.results_by_type
        strategy = self.strategy

        plot_cfg = _cfg_get(cfg, "integrator_family_comparison", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=12)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Distinct markers and a colormap for up to ~10 integrator types
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "8"]
        colors = plt.cm.tab10.colors  # 10 distinct colors

        sorted_types = sorted(results_by_type.keys())

        # Store all plotted lines and their metadata for hover lookup
        plotted_lines = []

        for idx, integrator_type in enumerate(sorted_types):
            type_results = results_by_type[integrator_type]
            sorted_timesteps = sorted(type_results.keys())
            timesteps_arr = np.array(sorted_timesteps)
            rms_arr = np.array([
                np.sqrt(np.mean(type_results[dt]["pos_diff_norm"] ** 2))
                for dt in sorted_timesteps
            ])
            max_arr = np.array([
                np.max(type_results[dt]["pos_diff_norm"])
                for dt in sorted_timesteps
            ])

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            line_rms, = ax.loglog(timesteps_arr, rms_arr, marker=marker, color=color,
                                  linestyle="-", linewidth=2, markersize=7,
                                  label=f"{integrator_type} (RMS)")
            line_max, = ax.loglog(timesteps_arr, max_arr, marker=marker, color=color,
                                  linestyle="--", linewidth=1.2, markersize=5,
                                  alpha=0.7,
                                  label=f"{integrator_type} (Max)")

            plotted_lines.append((line_rms, integrator_type, "RMS", timesteps_arr, rms_arr))
            plotted_lines.append((line_max, integrator_type, "Max", timesteps_arr, max_arr))

        strategy_label = "Forward-Backward" if strategy == "forward_backward" else "Richardson"
        ax.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "title",
                default=f"Integrator Family Comparison — {strategy_label}",
            )
        )
        ax.set_xlabel(
            _cfg_get(plot_cfg, "axes", "x_label", default="Timestep Δt [s]")
        )
        ax.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label", default="||Δr|| [m]")
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)

        # Interactive hover annotation
        hover_annot = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
            visible=False,
        )

        def _find_nearest_line(xdata, ydata):
            """Find the closest data point across all plotted lines."""
            best_dist = np.inf
            best_info = None
            for line, name, kind, xs, ys in plotted_lines:
                if len(xs) == 0:
                    continue
                # Find nearest point on this line
                dists = np.hypot(np.log10(xs) - np.log10(xdata),
                                 np.log10(ys) - np.log10(ydata))
                idx = np.argmin(dists)
                d = dists[idx]
                if d < best_dist:
                    best_dist = d
                    best_info = (name, kind, xs[idx], ys[idx])
            return best_info

        def _on_hover(event):
            if event.inaxes != ax:
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
                return
            info = _find_nearest_line(event.xdata, event.ydata)
            if info is None:
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
                return
            name, kind, dt, val = info
            hover_annot.xy = (event.xdata, event.ydata)
            hover_annot.set_text(f"{name} ({kind})\nΔt = {dt:.0f} s\n||Δr|| = {val:.3e} m")
            hover_annot.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_hover)

        fig.suptitle(
            _cfg_get(
                plot_cfg,
                "titles",
                "suptitle",
                default=f"Integrator Family Comparison — {strategy_label}",
            ),
            fontsize=14,
        )
        fig.set_tight_layout(True)

        return fig, ax


class IntegratorErrorVsTime(Plot):
    """Plot max error vs integration time for a single integrator type.

    A single log-log panel: x = integration time [s], y = max ||Δr|| [m].
    """

    def __init__(
        self,
        cfg: DictConfig,
        results: dict[float, dict],
        integrator_type: str,
        strategy: str = "richardson",
    ):
        super().__init__(cfg)
        self.results = results
        self.integrator_type = integrator_type
        self.strategy = strategy

    def _make_figure(self):
        cfg = self.cfg
        results = self.results
        integrator_type = self.integrator_type
        strategy = self.strategy

        plot_cfg = _cfg_get(cfg, "integrator_error_vs_time", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=6)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sorted_timesteps = sorted(results.keys())
        times_arr = np.array([results[dt]["integration_time"] for dt in sorted_timesteps])
        max_arr = np.array([np.max(results[dt]["pos_diff_norm"]) for dt in sorted_timesteps])

        ax.loglog(times_arr, max_arr, "o-", markersize=6, linewidth=2)

        for t, m, dt in zip(times_arr, max_arr, sorted_timesteps):
            ax.annotate(
                f"{dt:.0f} s",
                (t, m),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                ha="center",
            )

        strategy_label = "Forward-Backward" if strategy == "forward_backward" else "Richardson"
        ax.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "title",
                default=f"Max Error vs Integration Time — {integrator_type} ({strategy_label})",
            )
        )
        ax.set_xlabel(
            _cfg_get(plot_cfg, "axes", "x_label", default="Integration Time [s]")
        )
        ax.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label", default="Max ||Δr|| [m]")
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label", default="Time [s]"),
            _cfg_get(plot_cfg, "axes", "hover_y_label", default="Max ||Δr|| [m]"),
        )

        fig.suptitle(
            _cfg_get(
                plot_cfg,
                "titles",
                "suptitle",
                default=f"Max Error vs Integration Time — {integrator_type} ({strategy_label})",
            ),
            fontsize=14,
        )
        fig.set_tight_layout(True)

        return fig, ax


class IntegratorErrorVsTimeFamily(Plot):
    """Plot max error vs integration time for all integrator types on one figure.

    Each integrator type gets a distinct color and marker shape.
    """

    def __init__(
        self,
        cfg: DictConfig,
        results_by_type: dict[str, dict[float, dict]],
        strategy: str = "richardson",
    ):
        super().__init__(cfg)
        self.results_by_type = results_by_type
        self.strategy = strategy

    def _make_figure(self):
        cfg = self.cfg
        results_by_type = self.results_by_type
        strategy = self.strategy

        plot_cfg = _cfg_get(cfg, "integrator_error_vs_time_family", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=10)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=7)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "8"]
        colors = plt.cm.tab10.colors

        sorted_types = sorted(results_by_type.keys())

        for idx, integrator_type in enumerate(sorted_types):
            type_results = results_by_type[integrator_type]
            sorted_timesteps = sorted(type_results.keys())
            times_arr = np.array([type_results[dt]["integration_time"] for dt in sorted_timesteps])
            max_arr = np.array([np.max(type_results[dt]["pos_diff_norm"]) for dt in sorted_timesteps])

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            ax.loglog(times_arr, max_arr, marker=marker, color=color,
                      linestyle="-", linewidth=2, markersize=7,
                      label=integrator_type)

        strategy_label = "Forward-Backward" if strategy == "forward_backward" else "Richardson"
        ax.set_title(
            _cfg_get(
                plot_cfg,
                "titles",
                "title",
                default=f"Max Error vs Integration Time — {strategy_label}",
            )
        )
        ax.set_xlabel(
            _cfg_get(plot_cfg, "axes", "x_label", default="Integration Time [s]")
        )
        ax.set_ylabel(
            _cfg_get(plot_cfg, "axes", "y_label", default="Max ||Δr|| [m]")
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        ax.format_coord = _make_hover_formatter(
            _cfg_get(plot_cfg, "axes", "hover_x_label", default="Time [s]"),
            _cfg_get(plot_cfg, "axes", "hover_y_label", default="Max ||Δr|| [m]"),
        )

        fig.suptitle(
            _cfg_get(
                plot_cfg,
                "titles",
                "suptitle",
                default=f"Max Error vs Integration Time — {strategy_label}",
            ),
            fontsize=14,
        )
        fig.set_tight_layout(True)

        return fig, ax