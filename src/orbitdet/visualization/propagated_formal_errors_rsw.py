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


class PropagatedFormalErrorsRSW(Plot):
    """Plot propagated formal errors in the RSW frame (Radial / Along-track / Cross-track).

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    epochs : np.ndarray
        Epochs in seconds since J2000.
    formal_errors_rsw : np.ndarray
        Shape ``(N, >=3)`` array of formal errors in RSW frame;
        columns 0, 1, 2 = Radial, Along-track, Cross-track.
    observation_windows : list[tuple[float, float]], optional
        List of ``(start_epoch, end_epoch)`` pairs in seconds since J2000,
        marking observation windows with vertical dashed lines.
    """

    def __init__(
        self,
        cfg: DictConfig,
        epochs: np.ndarray,
        formal_errors_rsw: np.ndarray,
        observation_windows: list[tuple[float, float]] | None = None,
    ):
        super().__init__(cfg)
        self.epochs = epochs
        self.formal_errors_rsw = formal_errors_rsw
        self.observation_windows = observation_windows

    def _make_figure(self):
        cfg = self.cfg
        epochs = self.epochs
        formal_errors_rsw = self.formal_errors_rsw
        observation_windows = self.observation_windows

        plot_cfg = _cfg_get(cfg, "propagated_formal_errors_rsw", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=12)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=6)

        times = _seconds_since_j2000_to_datetimes(epochs)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.plot(times, formal_errors_rsw[:, 0], label="Radial", alpha=0.8)
        ax.plot(times, formal_errors_rsw[:, 1], label="Along-track", alpha=0.8)
        ax.plot(times, formal_errors_rsw[:, 2], label="Cross-track", alpha=0.8)

        if observation_windows is not None:
            for win_start, win_end in observation_windows:
                dt_start = _seconds_since_j2000_to_datetimes(np.asarray([win_start]))[0]
                dt_end = _seconds_since_j2000_to_datetimes(np.asarray([win_end]))[0]
                ax.axvline(dt_start, color="b", linestyle="--", alpha=0.5)
                ax.axvline(dt_end, color="b", linestyle="--", alpha=0.5)

        title = _cfg_get(
            plot_cfg, "titles", "title", default="Propagated Formal Errors - RSW"
        )
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Formal Error [m]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)
        _configure_datetime_axis(ax)
        ax.set_xlabel(x_label)

        fig.set_tight_layout(True)

        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, ax