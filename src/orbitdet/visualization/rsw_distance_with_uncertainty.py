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


class RSWDistanceWithUncertainty(Plot):
    """2x2 plot of the RSW position difference between estimated and spice trajectories
    with :math:`\\pm 1\\sigma` and :math:`\\pm 3\\sigma` uncertainty envelopes from the
    propagated covariance.

    This combines the information from :class:`RSWDistance` (the line showing how the
    best-fit trajectory differs from the spice ephemeris in the RSW frame) with the
    formal errors from :class:`PropagatedFormalErrorsRSW` (the uncertainty envelopes),
    revealing how well the solution fits within the predicted uncertainty.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    epochs : np.ndarray
        Epochs in seconds since J2000 (same as used for covariance propagation).
    rsw_components : np.ndarray
        Shape ``(N, 3)`` array of R, S, W position differences [m].
    rsw_sigma : np.ndarray
        Shape ``(N, 3)`` array of 1-sigma formal errors in R, S, W [m].
    """

    def __init__(
        self,
        cfg: DictConfig,
        epochs: np.ndarray,
        rsw_components: np.ndarray,
        rsw_sigma: np.ndarray,
    ):
        super().__init__(cfg)
        self.epochs = epochs
        self.rsw_components = rsw_components
        self.rsw_sigma = rsw_sigma

    def _make_figure(self):
        cfg = self.cfg
        epochs = self.epochs
        rsw_components = self.rsw_components
        rsw_sigma = self.rsw_sigma

        times = _seconds_since_j2000_to_datetimes(epochs)

        # Norm of the RSW vector (position difference magnitude)
        rsw_norm = np.linalg.norm(rsw_components, axis=1)
        # Approximate 1-sigma uncertainty on the norm from quadrature sum
        sigma_norm = np.sqrt(np.sum(rsw_sigma**2, axis=1))

        plot_cfg = _cfg_get(cfg, "RSW_distance_with_uncertainty", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=16.54)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=11.0)

        titles = {
            "r": _cfg_get(plot_cfg, "titles", "r", default="Radial (R)"),
            "s": _cfg_get(plot_cfg, "titles", "s", default="Transverse (S)"),
            "w": _cfg_get(plot_cfg, "titles", "w", default="Cross-track (W)"),
            "norm": _cfg_get(plot_cfg, "titles", "norm", default="L2 Norm of RSW"),
        }
        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle",
            default="RSW Distance with 1-σ and 3-σ Uncertainty Envelopes",
        )

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Distance [m]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        # Line and envelope style defaults
        line_style = _cfg_get(plot_cfg, "style", "line", default="k-")
        line_width = _cfg_get(plot_cfg, "style", "linewidth", default=0.8)
        envelope_color = _cfg_get(plot_cfg, "style", "envelope_color", default="C0")
        sigma1_alpha = _cfg_get(plot_cfg, "style", "sigma1_alpha", default=0.25)
        sigma3_alpha = _cfg_get(plot_cfg, "style", "sigma3_alpha", default=0.1)

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        series = [
            (rsw_components[:, 0], rsw_sigma[:, 0], titles["r"]),
            (rsw_components[:, 1], rsw_sigma[:, 1], titles["s"]),
            (rsw_components[:, 2], rsw_sigma[:, 2], titles["w"]),
            (rsw_norm, sigma_norm, titles["norm"]),
        ]
        for ax, (values, sigma, title) in zip(axes, series):
            # 3σ envelope (most transparent — outermost band)
            ax.fill_between(
                times,
                values - 3.0 * sigma,
                values + 3.0 * sigma,
                alpha=sigma3_alpha,
                color=envelope_color,
                label=r"$3\sigma$",
            )
            # 1σ envelope (semi-transparent — inner band)
            ax.fill_between(
                times,
                values - sigma,
                values + sigma,
                alpha=sigma1_alpha,
                color=envelope_color,
                label=r"$1\sigma$",
            )
            # RSW / norm line
            ax.plot(
                times,
                values,
                line_style,
                linewidth=line_width,
                label=title,
            )
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize="small")
            ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        for ax in axes[2:]:
            _configure_datetime_axis(ax)
            ax.set_xlabel(x_label)

        fig.suptitle(suptitle, fontsize=14)
        fig.align_ylabels(axes)
        fig.set_tight_layout(True)

        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, axes