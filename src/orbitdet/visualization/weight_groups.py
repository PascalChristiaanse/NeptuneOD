"""Plot observation groups with coloured time-range spans.

Produces a two-panel figure:

- **Top**: RA residuals coloured by group, with coloured vertical spans
  indicating each group's time range.
- **Bottom**: DEC residuals coloured by group, with coloured vertical spans
  indicating each group's time range.

The input is the per-observation weights CSV written by
:class:`~orbitdet.observations.weighting.WeightEngine`.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from orbitdet.visualization.base import Plot

logger = logging.getLogger(__name__)


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


def _seconds_since_j2000_to_datetimes(
    seconds_since_j2000: np.ndarray,
) -> pd.DatetimeIndex:
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


class WeightGroups(Plot):
    """Plot groups and weights from a weights DataFrame.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment configuration.
    weights_df : pd.DataFrame
        Per-observation weights DataFrame as produced by
        :class:`~orbitdet.observations.weighting.WeightEngine` and saved to
        ``observation_weights.csv``.
    """

    def __init__(self, cfg: DictConfig, weights_df: pd.DataFrame):
        super().__init__(cfg)
        self.weights_df = weights_df

    def _make_figure(self):
        df = self.weights_df
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No weight data", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        # Convert time
        times = _seconds_since_j2000_to_datetimes(df["time"].values)

        # Load plotting configuration
        plot_cfg = _cfg_get(self.cfg, "weight_groups", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=16.54)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=12.0)
        cmap_name = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=12)
        alpha = _cfg_get(plot_cfg, "styling", "alpha", default=0.6)
        span_alpha = _cfg_get(plot_cfg, "styling", "span_alpha", default=0.08)

        # Identify unique groups for colouring
        unique_groups = sorted(df["group_id"].unique())
        cmap = plt.get_cmap(cmap_name, len(unique_groups))
        group_to_color = {g: cmap(i) for i, g in enumerate(unique_groups)}

        # Compute per-group time spans
        group_spans = {}
        for g in unique_groups:
            g_times = df.loc[df["group_id"] == g, "time"].values
            group_spans[g] = (
                _seconds_since_j2000_to_datetimes(np.array([g_times.min()])),
                _seconds_since_j2000_to_datetimes(np.array([g_times.max()])),
            )

        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        # ---- Panel 0: RA residuals with group spans ----
        ax = axs[0]
        for group_id in unique_groups:
            mask = df["group_id"] == group_id
            ax.scatter(
                times[mask],
                df.loc[mask, "ra_residual_arcsec"],
                color=group_to_color[group_id],
                s=marker_size,
                alpha=alpha,
                label=group_id,
            )
            t_start, t_end = group_spans[group_id]
            ax.axvspan(
                t_start[0], t_end[0], color=group_to_color[group_id], alpha=span_alpha, zorder=0
            )
        ax.set_ylabel("RA residual [arcsec]")
        ax.grid(True, alpha=0.3)

        # ---- Panel 1: DEC residuals with group spans ----
        ax = axs[1]
        for group_id in unique_groups:
            mask = df["group_id"] == group_id
            ax.scatter(
                times[mask],
                df.loc[mask, "dec_residual_arcsec"],
                color=group_to_color[group_id],
                s=marker_size,
                alpha=alpha,
                label=group_id,
            )
            t_start, t_end = group_spans[group_id]
            ax.axvspan(
                t_start[0], t_end[0], color=group_to_color[group_id], alpha=span_alpha, zorder=0
            )
        ax.set_ylabel("DEC residual [arcsec]")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

        # Single legend on the bottom panel
        ax.legend(
            title="Group",
            loc="upper right",
            framealpha=0.8,
            fontsize="small",
        )

        strategy_label = df["strategy"].iloc[0] if "strategy" in df.columns else "unknown"
        fig.suptitle(
            f"Weighting strategy: {strategy_label}",
            fontsize=14,
        )

        fig.tight_layout()
        return fig, axs
