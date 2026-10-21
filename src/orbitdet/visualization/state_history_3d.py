"""3D trajectory plot of a propagation result's state history.

Plots the Cartesian position history (x, y, z) of a
:class:`tudatpy.dynamics.propagation.SimulationResults` object on a 3D axis.
"""

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tudatpy.dynamics import propagation as prop

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


class StateHistory3D(Plot):
    """Plot the Cartesian position history of a propagation result in 3D.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration. Optional settings are read from the
        ``state_history_3d`` block (see ``conf/figures/state_history_3d.yaml``).
    result : prop.SimulationResults
        The propagation result whose ``state_history`` is plotted. An object
        exposing a ``propagation_results`` attribute (e.g. an estimation output)
        is also accepted.
    """

    def __init__(self, cfg: DictConfig, result: prop.SimulationResults):
        super().__init__(cfg, name="state_history_3d")
        self.result = result

    def _make_figure(self):
        cfg = self.cfg
        result = self.result

        # Accept either a SimulationResults directly or an object with a
        # .propagation_results attribute (e.g. EstimationOutput)
        if hasattr(result, "propagation_results"):
            sim_results = result.propagation_results
        else:
            sim_results = result

        state_history = sim_results.state_history
        if not state_history:
            raise ValueError("SimulationResults.state_history is empty; nothing to plot.")

        states = np.array([np.asarray(state).flatten() for state in state_history.values()])
        if states.shape[1] < 3:
            raise ValueError(
                f"state_history entries must have at least 3 components (position). "
                f"Got {states.shape[1]}."
            )

        positions = states[:, :3]

        plot_cfg = _cfg_get(cfg, "state_history_3d", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=10)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8)

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="x [m]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="y [m]")
        z_label = _cfg_get(plot_cfg, "axes", "z_label", default="z [m]")
        suptitle = _cfg_get(
            plot_cfg, "titles", "suptitle", default="Propagated Trajectory (Inertial Frame)"
        )

        color = _cfg_get(plot_cfg, "line", "color", default="tab:blue")
        linewidth = _cfg_get(plot_cfg, "line", "width", default=1.5)
        marker_start = _cfg_get(plot_cfg, "markers", "start", default="o")
        marker_end = _cfg_get(plot_cfg, "markers", "end", default="s")
        marker_size = _cfg_get(plot_cfg, "markers", "size", default=40)
        show_start_end = _cfg_get(plot_cfg, "markers", "enabled", default=True)

        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=linewidth,
            label="Trajectory",
        )

        if show_start_end:
            ax.scatter(
                *positions[0],
                color="green",
                marker=marker_start,
                s=marker_size,
                label="Start",
                depthshade=False,
            )
            ax.scatter(
                *positions[-1],
                color="red",
                marker=marker_end,
                s=marker_size,
                label="End",
                depthshade=False,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title(suptitle)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio so the orbit is not distorted
        if _cfg_get(plot_cfg, "equal_aspect", default=True):
            max_range = (positions.max(axis=0) - positions.min(axis=0)).max() / 2.0
            if max_range > 0:
                mid = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        fig.set_tight_layout(True)

        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, ax
