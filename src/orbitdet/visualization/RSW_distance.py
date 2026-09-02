import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary
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


class RSWDistance(Plot):
    """Decompose the relative position vector between two bodies into its Radial (R),
    Transverse (S), and Cross-track (W) components in the RSW frame.
    """

    def __init__(
        self,
        cfg: DictConfig,
        result: prop.SimulationResults,
        position_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
        central_body: str = "Neptune",
    ):
        super().__init__(cfg)
        self.result = result
        self.position_dependent_variable = position_dependent_variable
        self.central_body = central_body

    def _make_figure(self):
        cfg = self.cfg
        result = self.result
        position_dependent_variable = self.position_dependent_variable
        central_body = self.central_body

        # Accept either a SimulationResults directly or an object with a
        # .propagation_results attribute (e.g. EstimationOutput)
        if hasattr(result, "propagation_results"):
            sim_results = result.propagation_results
        else:
            sim_results = result

        dep_var_dict = create_dependent_variable_dictionary(sim_results)

        # Check if dependent variable is available in results object
        try:
            dep_var_dict[position_dependent_variable]
        except KeyError as e:
            raise ValueError(
                f"Dependent variable not found in results object. "
                f"Dependent variable: {position_dependent_variable}."
            ) from e

        pos_value_dict = dep_var_dict[position_dependent_variable]

        epochs = list(dep_var_dict.time_history)
        n_epochs = len(epochs)
        times = _seconds_since_j2000_to_datetimes(np.asarray(epochs))

        # --- get propagated state history ---
        state_history = sim_results.state_history
        if len(state_history) != n_epochs:
            raise ValueError(
                f"Mismatch between state history ({len(state_history)} entries) and "
                f"dependent variable history ({n_epochs} entries)."
            )

        # --- decompose relative position into RSW components ---
        rsw_components = np.zeros((n_epochs, 3))
        for i, epoch in enumerate(epochs):
            inertial_state = np.asarray(state_history[epoch])
            rot = fc.inertial_to_rsw_rotation_matrix(inertial_state)
            rel_pos = np.asarray(pos_value_dict[epoch]).flatten()
            rsw_components[i, :] = rot @ rel_pos

        rsw_norm = np.linalg.norm(rsw_components, axis=1)

        plot_cfg = _cfg_get(cfg, "RSW_distance", default=None)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=12)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8)

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
            default=f"RSW Distance for {position_dependent_variable.associated_body}",
        )
        central = _cfg_get(plot_cfg, "central_body", default=central_body)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(
                    body1=position_dependent_variable.associated_body,
                    body2=position_dependent_variable.secondary_body,
                    central_body=central,
                )
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Distance [m]")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=y_label)

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        series = [
            (rsw_components[:, 0], titles["r"]),
            (rsw_components[:, 1], titles["s"]),
            (rsw_components[:, 2], titles["w"]),
            (rsw_norm, titles["norm"]),
        ]
        for ax, (values, title) in zip(axes, series):
            ax.plot(times, values)
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        for ax in axes[2:]:
            _configure_datetime_axis(ax)
            ax.set_xlabel(x_label)

        fig.suptitle(
            suptitle,
            fontsize=14,
        )
        fig.align_ylabels(axes)
        fig.set_tight_layout(True)

        out = _cfg_get(plot_cfg, "output_file", default=None)
        if out:
            fig.savefig(out)

        return fig, axes
