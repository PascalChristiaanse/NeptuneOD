import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.simulation_base import SimulationPlot, _cfg_get


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


class RSWDistance(SimulationPlot):
    """Decompose the relative position vector between two bodies into its Radial (R),
    Transverse (S), and Cross-track (W) components in the RSW frame.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``central_body`` is read from here (default ``"Neptune"``).
    result : prop.SimulationResults
        The propagation result to plot.
    position_dependent_variable : SingleDependentVariableSaveSettings
        The relative-position dependent variable to decompose.
    """

    def __init__(
        self,
        cfg: DictConfig,
        result: prop.SimulationResults,
        position_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
    ):
        super().__init__(cfg, result)
        self.position_dependent_variable = position_dependent_variable

    def _make_figure(self):
        plot_cfg = self.cfg
        sim_results = self.result
        position_dependent_variable = self.position_dependent_variable
        central_body = _cfg_get(plot_cfg, "central_body", default="Neptune")

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
        skipped_epochs = 0
        for i, epoch in enumerate(epochs):
            inertial_state = np.asarray(state_history[epoch])
            # Check for degenerate state where position and velocity are parallel
            radius = inertial_state[:3]
            velocity = inertial_state[3:]
            cross_norm = np.linalg.norm(np.cross(radius, velocity))
            if cross_norm == 0.0:
                skipped_epochs += 1
                rsw_components[i, :] = np.nan
                continue
            rot = fc.inertial_to_rsw_rotation_matrix(inertial_state)
            rel_pos = np.asarray(pos_value_dict[epoch]).flatten()
            rsw_components[i, :] = rot @ rel_pos

        if skipped_epochs > 0:
            import warnings

            warnings.warn(
                f"Skipped {skipped_epochs} / {n_epochs} epoch(s) where "
                f"position and velocity are parallel (cannot form RSW frame)."
            )

        rsw_norm = np.linalg.norm(rsw_components, axis=1)

        fig_w, fig_h = self._figure_size(12, 8)

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

        return fig, axes
