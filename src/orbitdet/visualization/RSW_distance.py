import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary


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


def plot_RSW_distance(
    cfg: DictConfig,
    result: prop.SimulationResults,
    position_dependent_variable: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
    central_body: str = "Neptune",
):
    """
    Decomposes the relative position vector between two bodies into its Radial
    (R), Transverse (S), and Cross-track (W) components in the RSW frame.

    The RSW frame is defined epoch-by-epoch from the *propagated* body's
    inertial state relative to the central body. The relative position vector
    (e.g. from ``relative_position("Triton Spice", "Triton")``) is rotated
    from the inertial frame into the RSW frame at each epoch, producing a
    3-panel plot of R, S, W.

    Parameters
    ----------
    cfg : DictConfig
        Top-level Hydra configuration. Supports an ``RSW_distance`` sub-key
        with the following optional settings:

        - ``central_body`` (str): name of the central body (default "Neptune").
        - ``figure.width`` (float): figure width in inches (default 12).
        - ``figure.height`` (float): figure height in inches (default 8).
        - ``output_file`` (str, optional): path to save the figure.

    result
        Simulation results containing ``state_history`` and
        ``dependent_variable_history`` (e.g. ``SimulationResults`` or
        ``EstimationOutput``).

    position_dependent_variable : SingleDependentVariableSaveSettings
        Dependent variable settings for the relative position vector, typically
        ``prop_setup.dependent_variable.relative_position("Triton Spice", "Triton")``.

    central_body : str
        Name of the central body used to define the RSW frame. The propagated
        body's inertial state relative to this body determines the R/S/W axes
        at each epoch.  Default is "Neptune".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of 3 Axes objects: [R, S, W].

    Raises
    ------
    ValueError
        If a dependent variable is not found in the results.
    """
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

    # --- plotting configuration ---
    plot_cfg = _cfg_get(cfg, "RSW_distance", default={})
    fig_w = _cfg_get(plot_cfg, "figure", "width", default=12)
    fig_h = _cfg_get(plot_cfg, "figure", "height", default=8)

    component_names = ["R", "S", "W"]
    component_units = ["m", "m", "m"]

    # --- build the figure ---
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(times, rsw_components[:, i])
        ax.set_ylabel(f"{component_names[i]} [{component_units[i]}]")
        ax.grid(True, alpha=0.3)
        ax.format_coord = _make_hover_formatter("Epoch", component_names[i])

    _configure_datetime_axis(axes[-1])
    axes[-1].set_xlabel("Epoch")

    # --- title ---
    body1 = position_dependent_variable.associated_body
    body2 = position_dependent_variable.secondary_body
    central = _cfg_get(plot_cfg, "central_body", default=central_body)

    fig.suptitle(
        f"Relative State of {body1} w.r.t. {body2} in RSW Frame (centered on {central})",
        fontsize=14,
    )
    fig.align_ylabels(axes)
    fig.set_tight_layout(True)

    # --- optional file output ---
    out = _cfg_get(plot_cfg, "output_file", default=None)
    if out:
        fig.savefig(out)

    return fig, axes
