import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
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


def plot_differenced_dependent_variables(
    cfg: DictConfig,
    result_A: prop.SimulationResults,
    result_B: prop.SimulationResults,
    dependent_variable_A: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
    dependent_variable_B: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
) -> tuple[plt.Figure, np.ndarray]:
    # Check if result objects are of the correct type
    if not issubclass(result_A.__class__, prop.SimulationResults) or not issubclass(
        result_B.__class__, prop.SimulationResults
    ):
        raise TypeError(
            "Both result_A and result_B must be of (derived) type 'SimulationResults'. "
            f"Got {type(result_A)} and {type(result_B)}."
        )

    # Check if dependent variable types are the same
    if dependent_variable_A.dependent_variable_type != dependent_variable_B.dependent_variable_type:
        raise ValueError(
            "Dependent variable types must be the same for differencing. "
            f"Got {dependent_variable_A.dependent_variable_type} and "
            f"{dependent_variable_B.dependent_variable_type}."
        )
    if (
        dependent_variable_A.dependent_variable_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
        or dependent_variable_A.dependent_variable_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_type
    ):
        if (
            dependent_variable_A.acceleration_model_type
            is not dependent_variable_B.acceleration_model_type
        ):
            raise ValueError(
                "Acceleration model types must be the same for differencing "
                "when dependent variable type is single_acceleration_norm_type "
                "or single_acceleration_type. "
                f"Got {dependent_variable_A.acceleration_model_type} and "
                f"{dependent_variable_B.acceleration_model_type}."
            )

    # Create dependent variable dictionaries for both results
    dependent_variable_dict_A = create_dependent_variable_dictionary(result_A)
    dependent_variable_dict_B = create_dependent_variable_dictionary(result_B)

    # Check if dependent variable is available in both results objects
    try:
        dependent_variable_dict_A[dependent_variable_A]
        dependent_variable_dict_B[dependent_variable_B]
    except KeyError as e:
        raise ValueError(
            "Dependent variable not found in one of the results objects. "
            f"Dependent variable A: {dependent_variable_A}, "
            f"Dependent variable B: {dependent_variable_B}."
        ) from e

    # Compute the difference
    difference_dict = {
        epoch: dependent_variable_dict_A[dependent_variable_A][epoch]
        - dependent_variable_dict_B[dependent_variable_B][epoch]
        for epoch in dependent_variable_dict_A.time_history
    }
    # Check how large the dependent variable is to determine how many plots to make
    number_of_plots = difference_dict[dependent_variable_dict_A.time_history[0]].size

    # Load plotting configuration
    plot_cfg = _cfg_get(cfg, "dependent_variable_differenced", default=None)['dependent_variable_differenced']
    if plot_cfg is None:
        plot_cfg = cfg
    fig_w = _cfg_get(plot_cfg, "figure", "width", default=10)
    fig_h = _cfg_get(plot_cfg, "figure", "height", default=5 * number_of_plots)

    fig, axes = plt.subplots(number_of_plots, 1, figsize=(fig_w, fig_h))

    # Ensure axes is always iterable
    if number_of_plots == 1:
        axes = np.array([axes])

    # Build default title components
    dependent_variable_name_raw = dependent_variable_A.dependent_variable_type.name
    dependent_variable_name = (
        dependent_variable_name_raw.replace("_", " ").replace(" type", "").title()
    )
    associated_body = dependent_variable_A.associated_body
    secondary_body = dependent_variable_A.secondary_body

    if (
        dependent_variable_A.dependent_variable_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
        or dependent_variable_A.dependent_variable_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_type
    ):
        acceleration_model_type = dependent_variable_A.acceleration_model_type.name
        default_plot_title = (
            f"Difference in {dependent_variable_name} ({acceleration_model_type}) "
            f"for {associated_body} w.r.t. {secondary_body}"
        )
    else:
        if secondary_body is None:
            default_plot_title = f"Difference in {dependent_variable_name} for {associated_body}"
        else:
            default_plot_title = (
                f"Difference in {dependent_variable_name} for "
                f"{associated_body} w.r.t. {secondary_body}"
            )

    # Configurable labels and titles
    x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
    y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Difference")
    hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
    hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=None)

    # Support per-component y-labels: a single string is broadcast to all components,
    # while a list of strings maps one label per component.
    if isinstance(y_label, str):
        y_labels = [y_label] * number_of_plots
    else:
        y_labels = list(y_label)
        if len(y_labels) < number_of_plots:
            y_labels += [y_labels[-1]] * (number_of_plots - len(y_labels))

    if hover_y_label is None:
        hover_y_labels = y_labels
    elif isinstance(hover_y_label, str):
        hover_y_labels = [hover_y_label] * number_of_plots
    else:
        hover_y_labels = list(hover_y_label)
        if len(hover_y_labels) < number_of_plots:
            hover_y_labels += [hover_y_labels[-1]] * (number_of_plots - len(hover_y_labels))

    suptitle = _cfg_get(plot_cfg, "titles", "suptitle", default=default_plot_title)
    # Allow templates like "{dependent_variable_name} for {associated_body}" in config
    try:
        if isinstance(suptitle, str):
            suptitle = suptitle.format(
                dependent_variable_name=dependent_variable_name,
                associated_body=associated_body,
                secondary_body=secondary_body,
            )
    except Exception:
        pass

    for i in range(number_of_plots):
        # Per-component title (configurable), with fallback to default
        component_title = _cfg_get(
            plot_cfg, "titles", f"component_{i}", default=f"{default_plot_title} (Component {i})"
        )
        try:
            if isinstance(component_title, str):
                component_title = component_title.format(
                    dependent_variable_name=dependent_variable_name,
                    associated_body=associated_body,
                    secondary_body=secondary_body,
                    component=i,
                )
        except Exception:
            pass

        if i < number_of_plots - 1:
            x_data = dependent_variable_dict_A.time_history
            axes[i].tick_params(axis="x", which="both", labelbottom=False)
        else:
            x_data = _seconds_since_j2000_to_datetimes(
                np.asarray(dependent_variable_dict_A.time_history)
            )
            axes[i].set_xlabel(x_label)
            _configure_datetime_axis(axes[i])
        axes[i].plot(
            x_data,
            [difference_dict[epoch][i] for epoch in dependent_variable_dict_A.time_history],
        )
        axes[i].set_title(component_title)
        axes[i].set_ylabel(y_labels[i])
        axes[i].grid()

    # Hover formatter (per-axis when multiple y-labels are provided)
    for ax, hy_label in zip(axes, hover_y_labels):
        ax.format_coord = _make_hover_formatter(hover_x_label, hy_label)

    fig.suptitle(suptitle)
    fig.set_tight_layout(True)

    # Optionally save to file
    out = _cfg_get(plot_cfg, "output_file", default=None)
    if out:
        fig.savefig(out)

    return fig, axes
