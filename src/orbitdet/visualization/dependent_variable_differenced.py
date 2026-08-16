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
    reference_result: prop.SimulationResults,
    comparison_results: list[prop.SimulationResults],
    reference_dependent_variable: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
    comparison_dependent_variables: list[
        prop_setup.dependent_variable.SingleDependentVariableSaveSettings
    ],
    comparison_names: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the difference in a dependent variable between a reference result and one
    or more comparison results.

    Args:
        cfg: Configuration dictionary for plotting settings.
        reference_result (SimulationResults): The reference SimulationResults object.
        comparison_results (list[SimulationResults]): A list of SimulationResults objects
            to compare against the reference.
        reference_dependent_variable (SingleDependentVariableSaveSettings): The dependent
            variable to extract from the reference result.
        comparison_dependent_variables (list[SingleDependentVariableSaveSettings]): A list
            of dependent variables to extract from the comparison results.
        comparison_names (list[str] | None, optional)): Optional list of names for the
            comparison results, used in the legend. If not provided, default names will
            be generated.
    """

    # Check if result objects are of the correct type
    if not issubclass(reference_result.__class__, prop.SimulationResults):
        raise TypeError(
            "reference_result must be of (derived) type 'SimulationResults'. "
            f"Got {type(reference_result)}."
        )
    if not isinstance(comparison_results, list) or len(comparison_results) == 0:
        raise TypeError("comparison_results must be a non-empty list of SimulationResults.")
    for i, r in enumerate(comparison_results):
        if not issubclass(r.__class__, prop.SimulationResults):
            raise TypeError(
                f"comparison_results[{i}] must be of (derived) type 'SimulationResults'. "
                f"Got {type(r)}."
            )

    if (
        not isinstance(comparison_dependent_variables, list)
        or len(comparison_dependent_variables) == 0
    ):
        raise TypeError(
            "comparison_dependent_variables must be a non-empty list of "
            "SingleDependentVariableSaveSettings."
        )
    if len(comparison_results) != len(comparison_dependent_variables):
        raise ValueError(
            "comparison_results and comparison_dependent_variables must have the same length. "
            f"Got {len(comparison_results)} and {len(comparison_dependent_variables)}."
        )

    # Check if all dependent variable types are the same
    ref_dv_type = reference_dependent_variable.dependent_variable_type
    for i, dv in enumerate(comparison_dependent_variables):
        if dv.dependent_variable_type != ref_dv_type:
            raise ValueError(
                "All dependent variable types must be the same for differencing. "
                f"Reference type: {ref_dv_type}, "
                f"comparison[{i}] type: {dv.dependent_variable_type}."
            )

    is_acceleration_type = (
        ref_dv_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
        or ref_dv_type
        is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_type
    )
    if is_acceleration_type:
        ref_acc_type = reference_dependent_variable.acceleration_model_type
        for i, dv in enumerate(comparison_dependent_variables):
            if dv.acceleration_model_type is not ref_acc_type:
                raise ValueError(
                    "Acceleration model types must be the same for differencing "
                    "when dependent variable type is single_acceleration_norm_type "
                    "or single_acceleration_type. "
                    f"Reference type: {ref_acc_type}, comparison[{i}] type: "
                    f"{dv.acceleration_model_type}."
                )

    # Validate comparison_names
    if comparison_names is not None and len(comparison_names) != len(comparison_results):
        raise ValueError(
            "comparison_names must have the same length as comparison_results. "
            f"Got {len(comparison_names)} and {len(comparison_results)}."
        )
    if comparison_names is None:
        comparison_names = [f"Comparison {i}" for i in range(len(comparison_results))]

    # Create dependent variable dictionaries for all results
    reference_dv_dict = create_dependent_variable_dictionary(reference_result)
    comparison_dv_dicts = [create_dependent_variable_dictionary(r) for r in comparison_results]

    # Check if dependent variable is available in all results objects
    try:
        reference_dv_dict[reference_dependent_variable]
    except KeyError as e:
        raise ValueError(
            "Dependent variable not found in reference result. "
            f"Reference dependent variable: {reference_dependent_variable}."
        ) from e
    for i, (dvd, dv) in enumerate(zip(comparison_dv_dicts, comparison_dependent_variables)):
        try:
            dvd[dv]
        except KeyError as e:
            raise ValueError(
                f"Dependent variable not found in comparison result {i}. Dependent variable: {dv}."
            ) from e

    # Compute the differences (one per comparison)
    difference_dicts = []
    for dvd, dv in zip(comparison_dv_dicts, comparison_dependent_variables):
        diff = {
            epoch: reference_dv_dict[reference_dependent_variable][epoch] - dvd[dv][epoch]
            for epoch in reference_dv_dict.time_history
        }
        difference_dicts.append(diff)

    # Check how large the dependent variable is to determine how many plots to make
    number_of_plots = difference_dicts[0][reference_dv_dict.time_history[0]].size

    # Load plotting configuration
    plot_cfg = _cfg_get(cfg, "dependent_variable_differenced", default=None)[
        "dependent_variable_differenced"
    ]
    if plot_cfg is None:
        plot_cfg = cfg
    fig_w = _cfg_get(plot_cfg, "figure", "width", default=10)
    fig_h = _cfg_get(plot_cfg, "figure", "height", default=5 * number_of_plots)

    fig, axes = plt.subplots(number_of_plots, 1, figsize=(fig_w, fig_h))

    # Ensure axes is always iterable
    if number_of_plots == 1:
        axes = np.array([axes])

    # Build default title components
    dependent_variable_name_raw = reference_dependent_variable.dependent_variable_type.name
    dependent_variable_name = (
        dependent_variable_name_raw.replace("_", " ").replace(" type", "").title()
    )
    associated_body = reference_dependent_variable.associated_body
    secondary_body = reference_dependent_variable.secondary_body

    if is_acceleration_type:
        acceleration_model_type = reference_dependent_variable.acceleration_model_type.name
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

    # Plot each component with all comparisons overlaid
    time_history = reference_dv_dict.time_history
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
            x_data = time_history
            axes[i].tick_params(axis="x", which="both", labelbottom=False)
        else:
            x_data = _seconds_since_j2000_to_datetimes(np.asarray(time_history))
            axes[i].set_xlabel(x_label)
            _configure_datetime_axis(axes[i])

        # Plot each comparison as a separate line
        for diff_dict, name in zip(difference_dicts, comparison_names):
            axes[i].plot(
                x_data,
                [np.atleast_1d(diff_dict[epoch])[i] for epoch in time_history],
                label=name,
            )
        axes[i].set_title(component_title)
        axes[i].set_ylabel(y_labels[i])
        axes[i].grid()
        axes[i].legend()

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
