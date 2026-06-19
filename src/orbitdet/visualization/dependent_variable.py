import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary


def _seconds_since_j2000_to_datetimes(seconds_since_j2000):
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _configure_datetime_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_dependent_variable(
    result: prop.SimulationResults,
    dependent_variable: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
):
    # Check if result object is of the correct type
    if not issubclass(result.__class__, prop.SimulationResults):
        raise TypeError(
            f"result must be of (derived) type 'SimulationResults'. Got {type(result)}."
        )

    # Create dependent variable dictionary
    dependent_variable_dict = create_dependent_variable_dictionary(result)

    # Check if dependent variable is available in results object
    try:
        dependent_variable_dict[dependent_variable]
    except KeyError as e:
        raise ValueError(
            f"Dependent variable not found in results object. Dependent variable: {dependent_variable}."
        ) from e

    # Extract the dependent variable values
    value_dict = dependent_variable_dict[dependent_variable]

    # Check how large the dependent variable is to determine how many plots to make
    number_of_plots = value_dict[dependent_variable_dict.time_history[0]].size

    # Plot
    fig, axes = plt.subplots(number_of_plots, 1, figsize=(10, 5 * number_of_plots))
    for i in range(number_of_plots):
        dependent_variable_name_raw = dependent_variable.dependent_variable_type.name
        dependent_variable_name = dependent_variable_name_raw.replace("_", " ").replace(" type", "").title()
        associated_body = dependent_variable.associated_body
        secondary_body = dependent_variable.secondary_body

        if (
            dependent_variable.dependent_variable_type
            is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
            or dependent_variable.dependent_variable_type
            is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_type
        ):
            acceleration_model_type = dependent_variable.acceleration_model_type.name
            plot_title = (
                f"{dependent_variable_name} ({acceleration_model_type}) "
                f"for {associated_body} w.r.t. {secondary_body}"
            )
        else:
            if secondary_body is None:
                plot_title = f"{dependent_variable_name} for {associated_body}"
            else:
                plot_title = f"{dependent_variable_name} for {associated_body} w.r.t. {secondary_body}"

        if i < number_of_plots - 1:
            x_data = dependent_variable_dict.time_history
            axes[i].tick_params(axis="x", which="both", labelbottom=False)
        else:
            x_data = _seconds_since_j2000_to_datetimes(
                np.asarray(dependent_variable_dict.time_history)
            )
            axes[i].set_xlabel("Epoch")
            _configure_datetime_axis(axes[i])
        axes[i].plot(
            x_data,
            [value_dict[epoch][i] for epoch in dependent_variable_dict.time_history],
        )
        axes[i].set_title(f"{plot_title} (Component {i})")
        axes[i].set_ylabel("Value")
        axes[i].grid()
    # Return the plot
    return fig, axes
