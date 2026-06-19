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


def plot_differenced_dependent_variables(
    result_A: prop.SimulationResults,
    result_B: prop.SimulationResults,
    dependent_variable_A: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
    dependent_variable_B: prop_setup.dependent_variable.SingleDependentVariableSaveSettings,
):
    # Check if result objects are of the correct type
    if not issubclass(result_A.__class__, prop.SimulationResults) or not issubclass(result_B.__class__, prop.SimulationResults):
        raise TypeError(
            "Both result_A and result_B must be of (derived) type 'SimulationResults'. "
            f"Got {type(result_A)} and {type(result_B)}."
        )
    
    # Check if dependent variable types are the same
    if dependent_variable_A.dependent_variable_type != dependent_variable_B.dependent_variable_type:
        raise ValueError(
            "Dependent variable types must be the same for differencing. "
            f"Got {dependent_variable_A.dependent_variable_type} and {dependent_variable_B.dependent_variable_type}."
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
                "Acceleration model types must be the same for differencing when dependent variable "
                "type is single_acceleration_norm_type or single_acceleration_type. "
                f"Got {dependent_variable_A.acceleration_model_type} and {dependent_variable_B.acceleration_model_type}."
            )

    # Create dependent variable dictionaries for both results
    dependent_variable_dict_A = create_dependent_variable_dictionary(result_A)
    dependent_variable_dict_B = create_dependent_variable_dictionary(result_B)

    # Check if dependent variable is available in both results objects
    try:
        dependent_variable_dict_A[dependent_variable_A]
        dependent_variable_dict_B[dependent_variable_B]

    except KeyError as e :
        raise ValueError(
            "Dependent variable not found in one of the results objects. "
            f"Dependent variable A: {dependent_variable_A}, Dependent variable B: {dependent_variable_B}."
        ) from e

    # Compute the difference
    difference_dict = {
        epoch: dependent_variable_dict_A[dependent_variable_A][epoch]
        - dependent_variable_dict_B[dependent_variable_B][epoch]
        for epoch in dependent_variable_dict_A.time_history
    }
    # Check how large the dependent variable is to determine how many plots to make
    number_of_plots = difference_dict[dependent_variable_dict_A.time_history[0]].size

    # Plot
    fig, axes = plt.subplots(number_of_plots, 1, figsize=(10, 5 * number_of_plots))
    for i in range(number_of_plots):
        dependent_variable_name_raw = dependent_variable_A.dependent_variable_type.name
        dependent_variable_name = dependent_variable_name_raw.replace("_", " ").replace(" type", "").title()
        associated_body = dependent_variable_A.associated_body
        secondary_body = dependent_variable_A.secondary_body

        if (
            dependent_variable_A.dependent_variable_type
            is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
            or dependent_variable_A.dependent_variable_type
            is prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_type
        ):
            acceleration_model_type = dependent_variable_A.acceleration_model_type.name
            plot_title = (
                f"Difference in {dependent_variable_name} ({acceleration_model_type}) "
                f"for {associated_body} w.r.t. {secondary_body}"
            )
        else:
            if secondary_body is None:
                plot_title = f"Difference in {dependent_variable_name} for {associated_body}"
            else:
                plot_title = f"Difference in {dependent_variable_name} for {associated_body} w.r.t. {secondary_body}"

        if i < number_of_plots - 1:
            x_data = dependent_variable_dict_A.time_history
            axes[i].tick_params(axis="x", which="both", labelbottom=False)
        else:
            x_data = _seconds_since_j2000_to_datetimes(
                np.asarray(dependent_variable_dict_A.time_history)
            )
            axes[i].set_xlabel("Epoch")
            _configure_datetime_axis(axes[i])
        axes[i].plot(
            x_data,
            [difference_dict[epoch][i] for epoch in dependent_variable_dict_A.time_history],
        )
        axes[i].set_title(f"{plot_title} (Component {i})")
        axes[i].set_ylabel("Difference")
        axes[i].grid()
    # Return the plot
    return fig, axes
