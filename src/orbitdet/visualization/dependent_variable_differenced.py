import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tudatpy.dynamics import propagation as prop
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.utility import create_dependent_variable_dictionary
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.RSW_distance import add_panel_legend, color_collection, color_for
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


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap angular values into ``(-pi, pi]``.

    An angular difference is only meaningful modulo a full turn: two states
    whose periapsis arguments differ by ``2*pi - 0.01`` are 0.01 rad apart, not
    a full revolution apart.  Without wrapping, a component that crosses the
    branch cut draws as a vertical jump across the whole panel and hides the
    actual drift.

    Parameters
    ----------
    values : np.ndarray
        Angles in radians.

    Returns
    -------
    np.ndarray
        The same angles, shifted by whole turns into ``(-pi, pi]``.
    """
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


class DifferencedDependentVariables(SimulationPlot):
    """Plot the difference in a dependent variable between a reference result and one
    or more comparison results.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
    reference_result : prop.SimulationResults
        The reference propagation result.
    comparison_results : list[prop.SimulationResults]
        The propagation results to compare against the reference.
    reference_dependent_variable : SingleDependentVariableSaveSettings
        The dependent variable to extract from the reference result.
    comparison_dependent_variables : list[SingleDependentVariableSaveSettings]
        The dependent variables to extract from the comparison results.
    comparison_names : list[str] | None
        Optional legend names for the comparison results.
    """

    def __init__(
        self,
        cfg,
        reference_result,
        comparison_results: list,
        reference_dependent_variable: (
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ),
        comparison_dependent_variables: list[
            prop_setup.dependent_variable.SingleDependentVariableSaveSettings
        ],
        comparison_names: list[str] | None = None,
    ):
        super().__init__(cfg, reference_result)
        self.reference_result = self.result
        self.comparison_results = comparison_results
        self.reference_dependent_variable = reference_dependent_variable
        self.comparison_dependent_variables = comparison_dependent_variables
        self.comparison_names = comparison_names

    def _make_figure(self):
        plot_cfg = self.cfg
        reference_result = self.reference_result
        comparison_results = self.comparison_results
        reference_dependent_variable = self.reference_dependent_variable
        comparison_dependent_variables = self.comparison_dependent_variables
        comparison_names = self.comparison_names

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
            is (
                prop_setup.dependent_variable.PropagationDependentVariables.single_acceleration_norm_type
            )
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
                    "Dependent variable not found in comparison result "
                    f"{i}. Dependent variable: {dv}."
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
        fig_w, fig_h = self._figure_size(panels=number_of_plots)

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
                default_plot_title = (
                    f"Difference in {dependent_variable_name} for {associated_body}"
                )
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
                plot_cfg,
                "titles",
                f"component_{i}",
                default=f"{default_plot_title} (Component {i})",
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


class DifferencedDependentVariablesPerResult(SimulationPlot):
    """Difference two dependent variables *within* each result, one line per result.

    Where :class:`DifferencedDependentVariables` subtracts a comparison result
    from a reference result, this subtracts one saved dependent variable from
    another inside every result, then overlays the results.  That is the figure
    for "how far does the propagated state drift from the reference ephemeris,
    under each dynamical variant?" — here, the propagated Triton Keplerian state
    minus the SPICE one, for each pole model.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``dependent_variable_index`` selects the minuend and
        ``comparison_dependent_variable_index`` the subtrahend; both are read
        from every result.
    results : list[prop.SimulationResults]
        The results to difference, in plotting order.
    labels : list[str] | None
        Legend labels, one per result.  Defaults to ``Result 0``, ``Result 1``,
        ...
    """

    def __init__(
        self,
        cfg,
        results: list,
        labels: list[str] | None = None,
    ):
        if not results:
            raise ValueError("results must be a non-empty list of SimulationResults.")
        super().__init__(cfg, results[0])
        self.results = results
        self.labels = labels or [f"Result {i}" for i in range(len(results))]

        if len(self.labels) != len(self.results):
            raise ValueError(
                f"labels has {len(self.labels)} entries but {len(self.results)} "
                "result(s) were given."
            )

    def _make_figure(self):
        plot_cfg = self.cfg
        reference_index = int(_cfg_get(plot_cfg, "dependent_variable_index", default=0))
        comparison_index = int(_cfg_get(plot_cfg, "comparison_dependent_variable_index", default=1))

        # Difference the two variables inside every result, up front.
        differences = []
        for result in self.results:
            dv_dict = create_dependent_variable_dictionary(result)
            settings = list(result.ordered_dependent_variable_settings.values())
            reference_dv = settings[reference_index]
            comparison_dv = settings[comparison_index]
            differences.append(
                {
                    epoch: dv_dict[reference_dv][epoch] - dv_dict[comparison_dv][epoch]
                    for epoch in dv_dict.time_history
                }
            )

        time_history = create_dependent_variable_dictionary(self.results[0]).time_history
        number_of_plots = np.atleast_1d(differences[0][time_history[0]]).size

        # The configured size is the size of the *whole* figure, so the panels
        # share one page rather than each getting a page of its own.  This is
        # why ``panels`` is not passed: it multiplies the height, which is the
        # right behaviour for a figure that grows with its panel count but not
        # for a fixed-page figure like this one.
        fig_w, fig_h = self._figure_size()
        fig, axes = plt.subplots(number_of_plots, 1, figsize=(fig_w, fig_h))
        if number_of_plots == 1:
            axes = np.array([axes])

        # Titles and labels, with the same per-component conventions as
        # DifferencedDependentVariables.
        settings = list(self.results[0].ordered_dependent_variable_settings.values())
        reference_dv = settings[reference_index]
        dependent_variable_name = (
            reference_dv.dependent_variable_type.name.replace("_", " ").replace(" type", "").title()
        )
        associated_body = reference_dv.associated_body
        secondary_body = reference_dv.secondary_body

        default_suptitle = (
            f"Difference in {dependent_variable_name} for {associated_body}"
            if secondary_body is None
            else f"Difference in {dependent_variable_name} for "
            f"{associated_body} w.r.t. {secondary_body}"
        )
        suptitle = _cfg_get(plot_cfg, "titles", "suptitle", default=default_suptitle)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(
                    dependent_variable_name=dependent_variable_name,
                    associated_body=associated_body,
                    secondary_body=secondary_body,
                )
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Difference")
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default=x_label)
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default=None)

        # A single string broadcasts to every component; a list maps one label
        # per component.
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

        colors = color_collection(plot_cfg)

        # Components whose difference is an angle, and so only meaningful
        # modulo a full turn.  Named by panel index in the figure config, e.g.
        # ``axes.wrap_components: [3, 4, 5]`` for the three Keplerian angles.
        wrap_components = {
            int(index)
            for index in (_cfg_get(plot_cfg, "axes", "wrap_components", default=[]) or [])
        }
        unknown = wrap_components - set(range(number_of_plots))
        if unknown:
            raise ValueError(
                f"axes.wrap_components names panel(s) {sorted(unknown)} but the "
                f"figure has {number_of_plots} panel(s)."
            )

        # A panel title and a y-axis label that say the same thing is noise.
        # ``axes.show_y_label: false`` drops the label and leaves the title,
        # which is the usual choice when the title already names the component.
        show_y_label = bool(_cfg_get(plot_cfg, "axes", "show_y_label", default=True))

        for i in range(number_of_plots):
            # Default the panel title to its y-axis label: for a multi-component
            # variable the label already names the component ("Δ Semi-major
            # Axis [m]"), which is far more useful than "Component 0".
            component_title = _cfg_get(
                plot_cfg,
                "titles",
                f"component_{i}",
                default=y_labels[i],
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

            for index, (difference, label) in enumerate(zip(differences, self.labels)):
                values = np.array([np.atleast_1d(difference[epoch])[i] for epoch in time_history])
                if i in wrap_components:
                    values = wrap_to_pi(values)
                axes[i].plot(
                    x_data,
                    values,
                    label=label,
                    color=color_for(colors, index),
                )
            axes[i].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            axes[i].set_title(component_title)
            if show_y_label:
                axes[i].set_ylabel(y_labels[i])
            axes[i].grid(True, alpha=0.3)

        for ax, hy_label in zip(axes, hover_y_labels):
            ax.format_coord = _make_hover_formatter(hover_x_label, hy_label)

        # One legend for the whole figure, inside a panel so it cannot collide
        # with the x-axis tick labels.
        add_panel_legend(axes, plot_cfg, len(self.labels))

        fig.suptitle(suptitle)

        # Line the y-axis labels up.  Each panel's tick labels have their own
        # width (a panel reading 1e-6 needs more room than one reading 1e-3), so
        # without this the labels sit at different horizontal positions.
        fig.align_ylabels(axes)

        # Leave room for the suptitle, which otherwise overlaps the first
        # panel's title.
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        return fig, axes
