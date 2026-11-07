import matplotlib.pyplot as plt
import numpy as np

from orbitdet.visualization.estimation_base import EstimationPlot, _cfg_get


class ParameterHistoryPerIteration(EstimationPlot):
    """Plot the parameter values against iteration number for each estimated parameter.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        Supported keys: ``parameter_names``, ``normalize``, ``y_label``,
        ``figure.width``, ``figure.height``.
    estimation_output : est_an.EstimationOutput
        The estimation output whose ``parameter_history`` is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        estimation_output = self.estimation_output

        parameter_history = estimation_output.parameter_history
        # parameter_history: rows = parameters, columns = iterations (col 0 = pre-estimation)
        n_params = parameter_history.shape[0]
        n_iterations = parameter_history.shape[1]

        # Default parameter names: Cowell propagation (position xyz, velocity xyz)
        default_names = ["x", "y", "z", "v\u2093", "v_y", "v_z"]
        parameter_names = self._parameter_names(n_params, default_names)

        # Normalize each parameter by its initial value to show relative change
        # Use column 0 (pre-estimation) as reference
        normalize = _cfg_get(plot_cfg, "normalize", default=True)
        if normalize:
            reference = parameter_history[:, 0]
            # Avoid division by zero
            reference_safe = np.where(np.abs(reference) > 0, reference, 1.0)
            plot_data = parameter_history / reference_safe[:, np.newaxis]
            y_label = _cfg_get(
                plot_cfg,
                "y_label",
                default="Relative Parameter Value (normalized to initial)",
            )
        else:
            plot_data = parameter_history
            y_label = _cfg_get(plot_cfg, "y_label", default="Parameter Value")

        fig_width, fig_height = self._figure_size(10.0, 6.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for i in range(n_params):
            ax.plot(
                range(n_iterations),
                plot_data[i],
                marker="o",
                markersize=3,
                label=parameter_names[i],
            )

        ax.set_xlabel("Iteration (0 = pre-estimation)")
        ax.set_ylabel(y_label)
        ax.set_title("Parameter History per Iteration")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax.grid()
        plt.tight_layout()
        return fig, ax
