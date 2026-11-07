import matplotlib.pyplot as plt

from orbitdet.visualization.estimation_base import EstimationPlot, _cfg_get


class ParameterCorrelationHeatmap(EstimationPlot):
    """Plot a heatmap of the estimated parameter correlations from the estimation output.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        Supported keys: ``parameter_names``, ``annotation``, ``decimal_places``,
        ``figure.width``, ``figure.height``.
    estimation_output : est_an.EstimationOutput
        The estimation output whose correlation matrix is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        estimation_output = self.estimation_output

        correlation_matrix = estimation_output.correlations
        n_params = correlation_matrix.shape[0]

        # Default parameter names: Cowell propagation (position xyz, velocity xyz)
        default_names = ["x", "y", "z", "v\u2093", "v_y", "v_z"]
        parameter_names = self._parameter_names(n_params, default_names)

        fig_width, fig_height = self._figure_size(8.0, 8.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cax = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        fig.colorbar(cax, ax=ax, label="Correlation")

        # Set tick labels
        ax.set_xticks(range(n_params))
        ax.set_yticks(range(n_params))
        ax.set_xticklabels(parameter_names, rotation=45, ha="right")
        ax.set_yticklabels(parameter_names)

        # Annotate cells with correlation values
        annotate = _cfg_get(plot_cfg, "annotation", default=True)
        if annotate:
            decimal_places = _cfg_get(plot_cfg, "decimal_places", default=2)
            for i in range(n_params):
                for j in range(n_params):
                    value = correlation_matrix[i, j]
                    # Choose text color based on background intensity
                    text_color = "white" if abs(value) > 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.{decimal_places}f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                    )

        ax.set_title("Parameter Correlation Heatmap")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Parameter")

        plt.tight_layout()
        return fig, ax
