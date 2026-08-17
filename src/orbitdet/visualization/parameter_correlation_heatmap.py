import matplotlib.pyplot as plt
from tudatpy.estimation import estimation_analysis as est_an


def plot_parameter_correlation_heatmap(cfg, estimation_output: est_an.EstimationOutput):
    """
    Plots a heatmap of the estimated parameter correlations from the estimation output.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing optional 'parameter_correlation_heatmap' settings.
        Supported keys:
            parameter_names : list of str
                Names for each parameter (used as tick labels and annotation).
                Defaults to Cowell propagation parameter names.
            figure : dict
                'width' and 'height' for figure size.
            annotation : bool
                Whether to annotate cells with correlation values (default True).
            decimal_places : int
                Number of decimal places for annotations (default 2).

    estimation_output : est_an.EstimationOutput
        The output of the estimation process containing the correlation matrix.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plot_cfg = cfg.get("parameter_correlation_heatmap", {})

    correlation_matrix = estimation_output.correlations
    n_params = correlation_matrix.shape[0]

    # Default parameter names: Cowell propagation (position xyz, velocity xyz)
    default_names = [
        "x",
        "y",
        "z",
        "v\u2093",
        "v_y",
        "v_z",
    ]
    parameter_names = plot_cfg.get("parameter_names", default_names)

    # Extend or truncate parameter_names to match matrix size
    if len(parameter_names) < n_params:
        parameter_names = list(parameter_names) + [
            f"p{i}" for i in range(len(parameter_names), n_params)
        ]
    else:
        parameter_names = parameter_names[:n_params]

    fig_width = plot_cfg.get("figure", {}).get("width", 8.0)
    fig_height = plot_cfg.get("figure", {}).get("height", 8.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cax = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="Correlation")

    # Set tick labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax.set_yticklabels(parameter_names)

    # Annotate cells with correlation values
    annotate = plot_cfg.get("annotation", True)
    if annotate:
        decimal_places = plot_cfg.get("decimal_places", 2)
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
