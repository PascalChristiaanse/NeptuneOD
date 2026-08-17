import matplotlib.pyplot as plt
import numpy as np
from tudatpy.estimation import estimation_analysis as est_an


def plot_parameter_history_per_iteration(cfg, estimation_output: est_an.EstimationOutput):
    """
    Plots the parameter values against iteration number for each estimated parameter.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing optional 'parameter_history_per_iteration' settings.
        Supported keys:
            parameter_names : list of str
                Names for each parameter (used as tick labels and legend).
                Defaults to Cowell propagation parameter names.
            figure : dict
                'width' and 'height' for figure size.
            normalize : bool
                Whether to normalize parameters by their initial value (default True).
            y_label : str
                Custom label for the y-axis.

    estimation_output : est_an.EstimationOutput
        The output of the estimation process containing the parameter history.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plot_cfg = cfg.get("parameter_history_per_iteration", {})

    parameter_history = estimation_output.parameter_history
    # parameter_history: rows = parameters, columns = iterations (col 0 = pre-estimation)
    n_params = parameter_history.shape[0]
    n_iterations = parameter_history.shape[1]

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

    # Normalize each parameter by its initial value to show relative change
    # Use column 0 (pre-estimation) as reference
    normalize = plot_cfg.get("normalize", True)
    if normalize:
        reference = parameter_history[:, 0]
        # Avoid division by zero
        reference_safe = np.where(np.abs(reference) > 0, reference, 1.0)
        plot_data = parameter_history / reference_safe[:, np.newaxis]
        y_label = plot_cfg.get("y_label", "Relative Parameter Value (normalized to initial)")
    else:
        plot_data = parameter_history
        y_label = plot_cfg.get("y_label", "Parameter Value")

    fig_width = plot_cfg.get("figure", {}).get("width", 10.0)
    fig_height = plot_cfg.get("figure", {}).get("height", 6.0)
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
