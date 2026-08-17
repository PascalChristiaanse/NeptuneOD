import matplotlib.pyplot as plt
import numpy as np
from tudatpy.estimation import estimation_analysis as est_an


def plot_residual_rms_per_iteration(cfg, estimation_output: est_an.EstimationOutput):
    """
    Plots the RMS of the residuals per iteration of the estimation process.

    Parameters
    ----------
    estimation_output : est_an.EstimationOutput
        The output of the estimation process containing the residuals per iteration.
    """

    rms_per_iteration = []
    for iteration in range(estimation_output.residual_history.shape[1]):
        residuals = estimation_output.residual_history[:, iteration]
        rms = np.sqrt(np.mean(residuals**2))
        rms_per_iteration.append(rms)

    fig, ax = plt.subplots()
    ax.plot(rms_per_iteration, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS of Residuals")
    ax.set_title("RMS of Residuals per Iteration")
    ax.grid()
    return fig, ax
